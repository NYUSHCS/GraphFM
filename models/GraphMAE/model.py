import os
from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss
from .utils import create_norm, drop_edge, node_classification_evaluation
from models.GraphModels import _GraphModels

from torch.utils.data import DataLoader
import torch.nn.functional as F
import dgl.function as fn


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


class GraphMAE(_GraphModels):
    def __init__(self, args, data):
        super(GraphMAE, self).__init__(args, data)
        self.data = data
        self._mask_rate = args.mask_rate
        self.batch_type = args.batch_type

        self._encoder_type = args.encoder
        self._decoder_type = args.decoder
        self._drop_edge_rate = args.drop_edge_rate
        self._output_hidden_size = args.num_hidden
        self._concat_hidden = args.concat_hidden

        self._replace_rate = args.replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        self.num_hidden = args.num_hidden
        self.nhead = args.num_heads
        self.nhead_out = args.num_out_heads

        self.activation = args.activation
        self.feat_drop = args.in_drop
        self.attn_drop = args.attn_drop
        self.negative_slope = args.negative_slope
        self.residual = args.residual
        self.norm = args.norm

        self.lr_f = args.lr_f
        self.weight_decay_f = args.weight_decay_f
        self.max_epoch_f = args.max_epoch_f

        assert self.num_hidden % self.nhead == 0
        assert self.num_hidden % self.nhead_out == 0
        if self._encoder_type in ("gat", "dotgat"):
            enc_num_hidden = self.num_hidden // self.nhead
            enc_nhead = self.nhead
        else:
            enc_num_hidden = self.num_hidden
            enc_nhead = 1

        dec_in_dim = self.num_hidden
        dec_num_hidden = self.num_hidden // self.nhead_out if self._decoder_type in ("gat", "dotgat") else self.num_hidden

        # build encoder
        self.encoder = setup_module(
            m_type=self._encoder_type,
            enc_dec="encoding",
            in_dim=self.num_feats,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=self.num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=self._decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=self.num_feats,
            num_layers=1,
            nhead=self.nhead,
            nhead_out=self.nhead_out,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.num_feats))
        if args.concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * self.num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(args.loss_fn, args.alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def train_net(self, input_dict):
        self.device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        linear_prob = input_dict["linear_prob"]
        save_path = input_dict["save_path"]
        self.x = input_dict["x"].to(self.device)
        self.y = input_dict["y"].to(self.device)
        self.pyg_data = input_dict["pyg_data"]
        self.batch_size = input_dict["batch_size"]
        self.dgl_data = input_dict["dgl_data"]
        self.num_classes = input_dict["num_classes"]
        self.eval_batch_size = input_dict["eval_batch_size"]
        self.split_edge = input_dict["split_edge"]
        self.dataset = input_dict["dataset"]
        self.predictor_lp = input_dict["predictor_lp"]

        self.data = self.data.to(self.device)
        self.linear_prob = linear_prob
        self.save_path = save_path

        optimizer.zero_grad()
        loss = self.mask_attr_prediction(self.data, self.x)
        loss.backward()
        optimizer.step()

        return loss

    def nc_eval_net(self, h):
        from eval.node_classification import fit_logistic_regression
        import numpy as np
        final_acc, early_stp_acc = fit_logistic_regression(data=self.pyg_data, features=h, labels=self.y, data_random_seeds=[0],
                                                           dataset_name=self.dataset, device=self.device, mute=True)
        return np.mean(early_stp_acc)

    
    def lp_eval_net(self, h):
        results = self.lp_test(h.to(self.device), self.predictor_lp, self.eval_batch_size)
        return results
    
    @torch.no_grad()
    def lp_test(self, h, score_func, batch_size):
        from eval.link_prediction import get_metric_score
        from eval.utils import test_edge_mae as test_edge

        pos_train_pred = test_edge(score_func, self.split_edge['train']['edge'], h, batch_size)

        neg_valid_pred = test_edge(score_func, self.split_edge['valid']['edge_neg'], h, batch_size)

        pos_valid_pred = test_edge(score_func, self.split_edge['valid']['edge'], h, batch_size)

        pos_test_pred = test_edge(score_func, self.split_edge['test']['edge'], h, batch_size)

        neg_test_pred = test_edge(score_func, self.split_edge['test']['edge_neg'], h, batch_size)

        pos_train_pred = torch.flatten(pos_train_pred)
        neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred), torch.flatten(pos_valid_pred)
        pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)

        print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(),
              neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())

        result = get_metric_score(pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

        #        score_emb = [pos_valid_pred.cpu(), neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x.cpu()]

        return result

    @torch.no_grad()
    def nclustering_eval_net(self, h):
        from eval.node_clustering import clustering
        nmi, ari, _ = clustering(h, self.y, self.num_classes)
        return nmi, ari


    def mask_attr_prediction(self, g, x):
        self.encoder.train()

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)

        return loss


    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

    def embed(self, data):
        import copy
        tmp_encoder = copy.deepcopy(self.encoder).eval().cpu()
        return tmp_encoder(data.cpu(), data.ndata['feat'].cpu())

    def mem_speed_bench(self, input_dict):
        import time
        import json
        from utils import GB, MB, compute_tensor_bytes, get_memory_usage
        torch.cuda.empty_cache()
        data = input_dict["data"]
        x = input_dict["x"]
        optimizer = input_dict["optimizer"]
        device = input_dict["device"]
        saved_args = input_dict["saved_args"]
        log_path = input_dict["log_path"]
        torch.cuda.empty_cache()
        model_opt_usage = get_memory_usage(0, False)
        usage_dict = {
            "model_opt_usage": model_opt_usage,
            "data_mem": [],
            "act_mem": [],
            "peak_mem": [],
            "duration": [],
        }
        print(
            "model + optimizer only, mem: %.2f MB"
            % (usage_dict["model_opt_usage"] / MB)
        )
        epoch_start_time = time.time()
        torch.cuda.synchronize()

        iter_start_time = time.time()
        torch.cuda.synchronize()
        optimizer.zero_grad()
        data = data.to(device)
        x = x.to(device)
        init_mem = get_memory_usage(0, False)
        data_mem = init_mem - usage_dict["model_opt_usage"]
        usage_dict["data_mem"].append(data_mem)
        print("---> num_sampled_nodes: {}".format(x.shape[0]))
        print("data mem: %.2f MB" % (data_mem / MB))
        #out = self(batch.x, batch.edge_index)


        num_nodes = data.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(self._mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = data.clone()

        pre_use_g, use_x = use_g, out_x

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)

        out = self.encoder(data, x)
        before_backward = get_memory_usage(0, False)
        act_mem = before_backward - init_mem - compute_tensor_bytes([loss, out])
        usage_dict["act_mem"].append(act_mem)
        print("act mem: %.2f MB" % (act_mem / MB))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        iter_end_time = time.time()
        duration = iter_end_time - iter_start_time
        print("duration: %.4f sec" % duration)
        usage_dict["duration"].append(duration)
        peak_usage = torch.cuda.max_memory_allocated(0)
        usage_dict["peak_mem"].append(peak_usage)
        print(f"peak mem usage: {peak_usage / MB}")
        torch.cuda.empty_cache()
        del out, loss, data

        with open(
            "./{}/{}_mem_speed_log.json".format(log_path, saved_args["dataset"]), "w"
        ) as fp:
            info_dict = {**saved_args, **usage_dict}
            #del info_dict["device"]
            json.dump(info_dict, fp)
        exit()