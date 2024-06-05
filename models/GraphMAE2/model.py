import os
from itertools import chain

from typing import Optional
import torch
import torch.nn as nn
from functools import partial

from .gat import GAT

from .loss_func import sce_loss
from models.GraphMAE.utils import node_classification_evaluation
from models.GraphModels import _GraphModels


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True, **kwargs) -> nn.Module:
    if m_type in ("gat", "tsgat"):
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
            norm=norm,
            encoding=(enc_dec == "encoding"),
            **kwargs,
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden * 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden * 2, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


class GraphMAE2(_GraphModels):
    def __init__(self, args, data):
        super(GraphMAE2, self).__init__(args, data)
        self.data = data
        self._mask_rate = args.mask_rate
        self._remask_rate = args.remask_rate
        self._mask_method = args.mask_method
        self._alpha_l = args.alpha_l
        self._delayed_ema_epoch = args.delayed_ema_epoch

        self._encoder_type = args.encoder
        self._decoder_type = args.decoder
        self._drop_edge_rate = args.drop_edge_rate
        self._output_hidden_size = args.num_hidden
        self._momentum = args.momentum
        self._replace_rate = args.replace_rate
        self._num_remasking = args.num_remasking
        self._remask_method = args.remask_method
        self.activation = args.activation
        self.feat_drop = args.in_drop
        self.attn_drop = args.attn_drop
        self.negative_slope = args.negative_slope
        self.residual = args.residual
        self.norm = args.norm

        self.num_hidden = args.num_hidden
        self.nhead = args.num_heads
        self.nhead_out = args.num_out_heads

        self._token_rate = 1 - self._replace_rate
        self._lam = args.lam

        self.num_dec_layers = args.num_dec_layers
        self.zero_init = args.dataset in ("cora", "pubmed", "citeseer")

        self.lr_f = args.lr_f
        self.weight_decay_f = args.weight_decay_f
        self.max_epoch_f = args.max_epoch_f

        assert self.num_hidden % self.nhead == 0
        assert self.num_hidden % self.nhead_out == 0
        if self._encoder_type in ("gat",):
            enc_num_hidden = self.num_hidden // self.nhead
            enc_nhead = self.nhead
        else:
            enc_num_hidden = self.num_hidden
            enc_nhead = 1

        dec_in_dim = self.num_hidden
        dec_num_hidden = self.num_hidden // self.nhead if self._encoder_type in ("gat",) else self.num_hidden

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

        self.decoder = setup_module(
            m_type=self._decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=self.num_feats,
            nhead_out=self.nhead_out,
            num_layers=self.num_dec_layers,
            nhead=self.nhead,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.num_feats))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, self.num_hidden))

        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        if not self.zero_init:
            self.reset_parameters_for_token()

        # * setup loss function
        self.criterion = self.setup_loss_fn(args.loss_fn, args.alpha_l)

        self.projector = nn.Sequential(
            nn.Linear(self.num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, self.num_hidden),
        )
        self.projector_ema = nn.Sequential(
            nn.Linear(self.num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, self.num_hidden),
        )
        self.predictor = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.num_hidden, self.num_hidden)
        )

        self.encoder_ema = setup_module(
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
        self.encoder_ema.load_state_dict(self.encoder.state_dict())
        self.projector_ema.load_state_dict(self.projector.state_dict())

        for p in self.encoder_ema.parameters():
            p.requires_grad = False
            p.detach_()
        for p in self.projector_ema.parameters():
            p.requires_grad = False
            p.detach_()

        self.print_num_parameters()

    def print_num_parameters(self):
        num_encoder_params = [p.numel() for p in self.encoder.parameters() if p.requires_grad]
        num_decoder_params = [p.numel() for p in self.decoder.parameters() if p.requires_grad]
        num_params = [p.numel() for p in self.parameters() if p.requires_grad]

        print(
            f"num_encoder_params: {sum(num_encoder_params)}, num_decoder_params: {sum(num_decoder_params)}, num_params_in_total: {sum(num_params)}")

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            print(f"=== Use mse_loss ===")
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            print(f"=== Use sce_loss and alpha_l={alpha_l} ===")
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def train_net(self, input_dict):
        self.device = input_dict["device"]
        self.x = input_dict["x"].to(self.device)
        self.dataset = input_dict["dataset"]
        targets = input_dict["targets"]
        linear_prob = input_dict["linear_prob"]
        optimizer = input_dict["optimizer"]
        save_path = input_dict["save_path"]
        epoch = input_dict["epoch"]
        drop_g1 = input_dict["drop_g1"]
        drop_g2 = input_dict["drop_g2"]
        self.dgl_data = input_dict["dgl_data"]
        self.num_classes = input_dict["num_classes"]
        self.y = input_dict["y"]
        self.pyg_data = input_dict["pyg_data"]
        self.eval_batch_size = input_dict["eval_batch_size"]
        self.split_edge = input_dict["split_edge"]
        self.predictor_lp = input_dict["predictor_lp"]

        self.data = self.data.to(self.device)
        self.linear_prob = linear_prob
        self.save_path = save_path

        optimizer.zero_grad()
        loss = self.mask_attr_prediction(self.data.to(self.device), self.x.to(self.device), targets, epoch, drop_g1, drop_g2)

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

    def mask_attr_prediction(self, g, x, targets, epoch, drop_g1=None, drop_g2=None):
        self.encoder.train()

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        use_g = drop_g1 if drop_g1 is not None else g

        enc_rep = self.encoder(use_g, use_x, )

        with torch.no_grad():
            drop_g2 = drop_g2 if drop_g2 is not None else g
            latent_target = self.encoder_ema(drop_g2, x, )
            if targets is not None:
                latent_target = self.projector_ema(latent_target[targets])
            else:
                latent_target = self.projector_ema(latent_target[keep_nodes])

        if targets is not None:
            latent_pred = self.projector(enc_rep[targets])
            latent_pred = self.predictor(latent_pred)
            loss_latent = sce_loss(latent_pred, latent_target, 1)
        else:
            latent_pred = self.projector(enc_rep[keep_nodes])
            latent_pred = self.predictor(latent_pred)
            loss_latent = sce_loss(latent_pred, latent_target, 1)

        # ---- attribute reconstruction ----
        origin_rep = self.encoder_to_decoder(enc_rep)

        loss_rec_all = 0
        if self._remask_method == "random":
            for i in range(self._num_remasking):
                rep = origin_rep.clone()
                rep, remask_nodes, rekeep_nodes = self.random_remask(use_g, rep, self._remask_rate)
                recon = self.decoder(pre_use_g, rep)

                x_init = x[mask_nodes]
                x_rec = recon[mask_nodes]
                loss_rec = self.criterion(x_init, x_rec)
                loss_rec_all += loss_rec
            loss_rec = loss_rec_all
        elif self._remask_method == "fixed":
            rep = self.fixed_remask(g, origin_rep, mask_nodes)
            x_rec = self.decoder(pre_use_g, rep)[mask_nodes]
            x_init = x[mask_nodes]
            loss_rec = self.criterion(x_init, x_rec)
        else:
            raise NotImplementedError

        loss = loss_rec + self._lam * loss_latent

        if epoch >= self._delayed_ema_epoch:
            self.ema_update()

        return loss

    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
                # m = momentum_schedule[it]  # momentum parameter
                m = self._momentum
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        update(self.encoder, self.encoder_ema)
        update(self.projector, self.projector_ema)

    def embed(self, data):
        import copy
        tmp_encoder = copy.deepcopy(self.encoder).eval().cpu()
        return tmp_encoder(data.cpu(), data.ndata['feat'].cpu())

    def get_encoder(self):
        # self.encoder.reset_classifier(out_size)
        return self.encoder

    def reset_encoder(self, out_size):
        self.encoder.reset_classifier(out_size)

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

    def output_grad(self):
        grad_dict = {}
        for n, p in self.named_parameters():
            if p.grad is not None:
                grad_dict[n] = p.grad.abs().mean().item()
        return grad_dict

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def random_remask(self, g, rep, remask_rate=0.5):

        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes

    def fixed_remask(self, g, rep, masked_nodes):
        rep[masked_nodes] = 0
        return rep

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
        targets = input_dict["targets"]
        drop_g1 = input_dict["drop_g1"]
        drop_g2 = input_dict["drop_g2"]
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

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = data.clone()
        pre_use_g, use_x = use_g, out_x
        #pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(data, x, self._mask_rate)
        use_g = drop_g1 if drop_g1 is not None else data

        enc_rep = self.encoder(use_g, use_x, )

        with torch.no_grad():
            drop_g2 = drop_g2 if drop_g2 is not None else data
            latent_target = self.encoder_ema(drop_g2, x, )
            if targets is not None:
                latent_target = self.projector_ema(latent_target[targets])
            else:
                latent_target = self.projector_ema(latent_target[keep_nodes])

        if targets is not None:
            latent_pred = self.projector(enc_rep[targets])
            latent_pred = self.predictor(latent_pred)
            loss_latent = sce_loss(latent_pred, latent_target, 1)
        else:
            latent_pred = self.projector(enc_rep[keep_nodes])
            latent_pred = self.predictor(latent_pred)
            loss_latent = sce_loss(latent_pred, latent_target, 1)

        # ---- attribute reconstruction ----
        origin_rep = self.encoder_to_decoder(enc_rep)

        loss_rec_all = 0
        if self._remask_method == "random":
            for i in range(self._num_remasking):
                rep = origin_rep.clone()
                rep, remask_nodes, rekeep_nodes = self.random_remask(use_g, rep, self._remask_rate)
                recon = self.decoder(pre_use_g, rep)

                x_init = x[mask_nodes]
                x_rec = recon[mask_nodes]
                loss_rec = self.criterion(x_init, x_rec)
                loss_rec_all += loss_rec
            loss_rec = loss_rec_all
        elif self._remask_method == "fixed":
            rep = self.fixed_remask(data, origin_rep, mask_nodes)
            x_rec = self.decoder(pre_use_g, rep)[mask_nodes]
            x_init = x[mask_nodes]
            loss_rec = self.criterion(x_init, x_rec)
        else:
            raise NotImplementedError

        loss = loss_rec + self._lam * loss_latent
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