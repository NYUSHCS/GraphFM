import os
import torch
import numpy as np

from models.GraphModels import _GraphModels
from .gssl.transductive_model import Model, GCNEncoder
from .gssl.transductive_model_arxiv import ArxivModel
from .gssl.tasks import evaluate_node_classification_acc

from torch_geometric.utils import to_undirected
import torch.nn.functional as F

from .gssl.loss import get_loss


class LPDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LPDecoder, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class _GBT(_GraphModels):
    def __init__(self, args, data):
        super(_GBT, self).__init__(args, data)

        self.dataset = args.dataset
        self.data = data

        if self.dataset == "ogbn-arxiv":
            self.model = ArxivModel
        else:
            self.model = Model

        self.predictor_lp = LPDecoder(args.emb_dim, args.decode_channels_lp, 1, args.num_layers_lp, args.dropout_lp)

        self.seeds = args.seeds
        self._use_pytorch_eval_model = False

    def train_net(self, input_dict):
        loss_name = input_dict["loss_name"]
        self.masks = input_dict["masks"]
        self.save_path = input_dict["save_path"]
        total_epochs = input_dict["total_epochs"]
        warmup_epochs = input_dict["warmup_epochs"]
        log_interval = input_dict["log_interval"]
        emb_dim = input_dict["emb_dim"]
        lr_base = input_dict["lr_base"]
        p_x = input_dict["p_x"]
        p_e = input_dict["p_e"]
        self.batch_size = input_dict["batch_size"]
        self.eval_batch_size = input_dict["eval_batch_size"]
        self.device = input_dict["device"]
        self.n_h = emb_dim
        self.num_classes = input_dict["num_classes"]
        self.y = input_dict["y"]
        self.split_edge = input_dict["split_edge"]
        self.dataset = input_dict["dataset"]
        self.pyg_data = input_dict["pyg_data"]
        self.predictor_lp = input_dict["predictor_lp"]

        self.data.edge_index = to_undirected(self.data.edge_index, self.data.num_nodes)

        model = self.model(feature_dim=self.data.x.size(-1),
            emb_dim=emb_dim,
            loss_name=loss_name,
            p_x=p_x,
            p_e=p_e,
            lr_base=lr_base,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs)

        loss, _ = model.fit(
            self.data,
            masks=self.masks,
        )

        self.emb_model = model

        return loss


    def nc_eval_net(self, h):
        from eval.node_classification import fit_logistic_regression
        final_acc, early_stp_acc = fit_logistic_regression(data=self.pyg_data, features=h, labels=self.y, data_random_seeds=[0],
                                                           dataset_name=self.dataset, device=self.device, mute=True)
        return np.mean(early_stp_acc)

    def lp_eval_net(self, h):
        from eval.link_prediction import perform_nn_link_eval
        results = perform_nn_link_eval(self.predictor_lp, h, self.data.x, self.split_edge,
                                       self.eval_batch_size, self.device, mute=True)
        return results

    @torch.no_grad()
    def nclustering_eval_net(self, h):
        from eval.node_clustering import clustering
        nmi, ari, _ = clustering(h, self.y, self.num_classes)
        return nmi, ari

    def embed(self, data):
        import copy
        tmp_encoder = copy.deepcopy(self.emb_model)
        return tmp_encoder.predict(data)

    def mem_speed_bench(self, input_dict):
        import time
        import json
        from utils import GB, MB, compute_tensor_bytes, get_memory_usage
        torch.cuda.empty_cache()
        data = input_dict["data"]
        optimizer = input_dict["optimizer"]
        device = input_dict["device"]
        saved_args = input_dict["saved_args"]
        log_path = input_dict["log_path"]
        masks = input_dict["masks"]
        loss_name = input_dict["loss_name"]
        total_epochs = input_dict["total_epochs"]
        warmup_epochs = input_dict["warmup_epochs"]
        log_interval = input_dict["log_interval"]
        emb_dim = input_dict["emb_dim"]
        lr_base = input_dict["lr_base"]
        p_x = input_dict["p_x"]
        p_e = input_dict["p_e"]
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
        init_mem = get_memory_usage(0, False)
        data_mem = init_mem - usage_dict["model_opt_usage"]
        usage_dict["data_mem"].append(data_mem)
        print("---> num_sampled_nodes: {}".format(data.x.shape[0]))
        print("data mem: %.2f MB" % (data_mem / MB))
        #out = self(batch.x, batch.edge_index)

        loss_fn = get_loss("barlow_twins")

        _encoder = GCNEncoder(
            in_dim=data.x.size(-1), out_dim=emb_dim
        ).to(device)

        z_a = _encoder(x=data.x,
                edge_index=data.edge_index)
        z_b = _encoder(x=data.x,
                edge_index=data.edge_index)

        loss = loss_fn(z_a=z_a, z_b=z_b)

        out = z_a

        before_backward = get_memory_usage(0, False)
        act_mem = before_backward - init_mem - compute_tensor_bytes([loss, out])
        usage_dict["act_mem"].append(act_mem)
        print("act mem: %.2f MB" % (act_mem / MB))
        #loss.backward()
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