from typing import Optional
import os

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj
from .eval import log_regression, MulticlassEvaluator
from .utils import drop_feature, drop_edge_weighted, drop_feature_weighted_2

from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class GRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden

    def model_train(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def train_net(self, input_dict):
        self.data = input_dict["data"]
        self.dataset = input_dict["dataset"]
        self.device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        drop_scheme = input_dict["drop_scheme"]
        drop_weights = input_dict["drop_weights"]
        feature_weights = input_dict["feature_weights"]
        drop_feature_rate_1 = input_dict["drop_feature_rate_1"]
        drop_feature_rate_2 = input_dict["drop_feature_rate_2"]
        self.split_masks = input_dict["split_masks"]
        self.save_path = input_dict["save_path"]
        self.batch_size = input_dict["batch_size"]
        self.eval_batch_size = input_dict["eval_batch_size"]
        self.num_classes = input_dict["num_classes"]
        self.y = input_dict["y"]
        self.pyg_data = input_dict["pyg_data"]
        self.split_edge = input_dict["split_edge"]
        self.predictor_lp = input_dict["predictor_lp"]

        self.encoder.train()
        optimizer.zero_grad()

        def drop_edge(idx: int):
            if drop_scheme == 'uniform':
                return dropout_adj(self.data.edge_index.to(self.device), p=input_dict[f"drop_edge_rate_{idx}"])[0]
            elif drop_scheme in ['degree', 'evc', 'pr']:
                return drop_edge_weighted(self.data.edge_index.to(self.device), drop_weights, p=input_dict[f"drop_edge_rate_{idx}"],
                                          threshold=0.7)
            else:
                raise Exception(f'undefined drop scheme: {drop_scheme}')

        edge_index_1 = drop_edge(1)
        edge_index_2 = drop_edge(2)

        x_1 = drop_feature(self.data.x.to(self.device), drop_feature_rate_1)
        x_2 = drop_feature(self.data.x.to(self.device), drop_feature_rate_2)

        if drop_scheme in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted_2(self.data.x.to(self.device), feature_weights, drop_feature_rate_1)
            x_2 = drop_feature_weighted_2(self.data.x.to(self.device), feature_weights, drop_feature_rate_2)

        z1 = self.model_train(x_1.to(self.device), edge_index_1)
        z2 = self.model_train(x_2.to(self.device), edge_index_2)

        loss = self.loss(z1, z2)
        loss.backward()
        optimizer.step()

        return loss.item()

    def nc_eval_net(self, h):
        from eval.node_classification import fit_logistic_regression
        import numpy as np
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
        tmp_encoder = copy.deepcopy(self.encoder).eval().cpu()
        return tmp_encoder(data.x.cpu(), data.edge_index.cpu())

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
        drop_scheme = input_dict["drop_scheme"]
        drop_weights = input_dict["drop_weights"]
        feature_weights = input_dict["feature_weights"]
        drop_feature_rate_1 = input_dict["drop_feature_rate_1"]
        drop_feature_rate_2 = input_dict["drop_feature_rate_2"]
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

        def drop_edge(idx: int):
            if drop_scheme == 'uniform':
                return dropout_adj(data.edge_index.to(device), p=input_dict[f"drop_edge_rate_{idx}"])[0]
            elif drop_scheme in ['degree', 'evc', 'pr']:
                return drop_edge_weighted(data.edge_index.to(device), drop_weights, p=input_dict[f"drop_edge_rate_{idx}"],
                                          threshold=0.7)
            else:
                raise Exception(f'undefined drop scheme: {drop_scheme}')

        edge_index_1 = drop_edge(1)
        edge_index_2 = drop_edge(2)

        x_1 = drop_feature(data.x.to(device), drop_feature_rate_1)
        x_2 = drop_feature(data.x.to(device), drop_feature_rate_2)

        if drop_scheme in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted_2(data.x.to(device), feature_weights, drop_feature_rate_1)
            x_2 = drop_feature_weighted_2(data.x.to(device), feature_weights, drop_feature_rate_2)

        z1 = self.encoder(x_1.to(device), edge_index_1)
        z2 = self.encoder(x_2.to(device), edge_index_2)

        h1 = F.elu(self.fc1(z1))
        h1 = self.fc2(h1)
        h2 = F.elu(self.fc1(z2))
        h2 = self.fc2(h2)

        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(h1, h1))
        between_sim = f(self.sim(h1, h2))
        l1 = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

        refl_sim = f(self.sim(h2, h2))
        between_sim = f(self.sim(h2, h1))
        l2 = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

        ret = (l1 + l2) * 0.5
        loss = ret.mean()

        out = self.encoder(data.x, data.edge_index)

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