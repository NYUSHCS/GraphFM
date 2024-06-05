import copy
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.functional import cosine_similarity
from .transforms import *

from torch.utils.data import DataLoader
from torch_geometric.nn import BatchNorm, GCNConv, Sequential


class BGRL(torch.nn.Module):
    r"""BGRL architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, layer_sizes, predictor):
        super().__init__()
        # online network
        self.layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'), )
            self.layers.append(BatchNorm(out_dim, momentum=0.99))
            self.layers.append(nn.PReLU())
        self.model = Sequential('x, edge_index', self.layers)

        self.predictor = predictor

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.model.parameters(), self.model.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def model_train(self, online_x, target_x):
        self.predictor.train()

        # forward online network
        online_y = self.model(online_x.x, online_x.edge_index)

        # prediction
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.model(target_x.x, target_x.edge_index).detach()

        return online_q, target_y

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        from tqdm import tqdm
        pbar = tqdm(total=x_all.size(0) * len(self.layers))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i, conv in enumerate(self.model):
            xs = []
            conv = conv.to(x_all.device)
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to(x_all.device)
                edge_index = batch.edge_index.to(x_all.device)
                if i == 0:
                    x = conv(x, edge_index)
                else:
                    x = conv(x)
                x = x[:batch.batch_size]
                xs.append(x.cpu())

                pbar.update(batch.batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

    def train_net(self, input_dict):
        self.device = input_dict["device"]
        self.data = input_dict["data"].to(self.device)
        optimizer = input_dict["optimizer"]
        transform_1 = input_dict["transform_1"]
        transform_2 = input_dict["transform_2"]
        lr_scheduler = input_dict["lr_scheduler"]
        mm_scheduler = input_dict["mm_scheduler"]
        epoch = input_dict["epoch"]
        seeds = input_dict["seeds"]
        num_eval_splits = input_dict["num_eval_splits"]
        self.save_path = input_dict["save_path"]
        self.batch_size = input_dict["batch_size"]
        self.eval_batch_size = input_dict["eval_batch_size"]
        self.num_classes = input_dict["num_classes"]
        self.y = input_dict["y"]
        train_loader = input_dict["train_loader"]
        self.subgraph_loader = input_dict["subgraph_loader"]
        self.split_edge = input_dict["split_edge"]
        self.dataset = input_dict["dataset"]
        self.pyg_data = input_dict["pyg_data"]
        self.predictor_lp = input_dict["predictor_lp"]

        self.seeds = seeds
        self.num_eval_splits = num_eval_splits

        lr = lr_scheduler.get(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        self.model = self.model.to(self.device)
        self.predictor = self.predictor.to(self.device)
        self.predictor_lp = self.predictor_lp.to(self.device)

        # update momentum
        mm = 1 - mm_scheduler.get(epoch)

        total_loss = 0
        for batch in train_loader:
            # forward
            optimizer.zero_grad()

            batch = batch.to(self.device)
            x1, x2 = transform_1(batch), transform_2(batch)

            q1, y2 = self.model_train(x1, x2)
            q2, y1 = self.model_train(x2, x1)

            loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
            loss.backward()

            # update online network
            optimizer.step()
            # update target network
            self.update_target_network(mm)

            total_loss += loss

        return total_loss / len(train_loader)

    
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

    def mem_speed_bench(self, input_dict):
        import time
        import json
        from utils import GB, MB, compute_tensor_bytes, get_memory_usage
        torch.cuda.empty_cache()
        optimizer = input_dict["optimizer"]
        train_loader = input_dict["train_loader"]
        device = input_dict["device"]
        saved_args = input_dict["saved_args"]
        log_path = input_dict["log_path"]
        transform_1 = input_dict["transform_1"]
        transform_2 = input_dict["transform_2"]
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
        for batch in train_loader:
            iter_start_time = time.time()
            torch.cuda.synchronize()
            optimizer.zero_grad()
            batch = batch.to(device)
            init_mem = get_memory_usage(0, False)
            data_mem = init_mem - usage_dict["model_opt_usage"]
            usage_dict["data_mem"].append(data_mem)
            print("---> num_sampled_nodes: {}".format(batch.x.shape[0]))
            print("data mem: %.2f MB" % (data_mem / MB))
            #out = self(batch.x, batch.edge_index)

            batch = batch.to(device)
            x1, x2 = transform_1(batch), transform_2(batch)

            self.predictor.train()

            # forward online network
            online_y = self.model(x1.x, x1.edge_index)

            # prediction
            q1 = self.predictor(online_y)

            # forward target network
            with torch.no_grad():
                y2 = self.model(x2.x, x2.edge_index).detach()

            # forward online network
            online_y = self.model(x2.x, x2.edge_index)

            # prediction
            q2 = self.predictor(online_y)

            # forward target network
            with torch.no_grad():
                y1 = self.model(x1.x, x1.edge_index).detach()

            loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(),
                                                                                             dim=-1).mean()
            out = torch.cat([y1, y2], dim=1)
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
            del out, loss, batch
        with open(
            "./{}/{}_mem_speed_log.json".format(log_path, saved_args["dataset"]), "w"
        ) as fp:
            info_dict = {**saved_args, **usage_dict}
            #del info_dict["device"]
            json.dump(info_dict, fp)
        exit()

def load_trained_encoder(encoder, ckpt_path, device):
    r"""Utility for loading the trained encoder."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint['model'], strict=True)
    return encoder.to(device)


