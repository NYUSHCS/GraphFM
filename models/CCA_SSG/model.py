import os

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

import numpy as np
import torch as th
from .aug import random_aug


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)

        return x

class CCA_SSG(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_mlp = False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
        self.n_h = hid_dim

    def get_embedding(self, graph, feat):
        out = self.backbone(graph, feat)
        return out.detach()

    def model_train(self, graph1, feat1, graph2, feat2, save_path, device):
        self.backbone.train()

        self.backbone = self.backbone.to(device)
        h1 = self.backbone(graph1, feat1)
        h2 = self.backbone(graph2, feat2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        return z1, z2, self.backbone

    def train_net(self, input_dict):
        dataset = input_dict["dataset"]
        self.device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        data = input_dict["data"]
        self.pyg_data = input_dict["pyg_data"]
        x = input_dict["x"]
        y = input_dict["y"]
        split_masks = input_dict["split_masks"]
        dfr = input_dict["dfr"]
        der = input_dict["der"]
        lambd = input_dict["lambd"]
        N = input_dict["N"]
        lr2 = input_dict["lr2"]
        wd2 = input_dict["wd2"]
        num_class = input_dict["num_class"]
        save_path = input_dict["save_path"]
        self.dgl_data = input_dict["dgl_data"]
        self.batch_size = input_dict["batch_size"]
        self.eval_batch_size = input_dict["eval_batch_size"]
        self.num_classes = input_dict["num_classes"]
        self.split_edge = input_dict["split_edge"]
        self.dataset = input_dict["dataset"]
        self.predictor_lp = input_dict["predictor_lp"]

        self.data = data
        self.x = x
        self.y = y
        self.num_class = num_class
        self.lr2 = lr2
        self.wd2 = wd2
        self.split_masks = split_masks
        self.save_path = save_path

        optimizer.zero_grad()

        graph1, feat1 = random_aug(data, x, dfr, der)
        graph2, feat2 = random_aug(data, x, dfr, der)

        graph1 = graph1.add_self_loop()
        graph2 = graph2.add_self_loop()

        graph1 = graph1.to(self.device)
        graph2 = graph2.to(self.device)

        feat1 = feat1.to(self.device)
        feat2 = feat2.to(self.device)

        z1, z2, emb_model = self.model_train(graph1, feat1, graph2, feat2, save_path, self.device)

        c = th.mm(z1.T, z2)
        c1 = th.mm(z1.T, z1)
        c2 = th.mm(z2.T, z2)

        c = c / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = -th.diagonal(c).sum()
        iden = th.tensor(np.eye(c.shape[0])).to(self.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + lambd * (loss_dec1 + loss_dec2)

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
        from eval.link_prediction import perform_nn_link_eval
        results = perform_nn_link_eval(self.predictor_lp, h, self.x, self.split_edge,
                                       self.eval_batch_size, self.device, mute=True)
        return results

    @th.no_grad()
    def nclustering_eval_net(self, h):
        from eval.node_clustering import clustering
        nmi, ari, _ = clustering(h, self.y, self.num_classes)
        return nmi, ari

    def embed(self, data):
        import copy
        tmp_encoder = copy.deepcopy(self.backbone).eval().cpu()
        return tmp_encoder(data.cpu(), data.ndata['feat'].cpu())

    def mem_speed_bench(self, input_dict):
        import time
        import json
        from utils import GB, MB, compute_tensor_bytes, get_memory_usage
        th.cuda.empty_cache()
        data = input_dict["data"]
        x = input_dict["x"]
        optimizer = input_dict["optimizer"]
        device = input_dict["device"]
        saved_args = input_dict["saved_args"]
        log_path = input_dict["log_path"]
        lambd = input_dict["lambd"]
        N = input_dict["N"]
        dfr = input_dict["dfr"]
        der = input_dict["der"]
        th.cuda.empty_cache()
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
        th.cuda.synchronize()

        iter_start_time = time.time()
        th.cuda.synchronize()
        optimizer.zero_grad()
        data = data.to(device)
        init_mem = get_memory_usage(0, False)
        data_mem = init_mem - usage_dict["model_opt_usage"]
        usage_dict["data_mem"].append(data_mem)
        print("---> num_sampled_nodes: {}".format(x.shape[0]))
        print("data mem: %.2f MB" % (data_mem / MB))
        #out = self(batch.x, batch.edge_index)

        data = data.to("cpu")
        graph1, feat1 = random_aug(data, x, dfr, der)
        graph2, feat2 = random_aug(data, x, dfr, der)

        graph1 = graph1.add_self_loop()
        graph2 = graph2.add_self_loop()

        graph1 = graph1.to(device)
        graph2 = graph2.to(device)

        feat1 = feat1.to(device)
        feat2 = feat2.to(device)

        self.backbone.train()

        self.backbone = self.backbone.to(device)
        h1 = self.backbone(graph1, feat1)
        h2 = self.backbone(graph2, feat2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        c = th.mm(z1.T, z2)
        c1 = th.mm(z1.T, z1)
        c2 = th.mm(z2.T, z2)

        c = c / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = -th.diagonal(c).sum()
        iden = th.tensor(np.eye(c.shape[0])).to(device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + lambd * (loss_dec1 + loss_dec2)
        out = th.cat([z1, z2], dim=1)

        before_backward = get_memory_usage(0, False)
        act_mem = before_backward - init_mem - compute_tensor_bytes([loss, out])
        usage_dict["act_mem"].append(act_mem)
        print("act mem: %.2f MB" % (act_mem / MB))
        loss.backward()
        optimizer.step()
        th.cuda.synchronize()
        iter_end_time = time.time()
        duration = iter_end_time - iter_start_time
        print("duration: %.4f sec" % duration)
        usage_dict["duration"].append(duration)
        peak_usage = th.cuda.max_memory_allocated(0)
        usage_dict["peak_mem"].append(peak_usage)
        print(f"peak mem usage: {peak_usage / MB}")
        th.cuda.empty_cache()
        del out, loss, data

        with open(
            "./{}/{}_mem_speed_log.json".format(log_path, saved_args["dataset"]), "w"
        ) as fp:
            info_dict = {**saved_args, **usage_dict}
            #del info_dict["device"]
            json.dump(info_dict, fp)
        exit()