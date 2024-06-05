import os

from models.GraphModels import _GraphModels
from .gssl.transductive_model import augment
from .gssl.tasks import evaluate_node_classification_acc

from torch_geometric.utils import to_undirected
from torch_geometric.nn import Sequential
import torch.nn.functional as F

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch import nn

from torch_geometric import nn as tgnn

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

        self.feature_dim = self.data.x.size(-1)
        self.emb_dim = args.emb_dim
        self.lr_base = args.lr_base
        self.loss_name = "barlow_twins"
        self.warmup_epochs = args.warmup_epochs
        self.total_epochs = args.total_epochs

        self.layers = []
        if self.dataset == "ogbn-arxiv":
            self.layers.append((tgnn.GCNConv(self.feature_dim, self.emb_dim), 'x, edge_index -> x'))
            self.layers.append(nn.BatchNorm1d(self.emb_dim, momentum=0.01))
            self.layers.append(nn.PReLU())

            self.layers.append((tgnn.GCNConv(self.emb_dim, self.emb_dim), 'x, edge_index -> x'))
            self.layers.append(nn.BatchNorm1d(self.emb_dim, momentum=0.01))
            self.layers.append(nn.PReLU())

            self.layers.append((tgnn.GCNConv(self.emb_dim, self.emb_dim), 'x, edge_index -> x'))

        else:
            self.layers.append((tgnn.GCNConv(self.feature_dim, 2 * self.emb_dim), 'x, edge_index -> x'))
            self.layers.append(nn.BatchNorm1d(2 * self.emb_dim, momentum=0.01))  # same as `weight_decay = 0.99`
            self.layers.append(nn.PReLU())
            self.layers.append((tgnn.GCNConv(2 * self.emb_dim, self.emb_dim), 'x, edge_index -> x'))

        self.model = Sequential('x, edge_index', self.layers)

        self._loss_fn = get_loss(loss_name=self.loss_name)

        self._optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.lr_base,
            weight_decay=1e-5,
        )
        self._scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=self._optimizer,
            warmup_epochs=self.warmup_epochs,
            max_epochs=self.total_epochs,
        )


        self.predictor_lp = LPDecoder(self.emb_dim, args.decode_channels_lp, 1, args.num_layers_lp, args.dropout_lp)

        self.seeds = args.seeds
        self._use_pytorch_eval_model = False

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def train_net(self, input_dict):
        masks = input_dict["masks"]
        save_path = input_dict["save_path"]
        log_interval = input_dict["log_interval"]
        self._p_x = input_dict["p_x"]
        self._p_e = input_dict["p_e"]
        self.batch_size = input_dict["batch_size"]
        self.eval_batch_size = input_dict["eval_batch_size"]
        self.device = input_dict["device"]
        self.n_h = self.emb_dim
        self.num_classes = input_dict["num_classes"]
        self.y = input_dict["y"]
        train_loader = input_dict["train_loader"]
        self.subgraph_loader = input_dict["subgraph_loader"]
        self.split_edge = input_dict["split_edge"]
        self.dataset = input_dict["dataset"]
        self.pyg_data = input_dict["pyg_data"]
        self.predictor_lp = input_dict["predictor_lp"]

        self.data.edge_index = to_undirected(self.data.edge_index, self.data.num_nodes)

        self.masks = masks
        self.save_path = save_path

        total_loss = 0
        for batch in train_loader:
            loss, z = self.fit(
                batch,
                masks=masks,
            )

            total_loss += loss

        return total_loss / len(train_loader)

    def fit(self, data, masks,):
        data = data.to(self._device)
        self.model = self.model.to(self._device)

        self._optimizer.zero_grad()

        (x_a, ei_a), (x_b, ei_b) = augment(
            data=data, p_x=self._p_x, p_e=self._p_e,
        )

        z_a = self.model(x=x_a, edge_index=ei_a)
        z_b = self.model(x=x_b, edge_index=ei_b)

        loss = self._loss_fn(z_a=z_a, z_b=z_b)

        loss.backward()

        z = self.predict(data=data)

        self._optimizer.step()
        self._scheduler.step()

        return loss, z

    def predict(self, data):

        with torch.no_grad():
            z = self.model(data.x.to(self._device), data.edge_index.to(self._device))

            return z.cpu()

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
                if i in {0, 3, 6}:
                    x = conv(x, edge_index)
                elif i in {1, 2, 4, 5}:
                    x = conv(x)
                x = x[:batch.batch_size]
                xs.append(x.cpu())

                pbar.update(batch.batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

    
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
        data = input_dict["data"]
        optimizer = input_dict["optimizer"]
        train_loader = input_dict["train_loader"]
        device = input_dict["device"]
        saved_args = input_dict["saved_args"]
        log_path = input_dict["log_path"]
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

            self.model = self.model.to(device)

            optimizer.zero_grad()

            (x_a, ei_a), (x_b, ei_b) = augment(
                data=batch, p_x=p_x, p_e=p_e,
            )

            z_a = self.model(x=x_a, edge_index=ei_a)
            z_b = self.model(x=x_b, edge_index=ei_b)

            loss = self._loss_fn(z_a=z_a, z_b=z_b)

            out = self.model(batch.x.to(device), batch.edge_index.to(device))

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