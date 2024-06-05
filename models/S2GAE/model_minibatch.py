import os

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from models.GraphModels import _GraphModels
from .utils import *

from sklearn.metrics import roc_auc_score

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(Encoder, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx


class NC_Decoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout):
        super(NC_Decoder, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class LP_Decoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout, de_v='v1'):
        super(LP_Decoder, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        if de_v == 'v1':
            self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels * n_layer, in_channels * n_layer))
            self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class S2GAE(_GraphModels):
    def __init__(self, args, data):
        super(S2GAE, self).__init__(args, data)

        self.batch_size = args.batch_size
        self.mask_type = args.mask_type
        self.mask_ratio = args.mask_ratio
        self.dim_decode = args.decode_channels
        self.decode_layers = args.decode_layers
        self.num_feats = data.num_features
        self.dim_hidden = args.dim_hidden
        #self.model = Encoder(self.num_feats, self.dim_hidden, self.dim_hidden, self.num_layers, self.dropout)

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(self.num_feats, self.dim_hidden, cached=False, add_self_loops=False))
        for _ in range(self.num_layers - 2):
            self.convs.append(
                GCNConv(self.dim_hidden, self.dim_hidden, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(self.dim_hidden, self.dim_hidden, cached=False, add_self_loops=False))

    def model_train(self, model, predictor, x, edge_index, optimizer):
        model.train()
        predictor.train()

        total_loss = total_examples = 0
        if self.mask_type == 'um':
            adj, _, pos_train_edge = edgemask_um(self.mask_ratio, edge_index, x.device, x.shape[0])
        else:
            adj, _, pos_train_edge = edgemask_dm(self.mask_ratio, edge_index, x.device, x.shape[0])
        adj = adj.to(x.device)

        for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size,
                               shuffle=True):
            optimizer.zero_grad()

            #h = model(x, adj)

            h = []
            for conv in model[:-1]:
                x_t = conv(x, adj)
                x_t = F.relu(x_t)
                x_t = F.dropout(x_t, p=self.dropout, training=self.training)
                h.append(x_t)
            x_t = model[-1](x_t, adj)
            h.append(F.relu(x_t))

            edge = pos_train_edge[perm].t()

            pos_out = predictor(h, edge)
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            # Just do some trivial random sampling.
            edge = torch.randint(0, x.shape[0], edge.size(), dtype=torch.long,
                                 device=x.device)
            neg_out = predictor(h, edge)
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            loss = pos_loss + neg_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

            optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples, model, predictor

    def test(self, predictor, h, x, pos_test_edge, neg_test_edge, batch_size):
        predictor.eval()

        pos_test_edge = pos_test_edge.to(x.device)
        neg_test_edge = neg_test_edge.to(x.device)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
            edge = pos_test_edge[perm].t()
            pos_test_preds += [predictor(h, edge).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            neg_test_preds += [predictor(h, edge).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

        test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
        test_true = torch.cat([torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)], dim=0)
        test_pred, test_true = test_pred.detach().numpy(), test_true.detach().numpy()
        test_auc = roc_auc_score(test_true, test_pred)
        return test_auc

    @torch.no_grad()
    def lp_test(self, h, score_func, batch_size):
        from eval.link_prediction import get_metric_score
        h = [h_.to(self.device) for h_ in h]

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

    def train_net(self, input_dict):
        device = input_dict["device"]
        data = input_dict["data"].to(device)
        x = input_dict["x"].to(device)
        split_masks = input_dict["split_masks"]
        optimizer = input_dict["optimizer"]
        pos_test_edge = input_dict["test_edge"]
        neg_test_edge = input_dict["test_edge_neg"]
        self.save_path = input_dict["save_path"]
        self.batch_size = input_dict["batch_size"]
        self.eval_batch_size = input_dict["eval_batch_size"]
        self.num_classes = input_dict["num_classes"]
        self.y = input_dict["y"]
        self.batch_size = input_dict["batch_size"]
        self.split_edge = input_dict["split_edge"]
        train_loader = input_dict["train_loader"]
        self.subgraph_loader = input_dict["subgraph_loader"]
        self.dataset = input_dict["dataset"]
        self.pyg_data = input_dict["pyg_data"]
        self.predictor_lp = input_dict["predictor_lp"]

        self.data = data
        self.x = x
        self.split_masks = split_masks
        self.pos_test_edge = pos_test_edge
        self.neg_test_edge = neg_test_edge
        self.device = device

        self.predictor_lp.reset_parameters()

        total_loss = []
        for batch in train_loader:
            if batch.is_undirected():
                edge_index_batch = batch.edge_index
            else:
                edge_index_batch = to_undirected(batch.edge_index)
            edge_index, _, _ = do_edge_split_nc(edge_index_batch, batch.x.shape[0])
            loss, model, predictor = self.model_train(self.convs.to(device), self.predictor_lp.to(device), batch.x.to(device), edge_index, optimizer)
            total_loss.append(loss)
            if loss == min(total_loss):
                self.emb_predictor = predictor

        return np.mean(total_loss)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        from tqdm import tqdm
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        xx = []
        for i, conv in enumerate(self.convs):
            xs = []
            conv = conv.to(x_all.device)
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to(x_all.device)
                edge_index = batch.edge_index.to(x_all.device)
                if self.mask_type == 'um':
                    adj, _, pos_train_edge = edgemask_um(self.mask_ratio, edge_index, self.device, x.shape[0])
                else:
                    adj, _, pos_train_edge = edgemask_dm(self.mask_ratio, edge_index, self.device, x.shape[0])
                adj = adj.to(x_all.device)
                x = conv(x, adj)
                x = x[:batch.batch_size]
                x = F.relu(x)
                if i != len(self.convs) - 1:
                    x = F.dropout(x, p=self.dropout, training=self.training)

                xs.append(x)

                pbar.update(batch.batch_size)

            x_all = torch.cat(xs, dim=0)
            xx.append(x_all)

        pbar.close()

        return xx

    def nc_eval_net(self, h):

        from eval.node_classification import fit_logistic_regression
        from .utils import extract_feature_list_layer2
        import numpy as np
        representations = [representation.to(self.device) for representation in h]
        feature = [feature_.detach() for feature_ in representations]

        feature_list = extract_feature_list_layer2(feature)
        final_acc_list, early_stp_acc_list = [], []

        for i, feature_tmp in enumerate(feature_list):
            final_acc, early_stp_acc = fit_logistic_regression(data=self.pyg_data, features=feature_tmp, labels=self.y, data_random_seeds=[0],
                                                           dataset_name=self.dataset, device=self.device, mute=True)
        
            final_acc_list.append(final_acc)
            early_stp_acc_list.append(early_stp_acc)

        final_acc, early_stp_acc = np.mean(final_acc_list), np.mean(early_stp_acc_list)
        return early_stp_acc


    def lp_eval_net(self, h):
        results = self.lp_test(h, self.predictor_lp.to(self.device), self.eval_batch_size)
        return results

    @torch.no_grad()
    def nclustering_eval_net(self, h):
        from eval.node_clustering import clustering
        emb = torch.cat(h, dim=1)
        nmi, ari, _ = clustering(emb, self.y, self.num_classes)
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
        predictor_lp = input_dict["predictor_lp"].to(device)
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

            edge_index = batch.edge_index.to(device)
            if self.mask_type == 'um':
                adj, _, pos_train_edge = edgemask_um(self.mask_ratio, edge_index, device, batch.x.shape[0])
            else:
                adj, _, pos_train_edge = edgemask_dm(self.mask_ratio, edge_index, device, batch.x.shape[0])
            adj = adj.to(device)
            out = []
            for conv in self.convs[:-1]:
                x_t = conv(batch.x, adj)
                x_t = F.relu(x_t)
                x_t = F.dropout(x_t, p=self.dropout, training=self.training)
                out.append(x_t)
            x_t = self.convs[-1](x_t, adj)
            out.append(F.relu(x_t))

            edge = pos_train_edge.t()
            pos_out = predictor_lp(out, edge)
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            # Just do some trivial random sampling.
            edge = torch.randint(0, batch.x.shape[0], edge.size(), dtype=torch.long,
                                 device=batch.x.device)
            neg_out = predictor_lp(out, edge)
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            loss = pos_loss + neg_loss

            out = torch.cat(out, dim=1)
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


from torch.utils.data import DataLoader
def test_edge(score_func, input_data, h, batch_size):
    # input_data  = input_data.transpose(1, 0)
    # with torch.no_grad():
    preds = []
    for perm in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()

        preds += [score_func(h, edge).cpu()]

    pred_all = torch.cat(preds, dim=0)

    return pred_all

def dis_fun(x, c):
    xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
    cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
    xx_cc = xx + cc
    xc = x @ c.T
    distance = xx_cc - 2 * xc
    return distance
