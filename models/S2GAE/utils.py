from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np
import torch
import math
import random

from torch_geometric.utils import add_self_loops
from torch_geometric.utils import negative_sampling


def extract_feature_list_layer2(feature_list):
    xx_list = []
    xx_list.append(feature_list[-1])
    tmp_feat = torch.cat(feature_list, dim=-1)
    xx_list.append(tmp_feat)
    return xx_list

def do_edge_split_nc(edge_index, num_nodes, val_ratio=0.05, test_ratio=0.1):
    random.seed(234)
    torch.manual_seed(234)

    row, col = edge_index
    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]
    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))
    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    test_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    train_pos_edge_index = torch.stack([r, c], dim=0)
    # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
    neg_edge_index = negative_sampling(
        edge_index, num_nodes=num_nodes,
        num_neg_samples=row.size(0))
    test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]

    train_pos_edge = torch.cat([train_pos_edge_index, val_pos_edge_index], dim=1)

    return train_pos_edge.t(), test_pos_edge_index.t(), test_neg_edge_index.t()


def edgemask_um(mask_ratio, split_edge, device, num_nodes):
    if isinstance(split_edge, torch.Tensor):
        edge_index = split_edge
    else:
        edge_index = split_edge['train']['edge']
    num_edge = len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num])
    mask_index = torch.from_numpy(index[-mask_num:])
    edge_index_train = edge_index[pre_index].t()
    edge_index_mask = edge_index[mask_index].t()
    edge_index = to_undirected(edge_index_train)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index).t()
    return adj, edge_index, edge_index_mask.to(device)


def edgemask_dm(mask_ratio, split_edge, device, num_nodes):
    if isinstance(split_edge, torch.Tensor):
        edge_index = to_undirected(split_edge.t()).t()
    else:
        edge_index = torch.stack([split_edge['train']['edge'][:, 1], split_edge['train']['edge'][:, 0]], dim=1)
        edge_index = torch.cat([split_edge['train']['edge'], edge_index], dim=0)

    num_edge = len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num]).type(torch.int64)
    mask_index = torch.from_numpy(index[-mask_num:]).type(torch.int64)
    edge_index_train = edge_index[pre_index].t()
    edge_index_mask = edge_index[mask_index].to(device)

    edge_index = edge_index_train
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index).t()
    return adj, edge_index, edge_index_mask.to(device)