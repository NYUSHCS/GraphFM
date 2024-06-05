import torch
from torch.utils.data import DataLoader

import dgl
import numpy as np
import scipy.sparse as sp

def test_edge_S2GAE(score_func, input_data, h, batch_size):
    preds = []
    for perm in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()

        preds += [score_func(h, edge).cpu()]

    pred_all = torch.cat(preds, dim=0)

    return pred_all

def test_edge_mae(score_func, input_data, h, batch_size):
    preds = []
    for perm in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()
        src, dst = h[edge[0]], h[edge[1]]
        preds += [torch.sigmoid((src*dst).sum(-1)).cpu()]

    pred_all = torch.cat(preds, dim=0)

    return pred_all

def test_edge(score_func, input_data, h, batch_size):
    preds = []
    for perm in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()

        preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]

    pred_all = torch.cat(preds, dim=0)

    return pred_all

def dgl_prepare_datasets(g):
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.05)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:(test_size + val_size)]], v[eids[test_size:(test_size + val_size)]]
    train_pos_u, train_pos_v = u[eids[(test_size + val_size):]], v[eids[(test_size + val_size):]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    val_neg_u, val_neg_v = neg_u[eids[test_size:(test_size + val_size)]], neg_v[eids[test_size:(test_size + val_size)]]
    train_neg_u, train_neg_v = neg_u[neg_eids[(test_size + val_size):]], neg_v[neg_eids[(test_size + val_size):]]

    train_g = dgl.remove_edges(g, eids[:test_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=g.number_of_nodes())
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    return train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g