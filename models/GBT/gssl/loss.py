import torch
import torch.nn as nn
from .discriminator import Discriminator
from .readout import AvgReadout
import numpy as np
EPS = 1e-15
def _cross_correlation_matrix(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    batch_size = z_a.size(0)

    # Apply batch normalization
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)

    # Cross-correlation matrix
    c = (z_a_norm.T @ z_b_norm) / batch_size

    return c

def GraphCL_loss(z_0, z_1, z_a: torch.Tensor,z_b: torch.Tensor,):
    read = AvgReadout()
    sigm = nn.Sigmoid()
    batch_size = 1
    nb_nodes  = z_a.shape[0]
    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1).to("cuda")
    disc = Discriminator(z_a.shape[1]).to("cuda")
    c_a = read(z_a,None)
    c_a = sigm(c_a)
    c_b = read(z_b,None)
    c_b = sigm(c_b)
    ret1 = disc(c_a, z_0, z_1, None, None)
    ret2 = disc(c_b, z_0, z_1, None, None)
    logits = ret1+ret2
    logits = logits.unsqueeze(0)
    b_xent = nn.BCEWithLogitsLoss()
    loss = b_xent(logits, lbl)
    return loss

def barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    batch_size = z_a.size(0)
    feature_dim = z_a.size(1)
    _lambda = 1 / feature_dim

    # Apply batch normalization
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)

    # Cross-correlation matrix
    c = (z_a_norm.T @ z_b_norm) / batch_size

    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (
        (1 - c.diagonal()).pow(2).sum()
        + _lambda * c[off_diagonal_mask].pow(2).sum()
    )

    return loss


def hilbert_schmidt_independence_criterion(
    z_a: torch.Tensor,
    z_b: torch.Tensor
) -> torch.Tensor:
    feature_dim = z_a.size(1)
    _lambda = 1 / feature_dim

    # Cross-correlation matrix
    c = _cross_correlation_matrix(z_a=z_a, z_b=z_b)

    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (
        (1 - c.diagonal()).pow(2).sum()
        + _lambda * (1 + c[off_diagonal_mask]).pow(2).sum()
    )

    return loss

def get_loss(loss_name: str):
    if loss_name == "barlow_twins":
        return barlow_twins_loss
    elif loss_name == "GraphCL_loss":
        return GraphCL_loss
    elif loss_name == "hsic":
        return hilbert_schmidt_independence_criterion
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")