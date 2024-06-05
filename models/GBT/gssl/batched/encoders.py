import os
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import NeighborSampler
from torch_geometric import nn as tgnn
from tqdm import tqdm


class BatchedTwoLayerGCN(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.convs = nn.ModuleList([
            tgnn.GCNConv(in_dim, 2 * out_dim),
            tgnn.GCNConv(2 * out_dim, out_dim),
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(2 * out_dim, momentum=0.01),
        ])
        self.acts = nn.ModuleList([
            nn.PReLU(),
        ])

    def forward(self, x, edge_indexes, sizes):
        for i, (edge_index, size) in enumerate(zip(edge_indexes, sizes)):
            x = self.convs[i](x, edge_index)[:size, :]

            if i != self.num_layers - 1:
                x = self.bns[i](x)
                x = self.acts[i](x)

        return x

    def inference(self, x_all, edge_index_all, inference_batch_size, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers, leave=False)
        pbar.set_description("Evaluation")

        inference_loader = get_inference_loader(
            edge_index=edge_index_all,
            batch_size=inference_batch_size,
        )

        for i in range(len(self.convs)):
            pbar.set_description(f"Evaluation [{i+1}/{len(self.convs)}]")
            xs = []
            for batch_size, n_id, adj in inference_loader:
                edge_index, _, size = adj.to(device)

                x = x_all[n_id].to(device)

                x = self.convs[i](x, edge_index)[:size[1]]

                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.acts[i](x)

                xs.append(x.cpu())
                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

    @property
    def num_layers(self):
        return len(self.convs)


class BatchedThreeLayerGCN(BatchedTwoLayerGCN):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(in_dim=in_dim, out_dim=out_dim)

        self.convs = nn.ModuleList([
            tgnn.GCNConv(in_dim, out_dim),
            tgnn.GCNConv(out_dim, out_dim),
            tgnn.GCNConv(out_dim, out_dim),
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(out_dim, momentum=0.01),
            nn.BatchNorm1d(out_dim, momentum=0.01),
        ])

        self.acts = nn.ModuleList([
            nn.PReLU(),
            nn.PReLU(),
        ])


class BatchedGAT(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 256,
        heads: int = 4,
    ):
        super().__init__()


        self.convs = nn.ModuleList([
            tgnn.GATConv(in_dim, hidden, heads=heads, concat=True),
            tgnn.GATConv(heads * hidden, hidden, heads=heads, concat=True),
            tgnn.GATConv(heads * hidden, out_dim, heads=heads, concat=False),
        ])
        self.skips = nn.ModuleList([
            nn.Linear(in_dim, heads * hidden),
            nn.Linear(heads * hidden, heads * hidden),
            nn.Linear(heads * hidden, out_dim),
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(heads * hidden, momentum=0.01),
            nn.BatchNorm1d(heads * hidden, momentum=0.01),
        ])


    def forward(self, x, edge_indexes, sizes):
        for i, (edge_index, size) in enumerate(zip(edge_indexes, sizes)):
            x_target = x[:size]

            x = self.convs[i]((x, x_target), edge_index)
            x = x + self.skips[i](x_target)

            if i != self.num_layers - 1:
                x = self.bns[i](x)
                x = F.elu(x)

        return x

    def inference(self, x_all, edge_index_all, inference_batch_size, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers, leave=False)
        pbar.set_description("Evaluation")

        inference_loader = get_inference_loader(
            edge_index=edge_index_all,
            batch_size=inference_batch_size,
            num_nodes=x_all.shape[0],
        )

        for i in range(len(self.convs)):
            pbar.set_description(f"Evaluation [{i+1}/{len(self.convs)}]")
            xs = []
            for batch_size, n_id, adj in inference_loader:
                edge_index, _, size = adj.to(device)

                x = x_all[n_id].to(device)
                x_target = x[:size[1]]

                x = self.convs[i]((x, x_target), edge_index)
                x = x + self.skips[i](x_target)

                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = F.elu(x)

                xs.append(x.cpu())
                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

    @property
    def num_layers(self):
        return len(self.convs)


def get_inference_loader(
    edge_index: torch.Tensor,
    batch_size: int,
    num_nodes: Optional[int] = None,
) -> NeighborSampler:
    return NeighborSampler(
        edge_index,
        node_idx=None,
        num_nodes=num_nodes,
        sizes=[-1],
        return_e_id=False,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(os.environ.get("NUM_WORKERS", 0)),
    )
