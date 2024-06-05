from torch import nn
from torch.nn import functional as F
from torch_geometric import nn as tgnn


class TwoLayerGCNEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self._conv1 = tgnn.GCNConv(in_dim, 2 * out_dim)
        self._conv2 = tgnn.GCNConv(2 * out_dim, out_dim)

        self._bn1 = nn.BatchNorm1d(2 * out_dim, momentum=0.01)  # same as `weight_decay = 0.99`

        self._act1 = nn.PReLU()

    def forward(self, x, edge_index):
        x = self._conv1(x, edge_index)
        x = self._bn1(x)
        x = self._act1(x)

        x = self._conv2(x, edge_index)

        return x


class ThreeLayerGCNEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self._conv1 = tgnn.GCNConv(in_dim, out_dim)
        self._conv2 = tgnn.GCNConv(out_dim, out_dim)
        self._conv3 = tgnn.GCNConv(out_dim, out_dim)

        self._bn1 = nn.BatchNorm1d(out_dim, momentum=0.01)
        self._bn2 = nn.BatchNorm1d(out_dim, momentum=0.01)

        self._act1 = nn.PReLU()
        self._act2 = nn.PReLU()

    def forward(self, x, edge_index):
        x = self._conv1(x, edge_index)
        x = self._bn1(x)
        x = self._act1(x)

        x = self._conv2(x, edge_index)
        x = self._bn2(x)
        x = self._act2(x)

        x = self._conv3(x, edge_index)

        return x


class GATEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self._conv1 = tgnn.GATConv(in_dim, 256, heads=4, concat=True)
        self._skip1 = nn.Linear(in_dim, 4 * 256)

        self._conv2 = tgnn.GATConv(4 * 256, 256, heads=4, concat=True)
        self._skip2 = nn.Linear(4 * 256, 4 * 256)

        self._conv3 = tgnn.GATConv(4 * 256, out_dim, heads=6, concat=False)
        self._skip3 = nn.Linear(4 * 256, out_dim)

    def forward(self, x, edge_index):
        x = F.elu(self._conv1(x, edge_index) + self._skip1(x))
        x = F.elu(self._conv2(x, edge_index) + self._skip2(x))
        x = self._conv3(x, edge_index) + self._skip3(x)

        return x
