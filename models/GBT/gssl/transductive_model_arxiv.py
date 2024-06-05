from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch import nn
from torch_geometric import nn as tgnn

from .loss import get_loss
from .transductive_model import Model


class ArxivModel(Model):

    def __init__(
        self,
        feature_dim: int,
        emb_dim: int,
        loss_name: str,
        p_x: float,
        p_e: float,
        lr_base: float,
        total_epochs: int,
        warmup_epochs: int,
    ):
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._encoder = ThreeLayerGCNEncoder(
            in_dim=feature_dim, out_dim=emb_dim
        ).to(self._device)

        self._loss_fn = get_loss(loss_name=loss_name)

        self._optimizer = torch.optim.AdamW(
            params=self._encoder.parameters(),
            lr=lr_base,
            weight_decay=1e-5,
        )
        self._scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=self._optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=total_epochs,
        )

        self._p_x = p_x
        self._p_e = p_e

        self._total_epochs = total_epochs

        self._use_pytorch_eval_model = True


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
