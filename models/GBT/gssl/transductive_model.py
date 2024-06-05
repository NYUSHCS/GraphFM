import numpy as np
from typing import Dict, Optional, Tuple, Union

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric import nn as tgnn
from tqdm.auto import tqdm

from .loss import get_loss


class Model:

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

        self._encoder = GCNEncoder(
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

        self._use_pytorch_eval_model = False

    def fit(
        self,
        data: Data,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> dict:
        self._encoder.train()
        data = data.to(self._device)

        self._optimizer.zero_grad()

        (x_a, ei_a), (x_b, ei_b) = augment(
            data=data, p_x=self._p_x, p_e=self._p_e,
        )

        z_a = self._encoder(x=x_a, edge_index=ei_a)
        z_b = self._encoder(x=x_b, edge_index=ei_b)

        loss = self._loss_fn(z_a=z_a, z_b=z_b)

        loss.backward()

        z = self.predict(data=data)
        self._encoder.train()  # Predict sets `eval()` mode

        self._optimizer.step()
        self._scheduler.step()

        #data = data.to("cpu")

        return loss, z

    def predict(self, data: Data) -> torch.Tensor:
        self._encoder.eval()

        with torch.no_grad():
            z = self._encoder(
                x=data.x.to(self._device),
                edge_index=data.edge_index.to(self._device),
            )

            return z.cpu()


class GCNEncoder(nn.Module):

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


def augment(data: Data, p_x: float, p_e: float):
    device = data.x.device

    x = data.x
    num_fts = x.size(-1)

    ei = data.edge_index
    num_edges = ei.size(-1)

    x_a = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x
    x_b = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x

    ei_a = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]
    ei_b = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]

    return (x_a, ei_a), (x_b, ei_b)


def bernoulli_mask(size: Union[int, Tuple[int, ...]], prob: float):
    return torch.bernoulli((1 - prob) * torch.ones(size))
