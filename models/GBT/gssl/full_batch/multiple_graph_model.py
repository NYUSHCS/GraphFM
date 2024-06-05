from copy import deepcopy
from typing import Dict, Optional

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import PPI
from tqdm.auto import tqdm

from gssl.augment import GraphAugmentor
from gssl.loss import barlow_twins_loss
from gssl.tasks import evaluate_node_classification_multilabel_f1


class FullBatchMultipleGraphModel:

    def __init__(
        self,
        encoder: nn.Module,
        augmentor: GraphAugmentor,
        lr_base: float,
        total_epochs: int,
        warmup_epochs: int,
    ):
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._encoder = encoder.to(self._device)

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

        self._augmentor = augmentor

        self._total_epochs = total_epochs

    def fit(
        self,
        data: Dict[str, PPI],
        logger: Optional[SummaryWriter] = None,
        log_interval: Optional[int] = None,
    ) -> dict:
        self._encoder.train()
        logs = {
            "log_epoch": [],
            "train_f1": [],
            "val_f1": [],
            "test_f1": [],
            "z": [],
        }

        train_loader = DataLoader(
            dataset=data["train"], batch_size=1, shuffle=True,
        )

        for epoch in tqdm(iterable=range(self._total_epochs)):
            # Train loop
            total_loss = 0

            for batch in train_loader:
                batch = batch.to(self._device)

                self._optimizer.zero_grad()

                (x_a, ei_a), (x_b, ei_b) = self._augmentor(data=batch)

                z_a = self._encoder(x=x_a, edge_index=ei_a)
                z_b = self._encoder(x=x_b, edge_index=ei_b)

                loss = barlow_twins_loss(z_a=z_a, z_b=z_b)

                total_loss += loss.item() * batch.num_graphs

                loss.backward()

                self._optimizer.step()

            # Save loss on every epoch
            total_loss /= len(train_loader.dataset)
            if logger is not None:
                logger.add_scalar("Loss", total_loss, epoch)

            # Log other metrics only in given interval
            if log_interval is not None and epoch % log_interval == 0:
                assert logger is not None

                self._log(data, epoch, logger, logs)

            self._scheduler.step()

        # Save all metrics at the end
        if logger is not None:
            self._log(data, self._total_epochs, logger, logs)

        return logs

    def _log(self, data, epoch, logger, logs):
        z_train = self.predict(data=data["train"])
        z_val = self.predict(data=data["val"])
        z_test = self.predict(data=data["test"])
        self._encoder.train()

        y_train = torch.cat([trd.y for trd in data["train"]], dim=0)
        y_val = torch.cat([vd.y for vd in data["val"]], dim=0)
        y_test = torch.cat([ted.y for ted in data["test"]], dim=0)

        f1 = evaluate_node_classification_multilabel_f1(
            z_train=z_train, y_train=y_train,
            z_val=z_val, y_val=y_val,
            z_test=z_test, y_test=y_test,
        )

        logger.add_scalar("f1/train", f1["train"], epoch)
        logger.add_scalar("f1/val", f1["val"], epoch)
        logger.add_scalar("f1/test", f1["test"], epoch)

        logs["log_epoch"].append(epoch)
        logs["train_f1"].append(f1["train"])
        logs["val_f1"].append(f1["val"])
        logs["test_f1"].append(f1["test"])
        logs["z"].append((
            deepcopy(z_train), deepcopy(z_val), deepcopy(z_test)
        ))

        logger.add_scalar("norm/train", z_train.norm(dim=1).mean(), epoch)
        logger.add_scalar("norm/val", z_val.norm(dim=1).mean(), epoch)
        logger.add_scalar("norm/test", z_test.norm(dim=1).mean(), epoch)

    def predict(self, data: PPI) -> torch.Tensor:
        self._encoder.eval()

        zs = []

        loader = DataLoader(dataset=data, batch_size=1, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(loader, desc="Predict", leave=False):
                z = self._encoder(
                    x=batch.x.to(self._device),
                    edge_index=batch.edge_index.to(self._device),
                )

                zs.append(z.cpu())

            return torch.cat(zs, dim=0)
