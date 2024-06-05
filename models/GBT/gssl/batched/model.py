from copy import deepcopy
import os
from typing import Dict, Optional

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.data import NeighborSampler
from tqdm import tqdm


from gssl.augment import GraphAugmentor
from gssl.loss import barlow_twins_loss
from gssl.tasks import evaluate_node_classification_acc
from gssl.utils import plot_vectors


class BatchedModel:

    def __init__(
        self,
        encoder: nn.Module,
        augmentor: GraphAugmentor,
        lr_base: float,
        total_epochs: int,
        warmup_epochs: int,
        batch_size: int,
        inference_batch_size: int,
        use_pytorch_eval_model: bool,
        use_train_mask: bool,
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

        self._use_pytorch_eval_model = use_pytorch_eval_model

        self._train_loader_sizes = [10, ] * encoder.num_layers
        self._batch_size = batch_size
        self._inference_batch_size = inference_batch_size

        self._use_train_mask = use_train_mask

    def fit(
        self,
        data: Data,
        logger: Optional[SummaryWriter] = None,
        log_interval: Optional[int] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> dict:
        self._encoder.train()
        logs = {
            "log_epoch": [],
            "train_accuracies": [],
            "val_accuracies": [],
            "test_accuracies": [],
            "z": [],
        }

        train_loader = NeighborSampler(
            data.edge_index,
            node_idx=masks["train"] if self._use_train_mask else None,
            sizes=self._train_loader_sizes,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=int(os.environ.get("NUM_WORKERS", 0)),
        )

        for epoch in tqdm(
            iterable=range(self._total_epochs),
            desc="Epochs",
            leave=False,
        ):
            total_loss = 0

            for _, n_id, adjs in tqdm(
                iterable=train_loader,
                desc="Batches",
                leave=False,
            ):
                adjs = [adj.to(self._device) for adj in adjs]

                self._optimizer.zero_grad()

                (x_a, eis_a), (x_b, eis_b) = self._augmentor.augment_batch(
                    x=data.x[n_id].to(self._device),
                    adjs=adjs,
                )

                sizes = [adj.size[1] for adj in adjs]

                z_a = self._encoder(x=x_a, edge_indexes=eis_a, sizes=sizes)
                z_b = self._encoder(x=x_b, edge_indexes=eis_b, sizes=sizes)

                loss = barlow_twins_loss(z_a=z_a, z_b=z_b)

                loss.backward()
                total_loss += loss.item()

                self._optimizer.step()

            self._scheduler.step()
            torch.cuda.empty_cache()

            # Save loss on every epoch
            avg_loss = total_loss / len(train_loader)

            if logger is not None:
                logger.add_scalar("Loss", avg_loss, epoch)

            # Log other metrics only in given interval
            if log_interval is not None and epoch % log_interval == 0:
                assert logger is not None

                z = self.predict(data=data)
                self._encoder.train()  # Predict sets `eval()` mode

                logger.add_figure(
                    "latent",
                    plot_vectors(z, labels=data.y.cpu()),
                    epoch
                )

                accs = evaluate_node_classification_acc(
                    z, data, masks=masks,
                    use_pytorch=self._use_pytorch_eval_model,
                )

                logger.add_scalar("acc/train", accs["train"], epoch)
                logger.add_scalar("acc/val", accs["val"], epoch)
                logger.add_scalar("acc/test", accs["test"], epoch)

                logs["log_epoch"].append(epoch)
                logs["train_accuracies"].append(accs["train"])
                logs["val_accuracies"].append(accs["val"])
                logs["test_accuracies"].append(accs["test"])
                logs["z"].append(deepcopy(z))

                logger.add_scalar("norm", z.norm(dim=1).mean(), epoch)

        # Save all metrics at the end
        if logger is not None:
            z = self.predict(data=data)
            self._encoder.train()  # Predict sets `eval()` mode

            accs = evaluate_node_classification_acc(
                z, data, masks=masks,
                use_pytorch=self._use_pytorch_eval_model,
            )

            logger.add_figure(
                "latent",
                plot_vectors(z, labels=data.y.cpu()),
                self._total_epochs
            )
            logger.add_scalar("acc/train", accs["train"], self._total_epochs)
            logger.add_scalar("acc/val", accs["val"], self._total_epochs)
            logger.add_scalar("acc/test", accs["test"], self._total_epochs)

            logger.add_scalar("norm", z.norm(dim=1).mean(), self._total_epochs)

            logs["log_epoch"].append(self._total_epochs)
            logs["train_accuracies"].append(accs["train"])
            logs["val_accuracies"].append(accs["val"])
            logs["test_accuracies"].append(accs["test"])
            logs["z"].append(deepcopy(z))

        data = data.to("cpu")

        return logs

    def predict(self, data: Data) -> torch.Tensor:
        self._encoder.eval()

        with torch.no_grad():
            z = self._encoder.inference(
                x_all=data.x,
                edge_index_all=data.edge_index,
                inference_batch_size=self._inference_batch_size,
                device=self._device,
            ).cpu()

        return z
