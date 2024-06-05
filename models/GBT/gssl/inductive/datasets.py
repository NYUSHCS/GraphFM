import os
from typing import Tuple

from torch_geometric.datasets import PPI
from torch_geometric import transforms as T

from gssl import DATA_DIR


def load_ppi() -> Tuple[PPI, PPI, PPI]:
    ds_path = os.path.join(DATA_DIR, f"ssl/datasets/PPI")
    feature_norm = T.NormalizeFeatures()

    train_ppi = PPI(root=ds_path, split="train", transform=feature_norm)
    val_ppi = PPI(root=ds_path, split="val", transform=feature_norm)
    test_ppi = PPI(root=ds_path, split="test", transform=feature_norm)

    return train_ppi, val_ppi, test_ppi
