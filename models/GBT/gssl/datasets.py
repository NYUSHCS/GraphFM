import os
from typing import Dict, List, Tuple

from ogb.nodeproppred import PygNodePropPredDataset
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, PPI, WikiCS
from torch_geometric import transforms as T
from torch_geometric.utils import to_undirected

from GBT.gssl import DATA_DIR


def load_dataset(name: str) -> Tuple[Data, List[Dict[str, torch.Tensor]]]:
    ds_path = os.path.join(DATA_DIR, "datasets/", name)
    feature_norm = T.NormalizeFeatures()
    #create_masks = T.AddTrainValTestMask(
    #    split="train_rest",
    #    num_splits=20,
    #    num_val=0.1,
    #    num_test=0.8,
    #)

    if name == "WikiCS":
        data = WikiCS(
            root=ds_path,
            transform=feature_norm,
        )[0]
    elif name == "Amazon-CS":
        data = Amazon(
            root=ds_path,
            name="computers",
            transform=feature_norm,
            #pre_transform=create_masks,
        )[0]
    elif name == "Amazon-Photo":
        data = Amazon(
            root=ds_path,
            name="photo",
            transform=feature_norm,
            #pre_transform=create_masks,
        )[0]
    elif name == "Coauthor-CS":
        data = Coauthor(
            root=ds_path,
            name="cs",
            transform=feature_norm,
           #pre_transform=create_masks,
        )[0]
    elif name == "Coauthor-Physics":
        data = Coauthor(
            root=ds_path,
            name="physics",
            transform=feature_norm,
            #pre_transform=create_masks,
        )[0]
    elif name == "ogbn-arxiv":
        data = read_ogb_dataset(name=name, path=ds_path)
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    elif name == "ogbn-products":
        data = read_ogb_dataset(name=name, path=ds_path)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if name in ("ogbn-arxiv", "ogbn-products"):
        masks = [
            {
                "train": data.train_mask,
                "val": data.val_mask,
                "test": data.test_mask,
            }
        ]
    else:
        masks = [
            {
                "train": data.train_mask[:, i],
                "val": data.val_mask[:, i],
                "test": (
                    data.test_mask
                    if name == "WikiCS"
                    else data.test_mask[:, i]
                ),
            }
            for i in range(20)
        ]

    return data, masks


def read_ogb_dataset(name: str, path: str) -> Data:
    dataset = PygNodePropPredDataset(root=path, name=name)
    split_idx = dataset.get_idx_split()

    data = dataset[0]

    data.train_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
    data.train_mask[split_idx["train"]] = True

    data.val_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
    data.val_mask[split_idx["valid"]] = True

    data.test_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
    data.test_mask[split_idx["test"]] = True

    data.y = data.y.squeeze(dim=-1)

    return data


def load_ppi() -> Tuple[PPI, PPI, PPI]:
    ds_path = os.path.join(DATA_DIR, "datasets/PPI")
    feature_norm = T.NormalizeFeatures()

    train_ppi = PPI(root=ds_path, split="train", transform=feature_norm)
    val_ppi = PPI(root=ds_path, split="val", transform=feature_norm)
    test_ppi = PPI(root=ds_path, split="test", transform=feature_norm)

    return train_ppi, val_ppi, test_ppi
