import importlib
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition as sk_dec
from sklearn import preprocessing as sk_prep
import torch


def seed(value: int = 42):
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    random.seed(value)


def plot_vectors(latent: torch.Tensor, labels: torch.Tensor):
    latent = sk_prep.normalize(X=latent, norm="l2")
    z2d = sk_dec.PCA(n_components=2).fit_transform(latent)

    fig, ax = plt.subplots(figsize=(10, 10))

    for y in labels.unique():
        ax.scatter(
            z2d[labels == y, 0], z2d[labels == y, 1],
            marker=".", label=y.item(),
        )

    fig.legend()

    return fig


def load_cls_from_str(path):
    module, model = path.rsplit('.', maxsplit=1)
    return getattr(importlib.import_module(module), model)

