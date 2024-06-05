from typing import Dict

import numpy as np
from sklearn import linear_model as sk_lm
from sklearn import metrics as sk_mtr
from sklearn import model_selection as sk_ms
from sklearn import multiclass as sk_mc
from sklearn import preprocessing as sk_prep
import torch
from torch import nn
from torch_geometric.data import Data
from tqdm import tqdm


def evaluate_node_classification_acc(
    z: torch.Tensor,
    data: Data,
    masks: Dict[str, torch.Tensor],
    use_pytorch: bool = False,
) -> Dict[str, float]:
    # Normalize input
    z = sk_prep.normalize(X=z, norm="l2")
    masks = masks[0]
    train_mask = masks["train"]
    y = data.y.cpu().numpy()

    if use_pytorch:
        num_cls = y.max() + 1

        clf = train_pytorch_model(
            emb_dim=z.shape[1],
            num_cls=num_cls,
            X=z,
            y=y,
            masks=masks,
        )
    else:
        clf = train_sklearn_model(X=z[train_mask], y=y[train_mask])

    accs = {}

    # Compute accuracy on train, val and test sets
    for split in ("train", "val", "test"):
        mask = masks[split]
        y_true = y[mask]
        y_pred = clf.predict(X=z[mask])

        acc = sk_mtr.classification_report(
            y_true=y_true,
            y_pred=y_pred,
            output_dict=True,
            zero_division=0,
        )["accuracy"]

        accs[split] = acc

    return accs


def train_sklearn_model(
    X: np.ndarray,
    y: np.ndarray,
) -> sk_lm.LogisticRegression:
    # Define parameter space
    C = 2.0 ** np.arange(-10, 10, 1)

    # Use grid search with cross validation to find best estimator
    lr_model = sk_lm.LogisticRegression(
        solver="liblinear",
        max_iter=100,
    )

    clf = sk_ms.GridSearchCV(
        estimator=sk_mc.OneVsRestClassifier(lr_model),
        param_grid={"estimator__C": C},
        n_jobs=len(C),
        cv=5,
        scoring="accuracy",
    )

    clf.fit(X=X, y=y)

    return clf.best_estimator_


def train_pytorch_model(
    emb_dim: int,
    num_cls: int,
    X: np.ndarray,
    y: np.ndarray,
    masks: Dict[str, torch.Tensor],
) -> "LogisticRegression":
    # Define parameter space
    wd = 2.0 ** np.arange(-10, 10, 2)

    best_clf = None
    best_acc = -1

    pbar = tqdm(wd, desc="Train best classifier")
    for weight_decay in pbar:
        lr_model = LogisticRegression(
            in_dim=emb_dim,
            out_dim=num_cls,
            weight_decay=weight_decay,
            is_multilabel=False,
        )

        lr_model.fit(X[masks["train"]], y[masks["train"]])

        acc = sk_mtr.classification_report(
            y_true=y[masks["val"]],
            y_pred=lr_model.predict(X[masks["val"]]),
            output_dict=True,
            zero_division=0,
        )["accuracy"]

        if acc > best_acc:
            best_acc = acc
            best_clf = lr_model

            pbar.set_description(f"Best acc: {best_acc * 100.0:.2f}")

    pbar.close()

    return best_clf


def evaluate_node_classification_multilabel_f1(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    z_test: torch.Tensor,
    y_test: torch.Tensor,
) -> Dict[str, float]:
    # Normalize input
    z_train = sk_prep.StandardScaler().fit_transform(X=z_train)
    z_val = sk_prep.StandardScaler().fit_transform(X=z_val)
    z_test = sk_prep.StandardScaler().fit_transform(X=z_test)

    # Shapes
    emb_dim = z_train.shape[1]
    num_cls = y_train.size(1)

    # Find best classifier for given `weight_decay` space
    weight_decays = 2.0 ** np.arange(-10, 10, 2)

    best_clf = None
    best_f1 = -1

    pbar = tqdm(weight_decays, desc="Train best classifier")
    for wd in pbar:
        lr_model = LogisticRegression(
            in_dim=emb_dim,
            out_dim=num_cls,
            weight_decay=wd,
            is_multilabel=True,
        )

        lr_model.fit(z_train, y_train.numpy())

        f1 = sk_mtr.f1_score(
            y_true=y_val,
            y_pred=lr_model.predict(z_val),
            average="micro",
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_clf = lr_model

            pbar.set_description(f"Best F1: {best_f1 * 100.0:.2f}")

    pbar.close()

    # Compute metrics over all splits
    all_f1 = {
        "train": sk_mtr.f1_score(
            y_true=y_train,
            y_pred=best_clf.predict(z_train),
            average="micro",
            zero_division=0,
        ),
        "val": sk_mtr.f1_score(
            y_true=y_val,
            y_pred=best_clf.predict(z_val),
            average="micro",
            zero_division=0,
        ),
        "test": sk_mtr.f1_score(
            y_true=y_test,
            y_pred=best_clf.predict(z_test),
            average="micro",
            zero_division=0,
        ),
    }

    return all_f1


class LogisticRegression(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        weight_decay: float,
        is_multilabel: bool,
    ):
        super().__init__()

        self.fc = nn.Linear(in_dim, out_dim)

        self._optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=0.01,
            weight_decay=weight_decay,
        )

        self._is_multilabel = is_multilabel

        self._loss_fn = (
            nn.BCEWithLogitsLoss()
            if self._is_multilabel
            else nn.CrossEntropyLoss()
        )

        self._num_epochs = 1000
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        for m in self.modules():
            self.weights_init(m)

        self.to(self._device)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.fc(x)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.train()

        X = torch.from_numpy(X).float().to(self._device)
        y = torch.from_numpy(y).to(self._device)

        for _ in tqdm(range(self._num_epochs), desc="Epochs", leave=False):
            self._optimizer.zero_grad()

            pred = self(X)
            loss = self._loss_fn(input=pred, target=y)

            loss.backward()
            self._optimizer.step()

    def predict(self, X: np.ndarray):
        self.eval()

        with torch.no_grad():
            pred = self(torch.from_numpy(X).float().to(self._device))

        if self._is_multilabel:
            return (pred > 0).float().cpu()
        else:
            return pred.argmax(dim=1).cpu()
