import torch
import numpy as np
import torch.nn as nn
import json
import copy
from tqdm import tqdm
from torch import optim as optim
from sklearn.model_selection import train_test_split


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


class LogisticRegression_nn(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, x, *args):
        logits = self.linear(x)
        return logits


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def fit_logistic_regression(data, features, labels, data_random_seeds, dataset_name, device, mute=False, max_epoch=300,
                                k_shot=20, test_k_value=False):
    '''
    test_k_value=True: disable default split for arxiv dataset

    '''

    x = features.to(device)
    labels = labels.to(device)

    num_classes = labels.max().item() + 1

    final_accs_list = []
    estp_test_acc_list = []
    for data_random_seed in data_random_seeds:
        if "ogbn" in dataset_name.lower() and not test_k_value:
            split_idx = data.get_idx_split()

            # Extract indices from the JSON file
            train_idx = torch.tensor(split_idx['train'])
            val_idx = torch.tensor(split_idx['valid'])
            test_idx = torch.tensor(split_idx['test'])

            # Convert indices to boolean masks
            train_mask = torch.zeros(len(labels), dtype=torch.bool)
            val_mask = torch.zeros(len(labels), dtype=torch.bool)
            test_mask = torch.zeros(len(labels), dtype=torch.bool)

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
        else:
            #            assert False

            rng = np.random.RandomState(data_random_seed)  # this will ensure the dataset will be split exactly the same
            indices = np.arange(len(x))
            # train：0.2   val：0.2   test：0.6
            train_indices, temp_indices, y_train, y_temp = train_test_split(indices, labels, test_size=0.8,
                                                                            random_state=rng)
            val_indices, test_indices, y_val, y_test = train_test_split(temp_indices, y_temp, test_size=0.75,
                                                                        random_state=rng)
            # Create train_mask, val_mask, and test_mask
            train_mask = np.zeros(len(x), dtype=bool)
            val_mask = np.zeros(len(x), dtype=bool)
            test_mask = np.zeros(len(x), dtype=bool)
            train_mask[train_indices] = True
            val_mask[val_indices] = True
            test_mask[test_indices] = True

        best_val_acc = 0
        best_val_epoch = 0
        best_model = None
        ####
        criterion = torch.nn.CrossEntropyLoss()
        model = LogisticRegression_nn(x.shape[1], num_classes)
        model.to(device)
        optimizer_f = create_optimizer("adam", model, lr=0.01, weight_decay=1e-5)
        optimizer = optimizer_f

        if not mute:
            epoch_iter = tqdm(range(max_epoch))
        else:
            epoch_iter = range(max_epoch)

        for epoch in epoch_iter:
            model.train()
            out = model(x)
            loss = criterion(out[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

            with torch.no_grad():
                model.eval()
                pred = model(x)
                val_acc = accuracy(pred[val_mask], labels[val_mask])
                val_loss = criterion(pred[val_mask], labels[val_mask])
                test_acc = accuracy(pred[test_mask], labels[test_mask])
                test_loss = criterion(pred[test_mask], labels[test_mask])

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                best_model = copy.deepcopy(model)

            if not mute and epoch % 50 == 0:
                epoch_iter.set_description(
                    f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

        best_model.eval()
        with torch.no_grad():
            pred = best_model(x)
            estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
        if mute:
            print(
                f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
        else:
            print(
                f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

        final_accs_list.append(test_acc)
        estp_test_acc_list.append(estp_test_acc)
    # (final_acc, es_acc, best_acc)
    return final_accs_list, estp_test_acc_list


def node_classification_eval(data, representations, y, seeds, dataset, device):

    final_acc, early_stp_acc = fit_logistic_regression(data=data, features=representations, labels=y,
                                                           data_random_seeds=seeds,
                                                           dataset_name=dataset, device=device)
    final_acc.extend(final_acc)
    early_stp_acc.extend(early_stp_acc)

    return final_acc, early_stp_acc
