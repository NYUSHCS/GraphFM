import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import do_edge_split_direct

import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy

class LPDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LPDecoder, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

@torch.no_grad()
def link_prediction_eval(type_model, h, score_func, x, split_edge, batch_size):

    if type_model == "S2GAE":
        from .utils import test_edge_S2GAE as test_edge
    else:
        h = h.to(x.device)
        if score_func is not None:
            from .utils import test_edge
            score_func.to(x.device)
        else:
            from .utils import test_edge_mae as test_edge

    pos_train_pred = test_edge(score_func, split_edge['train']['edge'], h, batch_size)
    neg_valid_pred = test_edge(score_func, split_edge['valid']['edge_neg'], h, batch_size)
    pos_valid_pred = test_edge(score_func, split_edge['valid']['edge'], h, batch_size)
    pos_test_pred = test_edge(score_func, split_edge['test']['edge'], h, batch_size)
    neg_test_pred = test_edge(score_func, split_edge['test']['edge_neg'], h, batch_size)

    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred), torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)

    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(),
          neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())

    result = get_metric_score(pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    #        score_emb = [pos_valid_pred.cpu(), neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x.cpu()]

    return result


def evaluate_hits(evaluator, pos_pred, neg_pred, k_list):
    results = {}
    for K in k_list:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']
        # test_hits = evaluator.eval({
        #     'y_pred_pos': pos_test_pred,
        #     'y_pred_neg': neg_test_pred,
        # })[f'hits@{K}']

        hits = round(hits, 4)
        # test_hits = round(test_hits, 4)

        results[f'Hits@{K}'] = hits

    return results


def evaluate_auc(val_pred, val_true):
    val_pred, val_true = val_pred.detach().numpy(), val_true.detach().numpy()
    valid_auc = roc_auc_score(val_true, val_pred)
    # test_auc = roc_auc_score(test_true, test_pred)
    results = {}

    valid_auc = round(valid_auc, 4)
    # test_auc = round(test_auc, 4)

    results['AUC'] = valid_auc

    valid_ap = average_precision_score(val_true, val_pred)
    # test_ap = average_precision_score(test_true, test_pred)

    valid_ap = round(valid_ap, 4)
    # test_ap = round(test_ap, 4)

    results['AP'] = valid_ap

    return results


def get_metric_score(pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    result = {}

    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int),
                            torch.zeros(neg_val_pred.size(0), dtype=int)])

    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int),
                          torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int),
                           torch.zeros(neg_test_pred.size(0), dtype=int)])

    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])

    return result


from .utils import test_edge
def perform_nn_link_eval(
    score_func, h, x, edge_split, batch_size, device, mute=False, max_epoch=300
):
    """Trains a NN-based link prediction model on the provided embeddings in the transductive setting.
    Returns the trained link predictor and various evaluation metrics.
    """
    score_func.to(device)
    h = h.to(device)
    train_pos, valid_pos, test_pos = (
        edge_split['train']['edge'].to(device),
        edge_split['valid']['edge'].to(device),
        edge_split['test']['edge'].to(device),
    )
    valid_pos_neg, test_pos_neg = edge_split['valid']['edge_neg'].to(device), \
                                  edge_split['test']['edge_neg'].to(device)

    optimizer = torch.optim.Adam(score_func.parameters(),lr=0.01, weight_decay=1e-5)


    def train():
        score_func.train()
        # train_pos = train_pos.transpose(1, 0)
        total_loss = total_examples = 0

        for perm in DataLoader(range(train_pos.size(0)), batch_size,
                               shuffle=True):
            optimizer.zero_grad()

            num_nodes = x.size(0)

            ######################### remove loss edges from the aggregation
            mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
            mask[perm] = 0

            edge = train_pos[perm].t()

            pos_out = score_func(h[edge[0]], h[edge[1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            # Just do some trivial random sampling.
            edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                                 device=h.device)
            neg_out = score_func(h[edge[0]], h[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            loss = pos_loss + neg_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

            optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
        return total_loss / total_examples


    @torch.no_grad()
    def test():
        score_func.eval()

        pos_train_pred = test_edge(score_func, train_pos, h, batch_size)

        neg_valid_pred = test_edge(score_func, valid_pos_neg, h, batch_size)

        pos_valid_pred = test_edge(score_func, valid_pos, h, batch_size)

        pos_test_pred = test_edge(score_func, test_pos, h, batch_size)

        neg_test_pred = test_edge(score_func, test_pos_neg, h, batch_size)

        pos_train_pred = torch.flatten(pos_train_pred)
        neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred), torch.flatten(pos_valid_pred)
        pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)

        result = get_metric_score(pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

        return result


    best_val_auc = 0
    best_val_epoch = 0
    best_model = None
    
    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:

        loss = train()

        with torch.no_grad():
            results = test()

        if results["AUC"][1] >= best_val_auc:
            best_val_auc = results["AUC"][1]
            best_val_epoch = epoch
            best_model = copy.deepcopy(score_func)

        if not mute and epoch % 50 == 0:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_auc:{results['AUC'][1]: .4f}, test_auc:{results['AUC'][2]: .4f}")

    best_model.eval()
    with torch.no_grad():
        score_func = best_model
        best_results = test()
    if mute:
        print(
            f"# IGNORE: --- TestAUC: {results['AUC'][2]:.4f}, early-stopping-TestAUC: {best_results['AUC'][2]:.4f}, Best ValAUC: {best_val_auc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(
            f"--- TestAUC: {results['AUC'][2]:.4f}, early-stopping-TestAUC: {best_results['AUC'][2]:.4f}, Best ValAUC: {best_val_auc:.4f} in epoch {best_val_epoch} --- ")

    return best_results