import os
import os.path as osp

import dgl
import optuna
import torch
import torch_geometric.datasets
import numpy as np

from ogb.nodeproppred import PygNodePropPredDataset
from utils import CustomDGLDataset, do_edge_split_direct
from eval.link_prediction import LPDecoder

from torch_sparse import SparseTensor
from torch_geometric.transforms import ToSparseTensor, ToUndirected

if not os.path.exists('dataset/'):
    os.makedirs('dataset/')
path = osp.join('dataset/')

def load_data(dataset_name, to_sparse):
    if dataset_name in ["ogbn-arxiv", "ogbn-products"]:
        T = ToSparseTensor() if to_sparse else lambda x: x
        if to_sparse and dataset_name == "ogbn-arxiv":
            T = lambda x: ToSparseTensor()(ToUndirected()(x))
        dataset = PygNodePropPredDataset(name=dataset_name, root=path, transform=T)
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        split_masks = {}
        for split in ["train", "valid", "test"]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_idx[split]] = True
            data[f"{split}_mask"] = mask
            split_masks[f"{split}"] = data[f"{split}_mask"]

        if to_sparse:
            data.edge_index = data.adj_t.to_symmetric()

        x = data.x
        y = data.y = data.y.squeeze()

    elif dataset_name in ["Reddit", "Flickr"]:
        data_path = os.path.join(path, dataset_name)
        T = ToSparseTensor() if to_sparse else lambda x: x
        dataset_class = getattr(torch_geometric.datasets, dataset_name)
        dataset = dataset_class(data_path, transform=T)
        data = dataset[0]
        split_masks = {}
        split_masks["train"] = data.train_mask
        split_masks["valid"] = data.val_mask
        split_masks["test"] = data.test_mask
        x = data.x
        y = data.y

    elif dataset_name in ["cora", "pubmed", "citeseer"]:
        dataset_name = dataset_name.capitalize()
        dataset = torch_geometric.datasets.Planetoid(path, dataset_name)
        data = dataset[0]
        split_masks = {}
        split_masks["train"] = data.train_mask
        split_masks["valid"] = data.val_mask
        split_masks["test"] = data.test_mask
        x = data.x
        y = data.y

    else:
        raise Exception(f"the dataset of {dataset_name} has not been implemented")
    return dataset, data, x, y, split_masks


class trainer():
    def __init__(self, args, **kwargs):

        self.args = args
        self.dataset = args.dataset
        self.type_model = args.type_model
        self.device = torch.device(f"cuda:{args.cuda_num}" if args.cuda else "cpu")

        self.seeds = args.seeds
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.num_layers = args.num_layers
        self.epochs = args.epochs
        self.eval_epochs = args.eval_epochs
        self.saved_args = vars(args)
        self.load_model = args.load_model
        self.trial = kwargs.get("trial", None)

        self.save_path = osp.join("./checkpoints/{}/{}/{}".format(self.type_model, self.dataset, args.batch_type))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.log_path = osp.join('mem_speed_log/{}/{}'.format(self.type_model, args.batch_type))
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.pyg_data, self.data, self.x, self.y, self.split_masks = load_data(args.dataset, args.tosparse)

        if args.dataset == "ogbn-arxiv":
            from torch_geometric.utils.sparse import to_edge_index
            self.data.edge_index, _ = to_edge_index(self.data.edge_index)

        self.split_edge = do_edge_split_direct(self.data, self.device, args.fast_split)

        self.num_classes = args.num_classes

        if self.type_model in {"GraphMAE", "GraphMAE2", "CCA-SSG", "GraphECL"}:
            self.dgl_data = CustomDGLDataset(self.dataset, self.data)
            self.data = self.dgl_data[0]
            self.data = dgl.add_self_loop(self.data)

        if self.type_model == "GraphMAE":
            from models.GraphMAE.model import GraphMAE
            self.model = GraphMAE(args, self.data).to(self.device)
            self.max_epoch_f = args.max_epoch_f,
            self.linear_prob = args.linear_prob
            self.predictor_lp = None
        elif self.type_model == "GraphMAE2":
            from models.GraphMAE2.model import GraphMAE2
            self.model = GraphMAE2(args, self.data).to(self.device)
            self.target_nodes = torch.arange(self.x.shape[0], device=self.x.device, dtype=torch.long)
            self.linear_prob = args.linear_prob
            self.drop_g1 = args.drop_g1
            self.drop_g2 = args.drop_g2
            self.predictor_lp = None
        elif self.type_model == "S2GAE":
            from models.S2GAE import model, utils
            if self.data.is_undirected():
                self.edge_index = self.data.edge_index
            else:
                self.edge_index = model.to_undirected(self.data.edge_index)
            self.data.full_adj_t = SparseTensor.from_edge_index(self.edge_index).t()
            self.edge_index, self.test_edge, self.test_edge_neg = utils.do_edge_split_nc(self.edge_index, self.data.x.shape[0])
            self.model = model.S2GAE(args, self.data).to(self.device)
            self.predictor_lp = model.LP_Decoder(args.dim_hidden, args.decode_channels_lp, 1, args.num_layers, args.decode_layers_lp, args.dropout_lp, de_v=args.de_v)
        elif self.type_model == "CCA-SSG":
            from models.CCA_SSG.model import CCA_SSG
            self.model = CCA_SSG(args.num_feats, args.hid_dim, args.hid_dim, args.n_layers, args.use_mlp).to(self.device)
            self.dfr = args.dfr
            self.der = args.der
            self.lambd = args.lambd
            self.N = self.data.number_of_nodes()
            self.num_class = args.num_classes
            self.predictor_lp = LPDecoder(args.hid_dim, args.decode_channels_lp, 1, args.num_layers_lp,
                                       args.dropout_lp)
        elif self.type_model == "GBT":
            from models.GBT.model import _GBT as GBT
            self.model = GBT(args, self.data).to(self.device)
            self.masks = [
                {
                    "train": self.split_masks["train"],
                    "val": self.split_masks["valid"],
                    "test": self.split_masks["test"],
                }
            ]
            self.loss_name = "barlow_twins"
            self.total_epochs = args.total_epochs
            self.warmup_epochs = args.warmup_epochs
            self.log_interval = args.log_interval
            self.emb_dim = args.emb_dim
            self.lr_base = args.lr_base
            self.p_x = args.p_x
            self.p_e = args.p_e
            self.predictor_lp = LPDecoder(args.emb_dim, args.decode_channels_lp, 1, args.num_layers_lp,
                                          args.dropout_lp)
        elif self.type_model == "BGRL":  # -wz-ini
            from models.BGRL import transforms, models, predictors, model, scheduler
            self.transform_1 = transforms.get_graph_drop_transform(drop_edge_p=args.drop_edge_p_1, drop_feat_p=args.drop_feat_p_1)
            self.transform_2 = transforms.get_graph_drop_transform(drop_edge_p=args.drop_edge_p_2, drop_feat_p=args.drop_feat_p_2)
            self.input_size, self.representation_size = self.data.x.size(1), args.graph_encoder_layer[-1]
            self.encoder = models.GCN([self.input_size] + args.graph_encoder_layer, batchnorm=True)  # 512, 256, 128
            self.predictor = predictors.MLP_Predictor(self.representation_size, self.representation_size, hidden_size=args.predictor_hidden_size)
            self.predictor_lp = LPDecoder(self.representation_size, args.decode_channels_lp, 1, args.num_layers_lp, args.dropout_lp)
            self.model = model.BGRL(self.encoder, self.predictor)
            self.lr_scheduler = scheduler.CosineDecayScheduler(args.lr, args.lr_warmup_epochs, args.epochs)
            self.mm_scheduler = scheduler.CosineDecayScheduler(1 - args.mm, 0, args.epochs)
            self.num_eval_splits = args.num_eval_splits
        elif self.type_model == "GCA":
            from models.GCA import model, utils
            from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GraphConv
            import torch.nn.functional as F
            activations = {
                'relu': F.relu,
                'hardtanh': F.hardtanh,
                'elu': F.elu,
                'leakyrelu': F.leaky_relu,
                'prelu': torch.nn.PReLU(),
                'rrelu': F.rrelu
            }
            base_models = {
                'GCNConv': GCNConv,
                'SGConv': SGConv,
                'SAGEConv': SAGEConv,
                'GATConv': utils.gat_wrapper,
                'GraphConv': GraphConv,
                'GINConv': utils.gin_wrapper
            }
            self.encoder = model.Encoder(args.num_feats, args.num_hidden, activations[args.activations],
                              base_model=base_models[args.base_model], k=args.num_layers).to(self.device)
            self.predictor_lp = LPDecoder(args.num_hidden, args.decode_channels_lp, 1,
                                                     args.num_layers_lp, args.dropout_lp).to(self.device)

            self.model = model.GRACE(self.encoder, args.num_hidden, args.num_proj_hidden, args.tau).to(self.device)
            self.drop_scheme = args.drop_scheme
            self.drop_weights, self.feature_weights = utils.get_weights(args, self.data, self.device)
            self.drop_edge_rate_1 = args.drop_edge_rate_1
            self.drop_edge_rate_2 = args.drop_edge_rate_2
            self.drop_feature_rate_1 = args.drop_feature_rate_1
            self.drop_feature_rate_2 = args.drop_feature_rate_2
        elif self.type_model == "GraphECL":
            from models.GraphECL import model
            self.model = model.GraphECL(self.x.shape[1], args.hid_dim, args.hid_dim, args.n_layers, args.temp,
                                  args.use_mlp, args.moving_average_decay, args.num_MLP, args.lambda_loss, args.lam)
            self.neg_sample = args.neg_sample
            self.pos_sample = args.pos_sample
            self.lr2 = args.lr2
            self.wd2 = args.wd2
            self.num_class = args.num_classes
            self.predictor_lp = LPDecoder(args.hid_dim, args.decode_channels_lp, 1, args.num_layers_lp, args.dropout_lp)
        else:
            raise NotImplementedError("please specify `type_model`")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    def train_and_test(self, seed):
        if not self.load_model:
            self.best_acc = [0]
            for epoch in range(self.epochs):
                train_loss = self.train_net(epoch)  # -wz-run
                print(
                    f"Seed: {seed:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {train_loss:.4f}, "
                )
                if (epoch + 1) % self.eval_epochs == 0:
                    train_acc, results, nmi, ari = self.eval_net()
                    print(
                        f"Train Acc: {train_acc:.4f}, "
                        f"Train AUC: {results['AUC'][0]:.4f}, "
                        f"Train AP: {results['AP'][0]:.4f}, "
                        f"Train nmi: {nmi:.4f}, "
                        f"Train ari: {ari:.4f}, "
                    )
                    if self.trial is not None:
                        self.trial.report(train_acc, epoch)
                        if self.trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
        final_acc_list, early_stp_acc_list, results, nmi, ari = self.eval_test()
        print('#################################          ', seed, '          #################################')
        print(
            f"Test Acc: {np.mean(early_stp_acc_list):.4f}, "
            f"Test AUC: {results['AUC'][2]:.4f}, "
            f"Test AP: {results['AP'][2]:.4f}, "
            f"Test nmi: {nmi:.4f}, "
            f"Test ari: {ari:.4f}, "
        )
        return np.mean(final_acc_list), np.mean(early_stp_acc_list), results['AUC'][2], results['AP'][2], nmi, ari


    def train_net(self, epoch):
        self.model.train()

        input_dict = self.get_input_dict(epoch)
        train_loss = self.model.train_net(input_dict)

        return train_loss


    def eval_net(self):
        self.model.eval()
        with torch.no_grad():
            h = self.model.embed(self.data)
        train_acc = self.model.nc_eval_net(h)
        lp_results = self.model.lp_eval_net(h)
        nmi, ari = self.model.nclustering_eval_net(h)
        if train_acc > max(self.best_acc):
            self.best_acc.append(train_acc)
            save_path_model = os.path.join(self.save_path, f"checkpoint.pt")
            torch.save(self.model.state_dict(), save_path_model)
            if self.predictor_lp is not None:
                save_path_pred = os.path.join(self.save_path, f"pred.pt")
                torch.save(self.predictor_lp.state_dict(), save_path_pred)
        return train_acc, lp_results, nmi, ari


    def mem_speed_bench(self):
        input_dict = self.get_input_dict(0)
        self.model.mem_speed_bench(input_dict)


    def train_ensembling(self):
        # assert isinstance(self.model, (SAdaGCN, AdaGCN, GBGCN))
        input_dict = self.get_input_dict(0)
        acc = self.model.train_and_test(input_dict)
        return acc


    def get_input_dict(self, epoch):
        input_dict = {
            "x": self.x,
            "y": self.y,
            "data": self.data,
            "pyg_data": self.pyg_data,
            "dataset": self.dataset,
            "split_masks": self.split_masks,
            "epoch": epoch,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "optimizer": self.optimizer,
            "device": self.device,
            "save_path": self.save_path,
            "num_classes": self.num_classes,
            "split_edge": self.split_edge,
            "saved_args": self.saved_args,
            "log_path": self.log_path,
            "predictor_lp": self.predictor_lp,
        }
        if self.type_model == "GraphMAE":
            input_dict["max_epoch_f"] = self.max_epoch_f
            input_dict["linear_prob"] = self.linear_prob
            input_dict["dgl_data"] = self.dgl_data
        elif self.type_model == "GraphMAE2":
            input_dict["targets"] = self.target_nodes,
            input_dict["linear_prob"] = self.linear_prob
            input_dict["dgl_data"] = self.dgl_data
            input_dict["drop_g1"] = self.drop_g1
            input_dict["drop_g2"] = self.drop_g2
        elif self.type_model == "S2GAE":
            input_dict["edge_index"] = self.edge_index
            input_dict["test_edge"] = self.test_edge
            input_dict["test_edge_neg"] = self.test_edge_neg
        elif self.type_model == "GBT":
            input_dict["loss_name"] = self.loss_name
            input_dict["masks"] = self.masks
            input_dict["total_epochs"] = self.total_epochs
            input_dict["warmup_epochs"] = self.warmup_epochs
            input_dict["log_interval"] = self.log_interval
            input_dict["emb_dim"] = self.emb_dim
            input_dict["lr_base"] = self.lr_base
            input_dict["p_x"] = self.p_x
            input_dict["p_e"] = self.p_e
        elif self.type_model == "CCA-SSG":
            input_dict["dfr"] = self.dfr
            input_dict["der"] = self.der
            input_dict["lambd"] = self.lambd
            input_dict["N"] = self.N
            input_dict["num_class"] = self.num_class
            input_dict["dgl_data"] = self.dgl_data
        elif self.type_model == "BGRL":
            input_dict["transform_1"] = self.transform_1
            input_dict["transform_2"] = self.transform_2
            input_dict["lr_scheduler"] = self.lr_scheduler
            input_dict["mm_scheduler"] = self.mm_scheduler
            input_dict["num_eval_splits"] = self.num_eval_splits
            input_dict["seeds"] = self.seeds
        elif self.type_model == "GCA":
            input_dict["drop_scheme"] = self.drop_scheme
            input_dict["drop_weights"] = self.drop_weights
            input_dict["feature_weights"] = self.feature_weights
            input_dict["drop_edge_rate_1"] = self.drop_edge_rate_1
            input_dict["drop_edge_rate_2"] = self.drop_edge_rate_2
            input_dict["drop_feature_rate_1"] = self.drop_feature_rate_1
            input_dict["drop_feature_rate_2"] = self.drop_feature_rate_2
        elif self.type_model == "GraphECL":
            input_dict["pos_sample"] = self.pos_sample
            input_dict["neg_sample"] = self.neg_sample
            input_dict["num_class"] = self.num_class
            input_dict["dgl_data"] = self.dgl_data
        else:
            Exception(f"the model of {self.type_model} has not been implemented")

        return input_dict

    def eval_test(self):
        self.model.load_state_dict(torch.load(self.save_path + "/checkpoint.pt"))
        if self.predictor_lp is not None:
            self.predictor_lp.load_state_dict(torch.load(self.save_path + "/pred.pt"))

        representations = self.model.embed(self.data)

        from eval.node_classification import node_classification_eval
        if self.type_model == "S2GAE":
            from models.S2GAE.utils import extract_feature_list_layer2

            representations = [representation.to(self.device) for representation in representations]
            feature = [feature_.detach() for feature_ in representations]

            feature_list = extract_feature_list_layer2(feature)
            final_acc_list, early_stp_acc_list = [], []

            for i, feature_tmp in enumerate(feature_list):
                final_acc, early_stp_acc = node_classification_eval(self.pyg_data, feature_tmp, self.y, [0], self.dataset, self.device)
                final_acc_list.append(final_acc)
                early_stp_acc_list.append(early_stp_acc)

            final_acc, early_stp_acc = np.mean(final_acc_list), np.mean(early_stp_acc_list)

        else:
            final_acc, early_stp_acc = node_classification_eval(self.pyg_data, representations, self.y, [0], self.dataset, self.device)


        from eval.link_prediction import link_prediction_eval
        results = link_prediction_eval(self.type_model, representations, self.predictor_lp, self.x.to(self.device), self.split_edge, self.batch_size)

        if self.type_model == "S2GAE":
            representations = torch.cat(representations, dim=1)
        from eval.node_clustering import node_clustering_eval
        nmi, ari, _ = node_clustering_eval(representations.cpu(), self.y.cpu(), self.num_classes)

        return final_acc, early_stp_acc, results, nmi, ari