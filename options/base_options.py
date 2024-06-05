import argparse

class BaseOptions:
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""

    def initialize(self):
        parser = argparse.ArgumentParser(description="Constrained learing")

        parser.add_argument("--dataset", type=str, required=True, default="cora")
        parser.add_argument("--seeds", type=int, default=[0, 1])
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--batch_type", type=str, required=True,
                            choices=["full_batch", "node_sampling", "subgraph_sampling"])
        parser.add_argument("--load_model", type=bool, default=False)

        parser.add_argument("--tosparse", type=bool, default=False, required=False)
        parser.add_argument(
            "--debug_mem_speed",
            action="store_true",
            help="whether to get the memory usage and throughput",
            default=True
        )

        parser.add_argument(
            "--type_model",
            type=str,
            default="GraphMAE",
            required=True
        )

        parser.add_argument("--resume", action="store_true", default=False)
        parser.add_argument("--cuda", type=bool, default=True, required=False, help="run in cuda mode")
        parser.add_argument("--cuda_num", type=int, default=0, help="GPU number")

        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument(
            "--epochs",
            type=int,
            default=1000,
            help="number of training the one shot model",
        )
        parser.add_argument("--eval_epochs", type=int, default=250, help="number of eval epochs")

        parser.add_argument("--dropout", type=float, default=0.5, help="input feature dropout")
        parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
        parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")  # 5e-4

        parser.add_argument("--decode_channels_lp", type=int, default=256)
        parser.add_argument("--num_layers_lp", type=int, default=2)
        parser.add_argument("--decode_layers_lp", type=int, default=2)
        parser.add_argument("--dropout_lp", type=float, default=0.5)

        parser.add_argument("--eval_batch_size", type=int, default=512)
        parser.add_argument("--fast_split", type=bool, default=False)

        # parameters for ClusterGCN
        parser.add_argument("--num_parts", type=int, default=1500)

        args = parser.parse_args()

        args = self.reset_dataset_dependent_parameters(args)
        args = self.reset_model_dependent_parameters(args)

        return args

    # setting the common hyperparameters used for comparing different methods of a trick
    def reset_dataset_dependent_parameters(self, args):
        if args.dataset == "cora":
            args.num_classes = 7
            args.num_feats = 1433


        elif args.dataset == "pubmed":
            args.num_classes = 3
            args.num_feats = 500


        elif args.dataset == "citeseer":
            args.num_classes = 6
            args.num_feats = 3703


        elif args.dataset == "Flickr":
            args.num_classes = 7
            args.num_feats = 500


        elif args.dataset == "Reddit":
            args.num_classes = 41
            args.num_feats = 602


        elif args.dataset == "ogbn-arxiv":
            args.num_feats = 128
            args.num_classes = 40
            args.N_nodes = 169343
            args.tosparse = True
        
        elif args.dataset == "ogbn-products":
            args.num_classes = 47
            args.num_feats = 100

        return args


    def reset_model_dependent_parameters(self, args):
        if args.type_model == "GraphMAE":
            args.encoder = "gat"
            args.decoder = "gat"
            args.lr_f = 0.001
            args.weight_decay_f = 1e-4
            args.max_epoch_f = 300
            args.linear_prob = True
            args.num_heads = 4
            args.num_out_heads = 1
            args.num_hidden = 256
            args.residual = False
            args.attn_drop = 0.1
            args.in_drop = 0.2
            args.norm = None
            args.negative_slope = 0.2
            args.mask_rate = 0.5
            args.drop_edge_rate = 0.0
            args.replace_rate = 0.0
            args.activation = "prelu"
            args.loss_fn = "sce"
            args.alpha_l = 2
            args.concat_hidden = False
        elif args.type_model == "GraphMAE2":
            args.encoder = "gat"
            args.decoder = "gat"
            args.mask_method = "random"
            args.mask_rate = 0.5
            args.remask_method = "fixed"
            args.remask_rate = 0.5
            args.activation = "prelu"
            args.negative_slope = 0.2
            args.delayed_ema_epoch = 0
            args.num_remasking = 3
            args.drop_edge_rate = 0.0
            args.momentum = 0.996
            args.replace_rate = 0.0
            args.num_dec_layers = 1
            args.lr_f = 0.005
            args.weight_decay_f = 1e-4
            args.max_epoch_f = 300
            args.linear_prob = True
            args.num_heads = 4
            args.num_out_heads = 1
            args.num_hidden = 256
            args.attn_drop = 0.1
            args.in_drop = 0.2
            args.mask_rate = 0.5
            args.activation = "prelu"
            args.loss_fn = "sce"
            args.alpha_l = 3
            args.lam = 0.5
            args.residual = True
            args.norm = None
            args.drop_g1 = None
            args.drop_g2 = None
        elif args.type_model == "S2GAE":
            args.dim_hidden = 128
            args.mask_type = 'dm'
            args.mask_ratio = 0.5
            args.de_v = "v1"
        elif args.type_model == "BGRL":
            args.decode_channels = 256
            args.decode_layers = 2
            args.graph_encoder_layer = [512]
            args.predictor_hidden_size = 512
            args.num_eval_splits = 3
            args.lr_warmup_epochs = 1000
            args.mm = 0.99
            args.decode_channels_lp = 256
            args.num_layers_lp = 2
            args.dropout_lp = 0.5
            args.drop_edge_p_1 = 0.5
            args.drop_edge_p_2 = 0.5
            args.drop_feat_p_1 = 0.3
            args.drop_feat_p_2 = 0.2
        elif args.type_model == "GBT":
            args.total_epochs = 4000
            args.warmup_epochs = 400
            args.log_interval = 1000
            args.emb_dim = 256
            args.lr_base = 5.e-4
            args.p_x = 0.1
            args.p_e = 0.5
        elif args.type_model == "CCA-SSG":
            args.dfr = 0.2
            args.der = 0.2
            args.lambd = 1e-3
            args.hid_dim = 512
            args.n_layers = 2
            args.use_mlp = False
            args.lr2 = 1e-2
            args.wd2 = 1e-4
        elif args.type_model == "GCA":
            args.num_hidden = 512
            args.num_proj_hidden = 32
            args.activations = 'prelu'
            args.base_model = 'GCNConv'
            args.drop_edge_rate_1 = 0.3
            args.drop_edge_rate_2 = 0.4
            args.drop_feature_rate_1 = 0.1
            args.drop_feature_rate_2 = 0.0
            args.tau = 0.4
            if args.batch_type in {"full_batch", "node_sampling"}:
                args.drop_scheme = 'degree'
            else:
                args.drop_scheme = 'evc'
        elif args.type_model == "GraphECL":
            args.hid_dim = 512
            args.n_layers = 3
            args.temp = 0.8
            args.use_mlp = False
            args.moving_average_decay = 0.0
            args.num_MLP = 1
            args.lambda_loss = 1
            args.lam = 5e-6
            args.neg_sample = 0
            args.pos_sample = 0
            args.lr2 = 3e-4
            args.wd2 = 7e-4

        return args
