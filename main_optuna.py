import gc
import json
import os
import random
from datetime import datetime

import numpy as np
import optuna
import torch
from optuna.trial import TrialState

from options.base_options import BaseOptions


def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


def objective(trial):
    args = BaseOptions().initialize()
    args.lr = trial.suggest_loguniform("lr", 1e-6, 1e-2)
    args.weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    args.batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096, 10000, 20000])
    args.decode_channels_lp = trial.suggest_categorical("decode_channels_lp", [128, 256, 512, 1024])
    args.decode_layers_lp = trial.suggest_categorical("decode_layers_lp", [1, 2, 4, 8])
    if args.type_model == "GraphMAE":
        args.num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        args.num_hidden = trial.suggest_categorical("num_hidden", [256, 512, 1024])
        args.attn_drop = trial.suggest_categorical("attn_drop", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        args.in_drop = trial.suggest_categorical("in_drop", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        args.negative_slope = trial.suggest_categorical("negative_slope", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        args.mask_rate = trial.suggest_categorical("mask_rate", [0.4, 0.5, 0.6, 0.7, 0.8])
        args.drop_edge_rate = trial.suggest_categorical("drop_edge_rate", [0.0, 0.05, 0.15, 0.20])
        args.alpha_l = trial.suggest_categorical("alpha_l", [1, 2, 3])
    elif args.type_model == "GraphMAE2":
        args.num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        args.num_hidden = trial.suggest_categorical("num_hidden", [256, 512, 1024])
        args.attn_drop = trial.suggest_categorical("attn_drop", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        args.in_drop = trial.suggest_categorical("in_drop", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        args.negative_slope = trial.suggest_categorical("negative_slope", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        args.mask_rate = trial.suggest_categorical("mask_rate", [0.4, 0.5, 0.6, 0.7, 0.8])
        args.remask_rate = trial.suggest_categorical("remask_rate", [0.4, 0.5, 0.6, 0.7, 0.8])
        args.drop_edge_rate = trial.suggest_categorical("drop_edge_rate", [0.0, 0.05, 0.15, 0.20])
        args.alpha_l = trial.suggest_categorical("alpha_l", [1, 2, 3])
        args.replace_rate = trial.suggest_categorical("replace_rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        args.lam = trial.suggest_categorical("lam", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    elif args.type_model == "S2GAE":
        args.dim_hidden = trial.suggest_categorical("dim_hidden", [128, 256, 512, 1024])
        args.decode_channels = trial.suggest_categorical("decode_channels", [128, 256, 512, 1024])
        args.decode_layers = trial.suggest_int("decode_layers", 1, 8)
        args.mask_ratio = trial.suggest_categorical("mask_ratio", [0.4, 0.5, 0.6, 0.7, 0.8])
    elif args.type_model == "BGRL":
        args.drop_edge_p_1 = trial.suggest_categorical("drop_edge_p_1", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        args.drop_edge_p_2 = trial.suggest_categorical("drop_edge_p_2", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        args.drop_feat_p_1 = trial.suggest_categorical("drop_feat_p_1", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        args.drop_feat_p_2 = trial.suggest_categorical("drop_feat_p_2", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    elif args.type_model == "GBT":
        args.emb_dim = trial.suggest_categorical("emb_dim", [128, 256, 512, 1024])
        args.lr_base = trial.suggest_loguniform("lr_base", 1e-6, 1e-2)
        args.p_x = trial.suggest_categorical("p_x", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        args.p_e = trial.suggest_categorical("p_e", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    elif args.type_model == "CCA-SSG":
        args.dfr = trial.suggest_categorical("dfr", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        args.der = trial.suggest_categorical("der", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        args.hid_dim = trial.suggest_categorical("hid_dim", [128, 256, 512, 1024])
    elif args.type_model == "GCA":
        args.num_hidden = trial.suggest_categorical("num_hidden", [128, 256, 512, 1024])
        args.drop_edge_rate_1 = trial.suggest_categorical("drop_edge_rate_1", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        args.drop_edge_rate_2 = trial.suggest_categorical("drop_edge_rate_2", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        args.drop_feature_rate_1 = trial.suggest_categorical("drop_feature_rate_1", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        args.drop_feature_rate_2 = trial.suggest_categorical("drop_feature_rate_2", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    elif args.type_model == "GraphECL":
        args.hid_dim = trial.suggest_categorical("hid_dim", [128, 256, 512, 1024, 2048])
        args.n_layers = trial.suggest_int("n_layers", 1, 4)
        args.temp = trial.suggest_categorical("temp", [0.4, 0.5, 0.6, 0.7, 0.8])
        args.lam = trial.suggest_loguniform("lam", 1e-6, 1e-2)


    seed = 123
    args.random_seed = seed
    set_seed(args)
    # torch.cuda.empty_cache()
    if args.batch_type == "full_batch":
        from trainer import trainer
    else:
        from trainer_minibatch import trainer
    trnr = trainer(args, trial=trial)
    _, estp_acc, _, _, _, _ = trnr.train_and_test(seed)

    del trnr
    torch.cuda.empty_cache()
    gc.collect()
    return estp_acc


def main():
    args = BaseOptions().initialize()
    print(args)
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna.db",
        study_name=f"{args.type_model}_{args.batch_type}_{args.dataset}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=200)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
