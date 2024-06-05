import os
import random
import numpy as np

import torch
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

def main(args):
    final_acc_list, estp_acc_list = [], []
    final_auc_list, final_ap_list = [], []
    final_nmi_list, final_ari_list = [], []

    if args.batch_type == "full_batch":
        from trainer import trainer
    else:
        from trainer_minibatch import trainer

    for seed in range(len(args.seeds)):
        args.random_seed = seed
        set_seed(args)
        trnr = trainer(args)
        final_acc, estp_acc, auc, ap, nmi, ari = trnr.train_and_test(seed)
        final_acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)
        final_auc_list.append(auc)
        final_ap_list.append(ap)
        final_nmi_list.append(nmi)
        final_ari_list.append(ari)

    final_acc, final_acc_std = np.mean(final_acc_list), np.std(final_acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    final_auc, final_auc_std = np.mean(final_auc_list), np.std(final_auc_list)
    final_ap, final_ap_std = np.mean(final_ap_list), np.std(final_ap_list)
    final_nmi, final_nmi_std = np.mean(final_nmi_list), np.std(final_nmi_list)
    final_ari, final_ari_std = np.mean(final_ari_list), np.std(final_ari_list)

    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    print(f"# final_auc: {final_auc:.4f}±{final_auc_std:.4f}")
    print(f"# final_ap: {final_ap:.4f}±{final_ap_std:.4f}")
    print(f"# final_nmi: {final_nmi:.4f}±{final_nmi_std:.4f}")
    print(f"# final_ari: {final_ari:.4f}±{final_ari_std:.4f}")

    if args.debug_mem_speed:
        trnr = trainer(args)
        trnr.mem_speed_bench()

if __name__ == "__main__":
    args = BaseOptions().initialize()
    print(args)
    main(args)
