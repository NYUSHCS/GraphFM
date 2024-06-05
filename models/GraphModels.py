import torch
import torch.nn.functional as F


class _GraphModels(torch.nn.Module):
    def __init__(self, args, data):
        super(_GraphModels, self).__init__()

        self.type_model = args.type_model
        self.num_layers = args.num_layers
        self.num_classes = args.num_classes
        self.num_feats = args.num_feats
        self.batch_size = args.batch_size
        self.dropout = args.dropout

    def inference_ori(self, input_dict):
        device = input_dict["device"]
        x_all = input_dict["x"]
        for i, conv in enumerate(self.convs):
            xs = []
            for _, n_id, adj in self.test_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all
