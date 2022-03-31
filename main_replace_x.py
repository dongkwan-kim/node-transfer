import os.path as osp
import random
from itertools import product

import numpy as np
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, RandomPartitionGraphDataset
from torch_geometric.nn import (
    GCNConv, GATv2Conv, GATConv, SAGEConv, GINConv, TransformerConv, LabelPropagation
)

path_real = osp.join("/mnt/nas2/GNN-DATA/PYG", Planetoid.__name__)
dataset_real = Planetoid(path_real, "Cora")
data_real = dataset_real[0]

path_syn = osp.join("/mnt/nas2/GNN-DATA/PYG", RandomPartitionGraphDataset.__name__)
data_syn_list = []


def replace_x(_data_syn):

    # important
    mask = data_real.train_mask
    # or
    # mask = ~data_real.test_mask

    for c in range(dataset_real.num_classes):
        rxc = data_real.x[mask][data_real.y[mask] == c]
        sxc = _data_syn.x[_data_syn.y == c]
        rand = torch.randint(0, rxc.size(0), [sxc.size(0)])
        _data_syn.x[_data_syn.y == c] = rxc[rand, :]
    return _data_syn


kws_list = [
    dict(node_homophily_ratio=nhr, average_degree=ad)
    for nhr, ad in product(
        [0.63, 0.73, 0.83],
        [2.9, 3.4, 3.9, 4.4, 4.9],
    )
]
K = len(kws_list)

for kws in kws_list:
    dataset_syn = RandomPartitionGraphDataset(
        path_syn,
        num_channels=dataset_real.num_features,
        num_classes=dataset_real.num_classes,
        num_nodes_per_class=400,
        **kws,  # node_homophily_ratio=0.63, average_degree=3.90,
        transform=T.Compose([
            T.RandomNodeSplit("random", num_splits=K),
        ]),
    )
    data_syn = dataset_syn[0]
    data_syn_list.append(data_syn)


class Net(torch.nn.Module):
    def __init__(self, layer, hidden_channels=128):
        super().__init__()

        _cls = eval(layer)
        if layer == "GCNConv" or layer == "SAGEConv":
            self.conv1 = _cls(dataset_real.num_features, hidden_channels)
            self.conv2 = _cls(hidden_channels, dataset_real.num_classes)
        elif layer == "GATConv" or layer == "TransformerConv" or layer == "GATv2Conv":
            self.conv1 = _cls(dataset_real.num_features, hidden_channels // 8, heads=8)
            self.conv2 = _cls(hidden_channels, dataset_real.num_classes)
        else:
            raise ValueError(f"Wrong layer: {layer}")

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)

    def embed(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        return self.conv1(x, edge_index, edge_attr)


def train(data, idx=None):
    model.train()

    if data.train_mask.dim() >= 2 and idx is not None:
        mask = data.train_mask[:, idx]
    else:
        mask = data.train_mask

    (F.nll_loss(model(data)[mask], data.y[mask]) / K).backward()

    if idx is None or idx == K - 1:
        optimizer.step()
        optimizer.zero_grad()


@torch.no_grad()
def test(data, idx=None):
    model.eval()
    log_probs, accs = model(data), []
    for _, mask in data("train_mask", "val_mask", "test_mask"):

        if mask.dim() >= 2 and idx is not None:
            mask = mask[:, idx]

        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


@torch.no_grad()
def logistic_regression_test(data, h_type):
    reg = LogisticRegression()
    model.eval()
    if h_type == "x":
        h = data.x.cpu()
    elif h_type == "h1":
        h = model.embed(data).cpu()
    elif h_type == "h2":
        h = model(data_real).cpu()
    else:
        raise ValueError(f"Wrong type: {h_type}")

    y = data.y.cpu()
    reg.fit(h[data.train_mask].numpy(),
            y[data.train_mask].numpy())
    lr_test_acc = reg.score(h[data.test_mask].numpy(),
                            y[data.test_mask].numpy())
    return lr_test_acc


if __name__ == "__main__":

    EPOCH = 4000
    WD = 1e-4
    LAYER = "GATConv"  # GCNConv, SAGEConv, GATConv, GATv2Conv, TransformerConv

    print("Real", data_real)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_real = Net(LAYER).to(device), data_real.to(device)
    data_syn_list = [ds.to(device) for ds in data_syn_list]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=WD)

    for epoch in range(EPOCH):
        random.shuffle(data_syn_list)

        train_list, val_list, test_list = [], [], []
        for data_syn in data_syn_list:
            replace_x(data_syn)
            train(data_syn, idx=epoch % K)
            train_acc, val_acc, test_acc = test(data_syn, idx=epoch % K)
            train_list.append(train_acc)
            val_list.append(val_acc)
            test_list.append(test_acc)

        if epoch % 10 == 0:
            print("\t".join(str(v) for v in [
                    LAYER,
                    epoch,
                    np.mean(train_list),
                    np.mean(val_list),
                    np.mean(test_list),
                    logistic_regression_test(data_real, "x"),
                    logistic_regression_test(data_real, "h1"),
                    logistic_regression_test(data_real, "h2"),
                ]))
            """ 
            print(
                f"{LAYER} | "
                f"Epoch: {epoch:03d}, "
                f"Train: {np.mean(train_list):.4f}, "
                f"Val: {np.mean(val_list):.4f}, "
                f"Test: {np.mean(test_list):.4f}, "
                f'LRT X: {logistic_regression_test(data_real, "x"):.4f}, '
                f'LRT H1: {logistic_regression_test(data_real, "h1"):.4f}, '
                f'LRT H2: {logistic_regression_test(data_real, "h2"):.4f}'
            )
            """