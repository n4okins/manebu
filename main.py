import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_max
from torch_geometric.data import Data


import manebu.torch.nn as mnn
import manebu.torch.nn.common as mcnn

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint


def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
    transform = Compose([
        ToTensor(),
        # Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = {
            "conv1": mnn.FFGCNConv(10, 2, 16),
            "conv2": mnn.FFGCNConv(10, 16, 32),
            "conv3": mnn.FFGCNConv(10, 32, 64),
            "conv4": mnn.FFGCNConv(10, 64, 128),
            "linear1": mnn.FFLinear(10, 128, 64),
            "linear2": mnn.FFLinear(10, 64, 10),
        }
        for l in self.layers.values():
            l.cuda()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for name, layer in self.layers.items():
            if name == "linear1":
                x, _ = scatter_max(x, data.batch, dim=0)
            x = layer(x)
        return x.detach()

    def predict(self, x):
        return self.forward(x)

    def train(self, x, y):
        for y_label in set(y.cpu().detach().numpy()):
            h_pos = x[torch.where(y == y_label)]
            h_neg = x[torch.where(y != y_label)]
            h_neg = h_neg[torch.randperm(len(h_neg))][:len(h_pos)]

            for i, layer in enumerate(self.layers):
                if isinstance(layer, mnn.FFModule):
                    h_pos, h_neg = layer.train(h_pos, h_neg)
                else:
                    h_pos, h_neg = layer(h_pos), layer(h_neg)


def plot_output(output, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]
            ax.imshow(output[:, :, i * nrows + j], cmap="gray")
            ax.axis("off")


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()

    print(x.min(), x.max(), x[torch.where(x != 0)].mean(), x[(torch.where(torch.logical_and(torch.as_tensor(x != 0), torch.as_tensor(x != 1))))].mean())

    net = Net()
    # net = mnn.FFConv2d(1000, in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1).cuda()
    # net = nn.Conv2d(1, 16, (3, 3), padding=1).cuda()

    plt.imshow(x[0:1].detach().cpu().numpy().reshape((1, 28, 28)).transpose(1, 2, 0), cmap="gray")
    plt.show()

    mx = net(x[0:1]).detach().cpu().numpy().transpose((2, 3, 1, 0)).reshape(28, 28, 16)  # .reshape(8, 8, 1)
    print(mx.shape)
    plot_output(mx, 4, 4)
    plt.show()

    net.train(x, y)

    mx = net(x[0:1]).detach().cpu().numpy().transpose((2, 3, 1, 0)).reshape(28, 28, 16)
    plot_output(mx, 4, 4)
    plt.show()
    exit()

    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    net.train(x_pos, x_neg)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
