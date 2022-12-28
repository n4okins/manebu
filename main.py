import torch
import torch.nn as nn
import manebu.torch.nn as mnn

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint

def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
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
        self.layers = [
            mnn.FFLinear(100, in_features=784, out_features=500),
            mnn.FFLinear(100, in_features=500, out_features=500),
            mnn.FFLinear(100, in_features=500, out_features=10)
        ]

        for l in self.layers:
            l.cuda()

    def forward(self, x):
        return self.predict(x)

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            goodness = []
            h = x
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label

    def net_train(self, x, y):
        for y_label in set(y.cpu().detach().numpy()):
            h_pos = x[torch.where(y == y_label)]
            h_neg = x[torch.where(y != y_label)]
            h_neg = h_neg[torch.randperm(len(h_neg))][:len(h_pos)]

            for i, layer in enumerate(self.layers):
                if isinstance(layer, mnn.FFModule):
                    h_pos, h_neg = layer.layer_train(h_pos, h_neg)
                else:
                    h_pos, h_neg = layer(h_pos), layer(h_neg)


if __name__ == "__main__":
    # torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    # for y_label in range(10):
    #     h_pos = x[torch.where(y == y_label)].numpy()
    #     h_neg = x[torch.where(y != y_label)].numpy()
    #     h_neg = h_neg[torch.randperm(len(h_neg))][:len(h_pos)]
    #     nrows, ncols = 10, 10
    #     fig, axes = plt.subplots(nrows, ncols, figsize=(nrows, ncols))
    #     fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    #     for i in range(nrows):
    #         for j in range(ncols):
    #             ax = axes[i][j]
    #             ax.imshow(h_neg[i * nrows + j].transpose(1, 2, 0), cmap="gray")
    #             ax.axis("off")
    #     plt.suptitle(f"label={y_label}")
    #     plt.show()
    # exit()

    net = Net()

    # net = mnn.FFConv2d(1000, in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1).cuda()
    # net = nn.Conv2d(1, 16, (3, 3), padding=1).cuda()
    plt.imshow(x[0:1].detach().cpu().numpy().reshape((1, 28, 28)).transpose(1, 2, 0), cmap="gray")
    plt.show()

    mx = net.predict(x[0:1]).detach().cpu().numpy()
    pprint(list(enumerate(mx)))

    net.net_train(x, y)

    mx = net.predict(x[0:1]).detach().cpu().numpy()
    pprint(list(enumerate(mx)))

    # nrows, ncols = 4, 4
    # fig, axes = plt.subplots(nrows, ncols, figsize=(nrows, ncols))
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    # for i in range(nrows):
    #     for j in range(ncols):
    #         ax = axes[i][j]
    #         ax.imshow(mx[i * nrows + j], cmap="gray")
    #         ax.axis("off")
    #
    # plt.show()
    exit()

    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    net.train(x_pos, x_neg)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())

