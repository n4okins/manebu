import torch
import torch.nn as nn
import torch.optim as opt

from torch_geometric.nn import GCNConv

from torch.nn.common_types import (
    _size_2_t
)


from abc import abstractmethod
from uuid import uuid4
from tqdm import tqdm
from typing import Union

from manebu.utils.simple_utils import return_default

__all__ = [
    "FFModule", "FFLinear", "FFConv2d", "FFGCNConv"
]


class FFModule(nn.Module):
    __layer_cnt__ = 0
    __layer_dict__ = dict()

    def __init__(
            self, train_num_epochs_in_layer: int,
            activation_fn=None, optimizer_fn=None, layer_name=None,
            learning_rate=None, **kwargs
    ):
        nn.Module.__init__(self)

        learning_rate = return_default(learning_rate, 0.001)
        activation_fn = return_default(activation_fn, nn.ReLU)
        self._optimizer_fn = return_default(optimizer_fn, opt.Adam)

        self.train_num_epochs_in_layer = train_num_epochs_in_layer
        self.activation_fn = activation_fn()
        self.optimizer = None
        self.learning_rate = learning_rate

        self._name = layer_name or str(uuid4())[:8]
        self._layer_num = FFModule.__layer_cnt__

        FFModule.__layer_cnt__ += 1
        FFModule.__layer_dict__[self._name] = self

    def init_ffmodule(self):
        self.init_optimizer()

    def init_optimizer(self):
        self.optimizer = self._optimizer_fn(self.parameters(), lr=self.learning_rate)

    @abstractmethod
    def forward(self, x):
        return

    def calc_loss(self, positive_goodness, negative_goodness) -> torch.Tensor:
        return torch.log(
            1 + torch.exp(
                torch.cat(
                    [-positive_goodness, negative_goodness]
                )
            )
        ).mean()

    def train(self, positive_sample, negative_sample):
        with tqdm(range(self.train_num_epochs_in_layer), leave=False) as t:
            for i in t:
                t.set_description(
                    f"[training - L{self._layer_num:3d} | {self.__class__.__name__} '{self._name}'][epoch {i + 1:5d}]")
                loss = self.calc_loss(
                    self.forward(positive_sample).pow(2).mean(1),
                    self.forward(negative_sample).pow(2).mean(1),
                )
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

        return self.forward(positive_sample).detach(), self.forward(negative_sample).detach()


class FFLinear(FFModule, nn.Linear):
    def __init__(
            self, train_num_epochs_in_layer: int, in_features: int, out_features: int,
            bias: bool = True, device=None, dtype=None,
            activation_fn=None, optimizer_fn=None, layer_name=None,
            learning_rate=None, *args, **kwargs
    ) -> None:
        FFModule.__init__(self, train_num_epochs_in_layer, activation_fn, optimizer_fn, layer_name, learning_rate)
        nn.Linear.__init__(self, in_features, out_features, bias, device, dtype)

        self.init_ffmodule()

    def forward(self, x):
        return nn.Linear.forward(self, x)


class FFConv2d(FFModule, nn.Conv2d):
    def __init__(
            self,
            train_num_epochs_in_layer: int,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            activation_fn=None,
            optimizer_fn=None,
            layer_name=None,
            learning_rate=None,
            *args, **kwargs

    ):
        FFModule.__init__(self, train_num_epochs_in_layer, activation_fn, optimizer_fn, layer_name, learning_rate)
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding,
                           dilation, groups, bias, padding_mode, device, dtype)

        self.init_ffmodule()

    def forward(self, x):
        return nn.Conv2d.forward(self, x)


class FFGCNConv(FFModule, GCNConv):
    def __init__(
            self,
            train_num_epochs_in_layer: int,
            in_channels: int, out_channels: int,
            improved: bool = False,
            cached: bool = False,
            add_self_loops: bool = True,
            normalize: bool = True,
            bias: bool = True,
            activation_fn=None,
            optimizer_fn=None,
            layer_name=None,
            learning_rate=None,
    ):
        FFModule.__init__(self, train_num_epochs_in_layer, activation_fn, optimizer_fn, layer_name, learning_rate)
        GCNConv.__init__(self, in_channels, out_channels, improved, cached, add_self_loops, normalize, bias)

        self.init_ffmodule()

    def forward(self, x):
        return GCNConv.forward(self, x)