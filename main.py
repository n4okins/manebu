import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from functorch import vmap

import manebu.torch.nn as mnn

if __name__ == "__main__":
    print(torch.cuda.is_available())

    linear = mnn.FFLinear(10, 5, 2).cuda()

    def _tmp(a, b, c, d, e):
        return a + 3 * b - 2 * c + d * e / 2, 4 * a ** 2 - 12 * b * c / (d + 1) - e ** 2 + a * b * c / (d * e + 1)

    x = torch.randint(1000, 100000, (100000, 5), device="cuda") / 100000
    y = vmap(lambda i: _tmp(*i))(x)

    print(y)

    print(linear.train(x, y))
