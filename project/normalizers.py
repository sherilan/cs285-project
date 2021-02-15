
import numpy as np
import torch
import torch.nn as nn



class Normalizer(nn.Module):

    def forward(self, data):
        return self.normalize(data)

    def update(self, data):
        raise NotImplementedError()

    def normalize(self, data):
        raise NotImplementedError()

    def update_normalize(self, data):
        self.update(data)
        return self.normalize(data)


class StandardNormalizer(Normalizer):

    def __init__(self, dim=0, alpha=None):
        super().__init__()
        self.dim = dim
        self.m1 = None  # First moment
        self.m2 = None  # Second moment
        self.n = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.alpha = alpha

    @property
    def mean(self):
        return self.m1

    @property
    def var(self):
        return self.m2 - self.m1 ** 2

    @property
    def std(self):
        return self.var ** 0.5

    def update(self, data):
        n = data.size(self.dim)
        w = n / (self.n + n) if self.alpha is None else self.alpha
        m1 = data.mean(dim=self.dim).detach()
        m2 = data.pow(2).mean(dim=self.dim).detach() + 1e-6
        if self.m1 is None:
            self.m1 = nn.Parameter(m1, requires_grad=False)
        else:
            self.m1.data.mul_(1 - w)
            self.m1.data.add_(w * m1)
        if self.m2 is None:
            self.m2 = nn.Parameter(m2, requires_grad=False)
        else:
            self.m2.data.mul_(1 - w)
            self.m2.data.add_(w * m2)
        self.n.data.add_(n)

    def normalize(self, data):
        return (data - self.mean) / self.std
