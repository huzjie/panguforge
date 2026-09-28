"""Normalization layers."""
from . import tensor as T


class RMSNorm:
    def __init__(self, dim, eps=1e-6, seed=0):
        self.dim = dim
        self.eps = eps
        self.weight = T.ones(dim)

    def __call__(self, x):
        return T.rms_norm(x, self.weight, self.eps)


class LayerNorm:
    def __init__(self, dim, eps=1e-5, seed=0):
        self.dim = dim
        self.eps = eps
        self.weight = T.ones(dim)
        self.bias = T.zeros(dim)

    def __call__(self, x):
        return T.layer_norm(x, self.weight, self.bias, self.eps)
