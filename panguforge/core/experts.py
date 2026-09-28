"""Expert FFN with SwiGLU."""
from . import tensor as T


class SwiGLUExpert:
    def __init__(self, dim, hidden_dim, seed=0):
        self.w1 = T.randn(hidden_dim, dim, seed=seed, scale=0.05)
        self.w2 = T.randn(hidden_dim, dim, seed=seed + 1, scale=0.05)
        self.w3 = T.randn(dim, hidden_dim, seed=seed + 2, scale=0.05)

    def __call__(self, x):
        g = T.silu(T.matvec(self.w1, x))
        u = T.matvec(self.w2, x)
        h = [a * b for a, b in zip(g, u)]
        return T.matvec(self.w3, h)
