"""Multi-head self-attention (scaled dot-product)."""
import math
from . import tensor as T


class MultiHeadAttention:
    def __init__(self, dim, num_heads, seed=0):
        assert dim % num_heads == 0, "dim must divide num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wq = T.randn(dim, dim, seed=seed, scale=0.05)
        self.wk = T.randn(dim, dim, seed=seed + 1, scale=0.05)
        self.wv = T.randn(dim, dim, seed=seed + 2, scale=0.05)
        self.wo = T.randn(dim, dim, seed=seed + 3, scale=0.05)

    def _head_split(self, x):
        # x: [seq, dim] -> list of heads, each [seq, head_dim]
        heads = []
        for h in range(self.num_heads):
            base = h * self.head_dim
            heads.append([[row[base + i] for i in range(self.head_dim)] for row in x])
        return heads

    def _head_merge(self, heads):
        seq = len(heads[0])
        return [[v for h in heads for v in h[i]] for i in range(seq)]

    def __call__(self, x, mask=None):
        seq = len(x)
        q = [T.matvec(self.wq, row) for row in x]
        k = [T.matvec(self.wk, row) for row in x]
        v = [T.matvec(self.wv, row) for row in x]
        qh = self._head_split(q)
        kh = self._head_split(k)
        vh = self._head_split(v)
        scale = 1.0 / math.sqrt(self.head_dim)
        out_heads = []
        for h in range(self.num_heads):
            attn = [[T.dot(qh[h][i], kh[h][j]) * scale for j in range(seq)] for i in range(seq)]
            if mask is not None:
                for i in range(seq):
                    for j in range(seq):
                        if not mask(i, j):
                            attn[i][j] = -1e9
            attn = [T.softmax(row) for row in attn]
            oh = []
            for i in range(seq):
                acc = T.zeros(self.head_dim)
                for j in range(seq):
                    acc = T.add(acc, T.mul_scalar(vh[h][j], attn[i][j]))
                oh.append(acc)
            out_heads.append(oh)
        merged = self._head_merge(out_heads)
        return [T.matvec(self.wo, row) for row in merged]


def causal_mask(i, j):
    return j <= i
