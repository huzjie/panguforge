"""DeepSeek Sparse Attention (DSA) — top-n key selection per query."""
from . import tensor as T
from .attention import MultiHeadAttention


def dsa_mask(seq, n_select):
    # block-sparse approximation: keep self + n_select most recent neighbors
    def mask(i, j):
        if j > i:
            return False
        if i - j <= n_select:
            return True
        return False
    return mask


class SparseAttentionDSA(MultiHeadAttention):
    def __init__(self, dim, num_heads, n_select, seed=0):
        super().__init__(dim, num_heads, seed)
        self.n_select = n_select

    def __call__(self, x):
        return super().__call__(x, mask=dsa_mask(len(x), self.n_select))
