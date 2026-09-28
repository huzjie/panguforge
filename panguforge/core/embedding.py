"""Token embedding layer."""
from . import tensor as T


class Embedding:
    def __init__(self, vocab_size, dim, seed=0):
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = T.randn(vocab_size, dim, seed=seed, scale=0.05)

    def __call__(self, ids):
        return [list(self.weight[i]) for i in ids]

    def unembed(self, hidden):
        # hidden: [dim] -> logits [vocab]
        return T.matvec(self.weight, hidden)
