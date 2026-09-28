"""Mixture-of-Experts feed-forward layer."""
from . import tensor as T
from .router import TopKRouter
from .experts import SwiGLUExpert


class MoELayer:
    def __init__(self, dim, hidden_dim, num_experts, top_k=2, seed=0):
        self.router = TopKRouter(dim, num_experts, top_k, seed)
        self.experts = [SwiGLUExpert(dim, hidden_dim, seed + 100 + i) for i in range(num_experts)]
        self.num_experts = num_experts

    def __call__(self, x):
        top, weights, logits = self.router.route(x)
        out = T.zeros(len(x))
        for eid, w in zip(top, weights):
            e = self.experts[eid](x)
            out = T.add(out, T.mul_scalar(e, w))
        return out, logits
