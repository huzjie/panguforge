"""Top-k expert router with aux load-balancing loss."""
from . import tensor as T


class TopKRouter:
    def __init__(self, dim, num_experts, top_k=2, seed=0):
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = T.randn(num_experts, dim, seed=seed, scale=0.05)

    def route(self, x):
        """Return (expert_ids, weights, gate_logits) for a hidden vector x."""
        logits = T.matvec(self.gate, x)
        ranked = sorted(range(self.num_experts), key=lambda i: logits[i], reverse=True)
        top = ranked[:self.top_k]
        top_logits = [logits[i] for i in top]
        w = T.softmax(top_logits)
        return top, w, logits

    def load_balancing_loss(self, gate_logits_batch):
        """Aux loss encouraging uniform expert utilization."""
        if not gate_logits_batch:
            return 0.0
        n = len(gate_logits_batch)
        counts = [0.0] * self.num_experts
        for logits in gate_logits_batch:
            counts[T.argmax(logits)] += 1.0
        frac = [c / n for c in counts]
        mean = 1.0 / self.num_experts
        return sum((f - mean) ** 2 for f in frac)
