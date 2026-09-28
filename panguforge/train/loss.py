"""Loss functions."""
import math


def cross_entropy(logits, target):
    """Numerically stable softmax cross-entropy for one sample."""
    mx = max(logits)
    ex = [math.exp(v - mx) for v in logits]
    s = sum(ex)
    return -(logits[target] - mx - math.log(s))


def batch_cross_entropy(logits_list, targets):
    return sum(cross_entropy(l, t) for l, t in zip(logits_list, targets)) / max(1, len(logits_list))


def kl_divergence(p, q):
    return sum(a * math.log(a / max(b, 1e-12)) for a, b in zip(p, q) if a > 0)
