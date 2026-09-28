"""Learning-rate schedulers."""
import math


class CosineDecay:
    def __init__(self, base_lr, total_steps, min_lr=0.0, warmup=0):
        self.base = base_lr
        self.total = max(1, total_steps)
        self.min = min_lr
        self.warmup = warmup

    def __call__(self, step):
        if step < self.warmup:
            return self.base * (step + 1) / max(1, self.warmup)
        t = min(1.0, (step - self.warmup) / max(1, self.total - self.warmup))
        return self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * t))


class Constant:
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, step):
        return self.lr


def make_scheduler(kind, lr, total_steps, **kw):
    if kind == "cosine":
        return CosineDecay(lr, total_steps, **kw)
    return Constant(lr)
