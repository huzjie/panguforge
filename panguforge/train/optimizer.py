"""SGD / AdamW style optimizers (zero dependency)."""
import math


class SGD:
    def __init__(self, lr=1e-3, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self._v = {}

    def step(self, params, grads):
        for name, g in grads.items():
            if self.momentum > 0:
                v = self._v.get(name, 0.0) * self.momentum + g
                self._v[name] = v
                params[name] -= self.lr * v
            else:
                params[name] -= self.lr * g


class AdamW:
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self._m = {}
        self._v = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for name, g in grads.items():
            self._m[name] = self.b1 * self._m.get(name, 0.0) + (1 - self.b1) * g
            self._v[name] = self.b2 * self._v.get(name, 0.0) + (1 - self.b2) * g * g
            mhat = self._m[name] / (1 - self.b1 ** self.t)
            vhat = self._v[name] / (1 - self.b2 ** self.t)
            if self.wd > 0:
                params[name] -= self.lr * self.wd * params[name]
            params[name] -= self.lr * mhat / (math.sqrt(vhat) + self.eps)


def make_optimizer(kind, lr, **kw):
    if kind == "adamw":
        return AdamW(lr, **kw)
    return SGD(lr, **kw)
