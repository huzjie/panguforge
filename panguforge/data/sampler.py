"""Data samplers."""
import random


class RandomSampler:
    def __init__(self, n, seed=0):
        self.n = n
        self.rng = random.Random(seed)

    def sample(self, k=1):
        return [self.rng.randrange(self.n) for _ in range(k)]


class RoundRobinSampler:
    def __init__(self, n):
        self.n = n
        self.i = 0

    def sample(self, k=1):
        out = [(self.i + j) % self.n for j in range(k)]
        self.i += k
        return out
