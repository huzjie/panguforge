"""Random seed helpers."""
import random


def set_seed(seed):
    random.seed(seed)


def make_rng(seed):
    return random.Random(seed)
