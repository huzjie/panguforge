"""Deterministic hashing utilities.

Uses md5(key) as a seed for a local `random.Random` so every call with the same
key yields the same value — stable across processes and runs (unlike `hash()`).
"""
import hashlib
import random


def _seed(key):
    digest = hashlib.md5(str(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def stable_float(key, lo=0.0, hi=1.0):
    rng = random.Random(_seed(key))
    return lo + (hi - lo) * rng.random()


def stable_ints(key, n, lo=0, hi=100):
    rng = random.Random(_seed(key))
    return [rng.randint(lo, hi) for _ in range(n)]


def stable_choice(key, seq):
    rng = random.Random(_seed(key))
    return seq[rng.randrange(len(seq))]


def stable_shuffle(key, seq):
    rng = random.Random(_seed(key))
    out = list(seq)
    rng.shuffle(out)
    return out
