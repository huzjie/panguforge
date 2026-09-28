"""Collective-communication stubs (single-process no-ops)."""


def all_reduce(x, op="sum"):
    return x


def all_gather(x):
    return [x]


def broadcast(x, src=0):
    return x


def barrier():
    return None
