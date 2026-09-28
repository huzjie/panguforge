"""Process-group abstraction (single-process fallback)."""
import os

_WORLD = {"rank": 0, "world_size": 1}


def init_dist(rank=None, world_size=None, backend="gloo"):
    if world_size and int(world_size) > 1:
        _WORLD["rank"] = int(rank or os.environ.get("RANK", 0))
        _WORLD["world_size"] = int(world_size)
    else:
        _WORLD.update(rank=0, world_size=1)
    return _WORLD


def get_world():
    return dict(_WORLD)


def is_distributed():
    return _WORLD["world_size"] > 1
