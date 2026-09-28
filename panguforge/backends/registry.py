"""Backend registry."""
from ..registry import Registry

BACKENDS = Registry("backends")


def register_backend(name, cls=None):
    if cls is None:
        def deco(o):
            BACKENDS.register(name, o)
            return o
        return deco
    BACKENDS.register(name, cls)
    return cls


def get_backend(name, config):
    cls = BACKENDS.get(name)
    if cls is None:
        raise KeyError(f"unknown backend '{name}' (available: {BACKENDS.keys()})")
    return cls(config)


def list_backends():
    return BACKENDS.keys()
