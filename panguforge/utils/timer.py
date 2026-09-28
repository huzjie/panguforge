"""Timing helpers."""
import time
from contextlib import contextmanager


@contextmanager
def timer(name="op"):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    return dt


class Stopwatch:
    def __init__(self):
        self._t = {}

    def start(self, key):
        self._t[key] = time.perf_counter()

    def stop(self, key):
        return time.perf_counter() - self._t.get(key, time.perf_counter())

    def report(self):
        return dict(self._t)
