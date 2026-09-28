"""Lightweight metrics recorder."""
import time


class Metrics:
    def __init__(self):
        self._series = {}

    def record(self, name, value, step=None):
        s = self._series.setdefault(name, [])
        s.append((step if step is not None else len(s), value))

    def history(self, name):
        return self._series.get(name, [])

    def last(self, name, default=None):
        s = self._series.get(name)
        return s[-1][1] if s else default

    def summarize(self):
        out = {}
        for k, s in self._series.items():
            vals = [v for _, v in s]
            out[k] = {
                "count": len(vals),
                "last": vals[-1] if vals else None,
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
            }
        return out
