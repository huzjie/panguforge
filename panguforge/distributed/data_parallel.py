"""Data-parallel wrapper (shards batches across ranks)."""
from .dist import get_world


class DataParallel:
    def __init__(self, engine):
        self.engine = engine

    def reduce(self, metrics_list):
        """Average metric dicts across ranks (single-process: identity)."""
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        out = {}
        for k in keys:
            vals = [m[k] for m in metrics_list if k in m]
            out[k] = sum(vals) / max(1, len(vals))
        return out
