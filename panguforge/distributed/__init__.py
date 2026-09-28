"""Distributed training primitives (simulation-friendly)."""
from .dist import get_world, init_dist
from .data_parallel import DataParallel
from .expert_parallel import ExpertParallel

__all__ = ["get_world", "init_dist", "DataParallel", "ExpertParallel"]
