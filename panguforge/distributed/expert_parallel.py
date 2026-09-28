"""Expert-parallel sharding layout.

Maps the MoE layer's experts onto ranks so each rank only stores a subset.
In single-process mode this is a no-op that returns the full expert set.
"""
from .dist import get_world


class ExpertParallel:
    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.world = get_world()

    def expert_range(self):
        rank = self.world["rank"]
        ws = self.world["world_size"]
        per = max(1, self.num_experts // ws)
        lo = rank * per
        hi = self.num_experts if rank == ws - 1 else (rank + 1) * per
        return lo, hi

    def shard(self, experts):
        lo, hi = self.expert_range()
        return experts[lo:hi]
