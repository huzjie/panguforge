"""RL post-training loop (GRPO by default, PPO optional)."""
from .grpo import grpo_update
from .ppo import ppo_update


class RLLoop:
    def __init__(self, engine, config, logger=None):
        self.engine = engine
        self.cfg = config
        self.logger = logger
        self.algo = str(config.get("rl", {}).get("algorithm", "grpo")) if isinstance(config.get("rl"), dict) else "grpo"

    def run(self, dataset):
        steps = int(self.cfg.get("rl", {}).get("steps", 50)) if isinstance(self.cfg.get("rl"), dict) else 50
        prompts = dataset.prompts()
        update = grpo_update if self.algo == "grpo" else ppo_update
        for i in range(steps):
            m = update(self.engine, prompts)
            if self.logger and i % 10 == 0:
                self.logger.info(f"  [rl:{self.algo}] step {i} reward={m['reward']:.4f} skill={m['skill']:.4f}")
        final = self.engine.metrics()
        if self.logger:
            self.logger.info(f"[rl] done -> reward={final.get('reward'):.4f} skill={final.get('skill'):.4f}")
        return final
