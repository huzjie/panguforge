"""Supervised fine-tuning loop on instruction-response pairs."""
from .trainer import Trainer


class SFTLoop:
    def __init__(self, engine, config, logger=None):
        self.engine = engine
        self.cfg = config
        self.logger = logger

    def run(self, dataset):
        t = Trainer(self.engine, self.cfg, logger=self.logger)
        metrics = t.fit(dataset)
        if self.logger:
            self.logger.info(f"[sft] done -> loss={metrics.get('loss'):.4f} skill={metrics.get('skill'):.4f}")
        return metrics
