"""Generic training orchestration loop."""
import time

from .optimizer import make_optimizer
from .lr_scheduler import make_scheduler
from .checkpoint import Checkpointer
from .callbacks import MetricLogger


class Trainer:
    def __init__(self, engine, config, callbacks=None, logger=None):
        self.engine = engine
        self.cfg = config
        self.callbacks = callbacks or []
        self.logger = logger

    def fit(self, dataset, steps=None, epochs=None):
        if steps is None:
            steps = int(self.cfg.get("train", {}).get("steps", 200)) if isinstance(self.cfg.get("train"), dict) else int(self.cfg.get("steps", 200))
        state = {"log_every": int(self.cfg.get("log_every", 20)), "stop": False}
        for cb in self.callbacks:
            cb.on_train_begin(state)
        for step in range(steps):
            batch = dataset.sample_batch()
            metrics = self.engine.train_step(batch)
            for cb in self.callbacks:
                cb.on_step_end(state, step, metrics)
            if state.get("stop"):
                break
        for cb in self.callbacks:
            cb.on_train_end(state, self.engine.metrics())
        return self.engine.metrics()
