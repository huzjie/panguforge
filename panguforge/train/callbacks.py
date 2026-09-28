"""Training callbacks."""


class Callback:
    def on_train_begin(self, state):
        pass

    def on_epoch_begin(self, state, epoch):
        pass

    def on_step_end(self, state, step, metrics):
        pass

    def on_epoch_end(self, state, epoch, metrics):
        pass

    def on_train_end(self, state, metrics):
        pass


class MetricLogger(Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_step_end(self, state, step, metrics):
        if step % state.get("log_every", 10) == 0:
            parts = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in metrics.items())
            self.logger.info(f"  step {step}: {parts}")


class EarlyStopping(Callback):
    def __init__(self, patience=5, key="loss", mode="min"):
        self.patience = patience
        self.key = key
        self.mode = mode
        self.best = None
        self.wait = 0

    def on_epoch_end(self, state, epoch, metrics):
        v = metrics.get(self.key)
        if v is None:
            return
        if self.best is None or (self.mode == "min" and v < self.best) or (self.mode == "max" and v > self.best):
            self.best = v
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= self.patience:
            state["stop"] = True
