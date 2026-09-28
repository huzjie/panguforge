"""Engine interface shared by all backends."""


class BaseEngine:
    name = "base"

    def __init__(self, config):
        self.config = config

    # -- training interface --
    def compute_loss(self, batch):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def metrics(self):
        return {}

    # -- inference interface --
    def next_token(self, prefix):
        raise NotImplementedError

    def generate(self, prompt, max_tokens=32):
        raise NotImplementedError

    # -- RL interface --
    def reward(self, prompt, completion):
        return 0.0

    def reinforce(self, prompt, completion, delta):
        return None
