"""Deterministic, fully-trainable mock backend.

The single most important backend: it lets the ENTIRE pipeline (pretrain -> SFT
-> RL -> serving) run end-to-end with zero GPU / zero third-party dependency.

The "skill" scalar is the model's probability of producing the correct next
token. Training raises skill monotonically toward 1.0, so loss strictly
decreases and accuracy strictly increases — a well-defined learning signal.
"""
import math
import random

from .base import BaseEngine
from .registry import register_backend
from ..utils.stable import stable_float, stable_ints, stable_shuffle


@register_backend("mock")
class MockEngine(BaseEngine):
    name = "mock"

    def __init__(self, config):
        super().__init__(config)
        model = config.get("model", {})
        mock = config.get("mock", {})
        self.vocab_size = int(model.get("vocab_size", 512))
        self.skill = float(mock.get("skill_init", 0.4))
        self.lr = float(mock.get("lr", 0.05))
        self._reward_sum = 0.0
        self._reward_n = 0

    def _to_ids(self, x):
        if isinstance(x, str):
            return [ord(c) % self.vocab_size for c in x] or [1]
        return list(x) or [1]

    def world_target(self, key):
        return stable_ints("world:" + str(key), 1, lo=0, hi=self.vocab_size - 1)[0]

    def _p_correct(self):
        return min(0.999, max(0.001, self.skill))

    def compute_loss(self, batch):
        return -math.log(self._p_correct())

    def train_step(self, batch):
        self.skill = min(0.999, self.skill + self.lr * (1.0 - self.skill))
        return {"loss": self.compute_loss(batch), "skill": self.skill, "acc": self.skill}

    def next_token(self, prefix):
        key = str(list(prefix))
        if stable_float("next:" + key) < self.skill:
            return self.world_target(key)
        return stable_ints("noise:" + key, 1, hi=self.vocab_size - 1)[0]

    def sample_token(self, prefix):
        key = str(list(prefix))
        if random.random() < self.skill:
            return self.world_target(key)
        return stable_ints("noise:" + key, 1, hi=self.vocab_size - 1)[0]

    def generate(self, prompt, max_tokens=32, sample=False):
        ids = self._to_ids(prompt)
        pick = self.sample_token if sample else self.next_token
        for _ in range(max_tokens):
            nxt = pick(ids)
            ids.append(nxt)
            if nxt == 1:
                break
        return ids

    def reward(self, prompt, completion):
        p_ids = self._to_ids(prompt)
        gold = self.world_target(str(p_ids))
        if len(completion) > len(p_ids):
            first = completion[len(p_ids)]
        elif completion:
            first = completion[-1]
        else:
            first = -1
        r = 1.0 if first == gold else 0.0
        self._reward_sum += r
        self._reward_n += 1
        return r

    def reinforce(self, prompt, completion, delta):
        self.skill = min(0.999, max(0.0, self.skill + delta))

    def metrics(self):
        return {
            "loss": self.compute_loss([]),
            "skill": self.skill,
            "acc": self.skill,
            "reward": self._reward_sum / max(1, self._reward_n),
        }
