"""Datasets with deterministic world model for target generation."""
import random

from ..utils.stable import stable_float, stable_ints, stable_choice


class SyntheticDataset:
    """Synthetic language-modeling dataset.

    Each example's `target` (correct next token) comes from a deterministic
    world model so training has a well-defined signal (skill -> 1 => acc -> 1).
    """

    def __init__(self, vocab_size=512, num_examples=256, seed=0):
        self.vocab_size = vocab_size
        self.num_examples = num_examples
        self.seed = seed
        self.rng = random.Random(seed)

    def world_target(self, prefix):
        """Deterministic next token given a prefix string."""
        return stable_ints("world:" + str(prefix), 1, lo=0, hi=self.vocab_size - 1)[0]

    def sample_example(self):
        length = self.rng.randint(4, 16)
        prefix = [self.rng.randrange(self.vocab_size) for _ in range(length)]
        key = "".join(chr(65 + (x % 26)) for x in prefix)
        return {"input": prefix, "target": self.world_target(key)}

    def sample_batch(self, batch_size=16):
        return [self.sample_example() for _ in range(batch_size)]


class InstructionDataset:
    """Instruction -> response dataset for SFT."""

    def __init__(self, num_examples=128, seed=0):
        self.num_examples = num_examples
        self.rng = random.Random(seed)
        self._templates = [
            "总结这段话：{x}",
            "把以下内容翻译为代码注释：{x}",
            "请给出 {x} 的要点",
            "解释概念：{x}",
        ]

    def sample_example(self):
        x = stable_choice("instr:" + str(self.rng.random()), ["注意力机制", "稀疏专家", "梯度累积", "上下文窗口"])
        return {"input": x, "target": stable_ints("resp:" + x, 1, hi=99)[0]}

    def sample_batch(self, batch_size=16):
        return [self.sample_example() for _ in range(batch_size)]

    def prompts(self):
        return [self.sample_example()["input"] for _ in range(self.num_examples)]
