"""Byte/char-level tokenizer (deterministic, zero-dep)."""
from .vocab import Vocab


class MockTokenizer:
    def __init__(self, vocab_size=512):
        self.vocab = Vocab(vocab_size)
        self.vocab_size = vocab_size

    def encode(self, text):
        return self.vocab.encode(text)

    def decode(self, ids):
        return self.vocab.decode(ids)

    @property
    def eos_id(self):
        return 1
