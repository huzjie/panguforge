"""Synthetic vocabulary (deterministic)."""
import string


def build_vocab(vocab_size=512):
    base = list(string.ascii_letters + string.digits + " .,!?\n")
    extra = [f"<t{i}>" for i in range(vocab_size - len(base))]
    toks = base + extra
    return toks[:vocab_size]


class Vocab:
    def __init__(self, size=512):
        self.tokens = build_vocab(size)
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = self.tokens

    def __len__(self):
        return len(self.tokens)

    def encode(self, s):
        return [self.stoi.get(ch, 0) for ch in s]

    def decode(self, ids):
        return "".join(self.itos[i] if i < len(self.itos) else "?" for i in ids)
