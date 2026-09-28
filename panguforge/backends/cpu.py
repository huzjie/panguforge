"""CPU backend running a genuine small MoE forward pass via the pure-Python tensor lib."""
from .base import BaseEngine
from .registry import register_backend
from ..core.model import build_model
from ..core import tensor as T


@register_backend("cpu")
class CpuEngine(BaseEngine):
    name = "cpu"

    def __init__(self, config):
        super().__init__(config)
        self.model = build_model(config)
        self.vocab_size = self.model.vocab_size

    def next_token(self, prefix):
        ids = list(prefix)[-self.model.max_seq:]
        if not ids:
            ids = [0]
        logits, _ = self.model.forward(ids)
        return T.argmax(logits[-1])

    def generate(self, prompt, max_tokens=32):
        ids = [ord(c) % self.vocab_size for c in prompt] or [1]
        for _ in range(max_tokens):
            nxt = self.next_token(ids[-self.model.max_seq:])
            ids.append(nxt)
            if nxt == 1:
                break
        return ids
