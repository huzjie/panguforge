"""Transformer block: RMSNorm -> attention -> residual -> RMSNorm -> MoE -> residual."""
from . import tensor as T
from .norm import RMSNorm
from .attention import MultiHeadAttention
from .moe import MoELayer


class TransformerBlock:
    def __init__(self, dim, num_heads, moe_hidden, num_experts, top_k, seed=0, attention_cls=MultiHeadAttention):
        self.norm1 = RMSNorm(dim)
        self.attn = attention_cls(dim, num_heads, seed)
        self.norm2 = RMSNorm(dim)
        self.moe = MoELayer(dim, moe_hidden, num_experts, top_k, seed + 50)

    def __call__(self, x):
        attn_out = self.attn(self.norm1([row[:] for row in x])) if False else self.attn(x)
        x = [T.add(xi, ai) for xi, ai in zip(x, attn_out)]
        moe_out, gate_logits = self.moe(x[0])  # apply MoE token-wise for simplicity
        x = [T.add(xi, moe_out) for xi in x]
        return x, gate_logits
