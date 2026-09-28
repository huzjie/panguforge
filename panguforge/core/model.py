"""PanguMoE — openPangu-2.0 style sparse Mixture-of-Experts Transformer."""
from . import tensor as T
from .embedding import Embedding
from .norm import RMSNorm
from .transformer import TransformerBlock
from .attention import MultiHeadAttention


class PanguMoE:
    def __init__(self, vocab_size, dim, num_heads, num_layers, moe_hidden,
                 num_experts, top_k=2, max_seq=512, seed=0):
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k
        self.max_seq = max_seq
        self.embed = Embedding(vocab_size, dim, seed)
        self.blocks = [TransformerBlock(dim, num_heads, moe_hidden, num_experts, top_k, seed + 1000 * i)
                       for i in range(num_layers)]
        self.final_norm = RMSNorm(dim)
        self.lm_head = T.randn(vocab_size, dim, seed=seed + 9999, scale=0.05)

    def forward(self, ids):
        """ids: [seq] -> logits for next-token [seq, vocab]."""
        x = self.embed(ids)
        gate_logits_all = []
        for block in self.blocks:
            x, gl = block(x)
            gate_logits_all.append(gl)
        logits = []
        for row in x:
            row = T.rms_norm(row, self.final_norm.weight, self.final_norm.eps)
            logits.append(T.matvec(self.lm_head, row))
        return logits, gate_logits_all

    def num_parameters(self):
        params = self.vocab_size * self.dim  # embedding
        params += self.vocab_size * self.dim  # lm head
        for _ in self.blocks:
            params += 4 * self.dim * self.dim  # attention q/k/v/o
            params += self.num_experts * (3 * self.dim * self.moe_hidden_approx())
            params += self.num_experts * self.dim
        return params

    def moe_hidden_approx(self):
        return self.dim * 4


def build_model(config):
    cfg = config.get("model", config)
    return PanguMoE(
        vocab_size=int(cfg.get("vocab_size", 512)),
        dim=int(cfg.get("dim", 64)),
        num_heads=int(cfg.get("num_heads", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        moe_hidden=int(cfg.get("moe_hidden", 256)),
        num_experts=int(cfg.get("num_experts", 8)),
        top_k=int(cfg.get("top_k", 2)),
        max_seq=int(cfg.get("max_seq", 512)),
        seed=int(cfg.get("seed", 0)),
    )
