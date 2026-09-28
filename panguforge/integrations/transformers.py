"""Transformers integration helpers."""


def export_config(config):
    """Convert panguforge config to a transformers-style dict (informational)."""
    model = config.get("model", {})
    return {
        "model_type": "pangu_moe",
        "vocab_size": int(model.get("vocab_size", 512)),
        "hidden_size": int(model.get("dim", 64)),
        "num_attention_heads": int(model.get("num_heads", 4)),
        "num_hidden_layers": int(model.get("num_layers", 2)),
        "num_local_experts": int(model.get("num_experts", 8)),
        "num_experts_per_tok": int(model.get("top_k", 2)),
    }
