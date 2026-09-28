"""HuggingFace transformers backend (requires transformers installed)."""
from .base import BaseEngine
from .registry import register_backend


@register_backend("transformers")
class HFEngine(BaseEngine):
    name = "transformers"

    def __init__(self, config):
        super().__init__(config)
        self.model_id = config.get("hf", {}).get("model", "")

    def generate(self, prompt, max_tokens=32):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise RuntimeError("transformers not installed; pip install transformers")
        tok = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        ids = tok(prompt, return_tensors="pt")
        out = model.generate(**ids, max_new_tokens=max_tokens)
        return tok.decode(out[0], skip_special_tokens=True)
