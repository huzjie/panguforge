"""vLLM backend (high-throughput serving, requires vllm installed)."""
from .base import BaseEngine
from .registry import register_backend


@register_backend("vllm")
class VLLMEngine(BaseEngine):
    name = "vllm"

    def __init__(self, config):
        super().__init__(config)
        self.model = config.get("vllm", {}).get("model", "")

    def generate(self, prompt, max_tokens=32):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise RuntimeError("vllm not installed; pip install vllm")
        llm = LLM(model=self.model)
        out = llm.generate(prompt, SamplingParams(max_tokens=max_tokens))
        return out[0].outputs[0].text
