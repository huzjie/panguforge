"""OpenAI-compatible backend (requires OPENAI_API_KEY)."""
import json
import os
import urllib.request

from .base import BaseEngine
from .registry import register_backend


@register_backend("openai")
class OpenAIEngine(BaseEngine):
    name = "openai"

    def __init__(self, config):
        super().__init__(config)
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = config.get("openai", {}).get("base_url", "https://api.openai.com/v1")
        self.model = config.get("openai", {}).get("model", "gpt-4o-mini")

    def generate(self, prompt, max_tokens=32):
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        body = json.dumps({"model": self.model, "prompt": prompt, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(self.base_url + "/completions", data=body, headers={
            "Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["text"]
