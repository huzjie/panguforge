"""Request/response schemas for the OpenAI-compatible HTTP API."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CompletionRequest:
    prompt: str
    max_tokens: int = 32
    temperature: float = 1.0
    top_p: float = 1.0

    @classmethod
    def from_dict(cls, d):
        return cls(
            prompt=d.get("prompt", ""),
            max_tokens=int(d.get("max_tokens", 32)),
            temperature=float(d.get("temperature", 1.0)),
            top_p=float(d.get("top_p", 1.0)),
        )


@dataclass
class CompletionChoice:
    text: str
    finish_reason: str = "length"


@dataclass
class CompletionResponse:
    id: str = ""
    choices: List[CompletionChoice] = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "object": "text_completion",
            "choices": [{"text": c.text, "index": 0, "finish_reason": c.finish_reason} for c in self.choices],
        }
