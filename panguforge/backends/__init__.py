"""Compute backends: mock (trainable), cpu, ascend, openai, vllm, transformers."""
from .registry import get_backend, list_backends, register_backend
from .base import BaseEngine

# Import concrete backends so their @register_backend decorators run.
from . import mock, cpu, ascend, openai, vllm, hf  # noqa: F401

__all__ = ["get_backend", "list_backends", "register_backend", "BaseEngine"]
