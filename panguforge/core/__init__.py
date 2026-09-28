"""Core model primitives (zero-dependency pure-Python tensors + MoE Transformer)."""
from .model import PanguMoE, build_model
from .tensor import TensorOps

__all__ = ["PanguMoE", "build_model", "TensorOps"]
