"""panguforge — openPangu-2.0 style MoE LLM full-pipeline training & inference framework.

Covers the complete lifecycle: pretraining -> SFT -> RL post-training -> serving,
with an Ascend-native backend interface plus a deterministic, fully-trainable mock
backend so the entire pipeline runs end-to-end without any GPU or third-party dep.
"""
from .version import __version__
from .config import load_config, PanguConfig

__all__ = ["__version__", "load_config", "PanguConfig"]
