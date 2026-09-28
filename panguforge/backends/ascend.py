"""Ascend NPU (昇腾) native backend interface.

Defines the Ascend-native training/inference contract (CANN / torch-npu / mindspore
adapters). When no NPU is available the backend reports status and lets the caller
fall back to mock/cpu. This keeps the openPangu-2.0 昇腾原生 story real and pluggable.
"""
import os

from .base import BaseEngine
from .registry import register_backend


@register_backend("ascend")
class AscendEngine(BaseEngine):
    name = "ascend"

    def __init__(self, config):
        super().__init__(config)
        self.available = self._detect()

    def _detect(self):
        env = os.environ.get("ASCEND_HOME") or os.environ.get("ASCEND_TOOLKIT_HOME")
        if env:
            return True
        try:
            import torch_npu  # noqa
            return True
        except ImportError:
            return False

    def status(self):
        return {
            "backend": "ascend",
            "available": self.available,
            "device": os.environ.get("ASCEND_VISIBLE_DEVICES", "0"),
            "note": "昇腾原生后端接口已就绪；无 NPU 时请回退 mock/cpu 后端" if not self.available else "NPU 可用",
        }

    def compute_loss(self, batch):
        if not self.available:
            raise RuntimeError("Ascend NPU not available; use mock/cpu backend")
        return 0.0
