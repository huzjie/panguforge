"""Checkpoint save/load."""
import json
import os
from pathlib import Path


class Checkpointer:
    def __init__(self, out_dir, keep=3):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep

    def save(self, state, tag="latest"):
        path = self.out_dir / f"checkpoint-{tag}.json"
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def load(self, tag="latest"):
        path = self.out_dir / f"checkpoint-{tag}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_checkpoints(self):
        return sorted(self.out_dir.glob("checkpoint-*.json"))
