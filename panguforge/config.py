"""Configuration loading with PyYAML -> yamlish fallback."""
import json
from pathlib import Path

from .exceptions import ConfigError
from .utils import yamlish


def _load_yaml(path):
    try:
        import yaml  # noqa
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        return yamlish.load_file(path)


class PanguConfig(dict):
    """Dict subclass with attribute access (getattr fallback to getitem)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def get(self, key, default=None):
        v = super().get(key, default)
        if isinstance(v, dict):
            return PanguConfig(v)
        return v


def load_config(path, defaults=None):
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {path}")
    if p.suffix in (".json",):
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = _load_yaml(str(p))
    merged = dict(defaults or {})
    merged.update(data or {})
    return PanguConfig(merged)
