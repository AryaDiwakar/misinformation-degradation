"""Configuration loading shared by the canonical CLI entry points."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    for section in ("data", "model", "training", "logging"):
        config.setdefault(section, {})
    return config
