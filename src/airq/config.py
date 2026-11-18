"""Load configs/config.yaml and resolve paths relative to the repository root."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(os.environ.get("AIRQ_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_PATH = Path(os.environ.get("AIRQ_CONFIG", REPO_ROOT / "configs" / "config.yaml"))


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str | os.PathLike) -> Path:
    """Return an absolute path; relative paths are taken from the repository root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
