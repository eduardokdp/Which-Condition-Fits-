"""Configuration loading. All paths in a config are resolved against the repository root."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

_REQUIRED = ["run_name", "seed", "conditions", "data", "features", "model",
             "decision", "rationale", "output_dir", "eval_dir", "ground_truth_dir"]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"Config {path} is missing keys: {missing}")
    if len(set(cfg["conditions"])) != len(cfg["conditions"]):
        raise ValueError("Conditions must be unique.")
    cfg = copy.deepcopy(cfg)
    cfg["_config_path"] = str(path)
    return cfg


def resolve(rel: str | Path) -> Path:
    """Return an absolute path for a path written relative to the repository root."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p
