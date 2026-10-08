"""Configuration loader for SABRE simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
SEEDS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "seeds.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load and parse a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load primary simulator configuration from path or default.yaml."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return load_yaml(cfg_path)


def load_seeds(path: str | Path | None = None) -> dict[str, list[int]]:
    """Load seed split configuration (tuning, calibration, eval)."""
    cfg_path = Path(path) if path is not None else SEEDS_CONFIG_PATH
    return load_yaml(cfg_path)
