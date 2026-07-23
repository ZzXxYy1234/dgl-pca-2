"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_KEYS = ("experiment", "data", "model", "checkpoint")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_config(config, config_path)
    return config


def validate_config(config: dict[str, Any], source: str | Path = "<memory>") -> None:
    """Validate the project configuration schema."""
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ValueError(f"{source}: missing required config sections: {missing}")

    experiment = config.get("experiment", {})
    model = config.get("model", {})
    if not experiment.get("name"):
        raise ValueError(f"{source}: experiment.name is required")
    if not experiment.get("dataset"):
        raise ValueError(f"{source}: experiment.dataset is required")
    if model.get("graph_branch") != "han":
        raise ValueError(f"{source}: DGL-PCA configs must use graph_branch: han")


def iter_config_paths(root: str | Path = "configs") -> list[Path]:
    """Return all YAML config paths under a directory."""
    config_root = Path(root)
    return sorted([*config_root.glob("*.yaml"), *config_root.glob("*.yml")])
