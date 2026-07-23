"""Small pickle I/O helpers for model-ready DGL-PCA files."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def load_pickle(path: str | Path) -> Any:
    """Read a pickle file from disk."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Pickle file not found: {resolved}")
    with resolved.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Write a pickle file, creating the parent directory when needed."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
