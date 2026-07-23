"""Load DGL-PCA train/dev/test split files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .dataset import DialogueDataset
from .io import load_pickle


REQUIRED_SPLITS = ("train", "dev", "test")


def load_dialogue_splits(data_path: str | Path, args: Any) -> tuple[DialogueDataset, DialogueDataset, DialogueDataset]:
    """Load a model-ready pickle containing train, dev, and test splits."""
    data = load_pickle(data_path)
    if not isinstance(data, Mapping):
        raise ValueError(f"{data_path}: expected a mapping with train/dev/test splits")

    missing = [split for split in REQUIRED_SPLITS if split not in data]
    if missing:
        raise ValueError(f"{data_path}: missing required splits: {missing}")

    splits = []
    for split in REQUIRED_SPLITS:
        samples = data[split]
        if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
            raise ValueError(f"{data_path}: split {split!r} must be a sequence of dialogue samples")
        splits.append(DialogueDataset(samples, args))

    return tuple(splits)  # type: ignore[return-value]
