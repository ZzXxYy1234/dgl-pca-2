"""Random seed helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

