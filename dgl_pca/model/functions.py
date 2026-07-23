"""Tensor packing helpers shared by DGL-PCA model branches."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def feature_packing(modal_features: Sequence[torch.Tensor], lengths: torch.Tensor) -> torch.Tensor:
    """Pack padded modality tensors into modality-major utterance nodes."""
    if not modal_features:
        raise ValueError("modal_features must contain at least one tensor")
    if lengths.dim() != 1:
        raise ValueError(f"lengths must be 1D, got shape {tuple(lengths.shape)}")

    packed = []
    batch_size = int(lengths.numel())
    for feature in modal_features:
        if feature.dim() != 3:
            raise ValueError(f"feature tensors must be 3D, got shape {tuple(feature.shape)}")
        if feature.size(0) != batch_size:
            raise ValueError("feature batch size does not match lengths")
        for batch_index, length in enumerate(lengths.tolist()):
            packed.append(feature[batch_index, : int(length)])
    return torch.cat(packed, dim=0)


def multi_concat(nodes_feature: torch.Tensor, lengths: torch.Tensor, n_modals: int) -> torch.Tensor:
    """Concatenate modality-major node features into utterance-level multimodal features."""
    if nodes_feature.dim() != 2:
        raise ValueError(f"nodes_feature must be 2D, got shape {tuple(nodes_feature.shape)}")
    total_length = int(lengths.sum().item())
    expected_rows = total_length * int(n_modals)
    if nodes_feature.size(0) != expected_rows:
        raise ValueError(f"nodes_feature has {nodes_feature.size(0)} rows, expected {expected_rows}")

    chunks = [
        nodes_feature[modal_index * total_length : (modal_index + 1) * total_length]
        for modal_index in range(int(n_modals))
    ]
    return torch.cat(chunks, dim=-1)
