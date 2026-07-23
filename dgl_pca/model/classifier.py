"""Classifier head for DGL-PCA utterance predictions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Single-label focal loss over unnormalized class scores."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = float(gamma)

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(scores, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class Highway(nn.Module):
    """One or more gated residual projections."""

    def __init__(self, size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(nn.Linear(size, size) for _ in range(num_layers))
        self.gates = nn.ModuleList(nn.Linear(size, size) for _ in range(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transform, gate_layer in zip(self.transforms, self.gates):
            gate = torch.sigmoid(gate_layer(x))
            candidate = torch.relu(transform(x))
            x = (gate * candidate) + ((1.0 - gate) * x)
        return x


class Classifier(nn.Module):
    """MLP classifier used by DGL-PCA for single-label and multilabel tasks."""

    def __init__(self, input_dim: int, hidden_size: int, tag_size: int, args) -> None:
        super().__init__()
        self.args = args
        self.output_dim = int(getattr(args, "class_num", tag_size) or tag_size)
        self.loss_type = str(getattr(args, "loss_type", "nll")).lower()
        self.label_smoothing = float(getattr(args, "label_smoothing", 0.0))
        self.multilabel = bool(getattr(args, "multilabel", False))
        self.multilabel_threshold = float(getattr(args, "multilabel_threshold", 0.5))
        self.binary_threshold = float(getattr(args, "binary_threshold", 0.0))

        self.highway = Highway(input_dim) if bool(getattr(args, "use_highway", False)) else nn.Identity()
        self.lin1 = nn.Linear(int(input_dim), int(hidden_size))
        self.drop = nn.Dropout(float(getattr(args, "drop_rate", 0.0)))
        self.lin2 = nn.Linear(int(hidden_size), self.output_dim)

        class_weights = self._class_weights(args, self.output_dim)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        if self.multilabel:
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
            self.criterion = None
        elif self.loss_type == "focal":
            self.criterion = FocalLoss(weight=self.class_weights, gamma=float(getattr(args, "focal_gamma", 2.0)))
        else:
            self.criterion = None

    @staticmethod
    def _class_weights(args, output_dim: int) -> torch.Tensor | None:
        values = getattr(args, "class_weight_values", None)
        use_weights = bool(getattr(args, "class_weight", False) or getattr(args, "use_class_weights", False))
        if values is None:
            if use_weights:
                raise ValueError("class_weight is enabled but class_weight_values is not provided")
            return None
        if len(values) != output_dim:
            raise ValueError(f"class_weight_values length {len(values)} does not match output_dim {output_dim}")
        device = getattr(args, "device", torch.device("cpu"))
        return torch.tensor(values, dtype=torch.float32, device=device)

    def get_prob(self, h: torch.Tensor, text_len_tensor: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return log probabilities for single-label tasks and raw scores for all tasks."""
        _ = text_len_tensor
        features = self.highway(h)
        hidden = self.drop(torch.relu(self.lin1(features)))
        scores = self.lin2(hidden)
        log_prob = None if self.multilabel else F.log_softmax(scores, dim=-1)
        return log_prob, scores

    def forward(self, h: torch.Tensor, text_len_tensor: torch.Tensor) -> torch.Tensor:
        log_prob, scores = self.get_prob(h, text_len_tensor)
        if self.multilabel:
            return (torch.sigmoid(scores) >= self.multilabel_threshold).long()

        if self.output_dim == 2 and 0.0 < self.binary_threshold < 1.0:
            positive_prob = torch.softmax(scores, dim=-1)[:, 1]
            return (positive_prob >= self.binary_threshold).long()

        if log_prob is None:
            raise RuntimeError("single-label classifier did not produce log probabilities")
        return torch.argmax(log_prob, dim=-1)

    def get_loss(self, h: torch.Tensor, label_tensor: torch.Tensor, text_len_tensor: torch.Tensor) -> torch.Tensor:
        log_prob, scores = self.get_prob(h, text_len_tensor)
        if self.multilabel:
            targets = self._multilabel_targets(label_tensor, scores, text_len_tensor)
            return self.bce_loss(scores, targets)

        targets = self._single_label_targets(label_tensor, scores)
        if self.loss_type == "focal":
            if self.criterion is None:
                raise RuntimeError("focal criterion was not initialized")
            return self.criterion(scores, targets)
        if self.label_smoothing > 0.0:
            return F.cross_entropy(
                scores,
                targets,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
        if log_prob is None:
            raise RuntimeError("single-label classifier did not produce log probabilities")
        return F.nll_loss(log_prob, targets, weight=self.class_weights)

    @staticmethod
    def _single_label_targets(label_tensor: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        targets = label_tensor.reshape(-1).to(device=scores.device, dtype=torch.long)
        if targets.numel() != scores.size(0):
            raise ValueError(f"single-label target count {targets.numel()} does not match predictions {scores.size(0)}")
        return targets

    def _multilabel_targets(
        self,
        label_tensor: torch.Tensor,
        scores: torch.Tensor,
        text_len_tensor: torch.Tensor,
    ) -> torch.Tensor:
        labels = label_tensor.to(device=scores.device, dtype=torch.float32)
        if labels.shape == scores.shape:
            return labels
        if labels.dim() == 1 and labels.numel() == scores.numel():
            return labels.view_as(scores)

        batch_size = int(text_len_tensor.numel())
        if labels.dim() == 1 and labels.numel() == batch_size * self.output_dim:
            labels = labels.view(batch_size, self.output_dim)
        if labels.dim() == 2 and labels.size(0) == batch_size and labels.size(1) == self.output_dim:
            repeated = [
                labels[index].unsqueeze(0).repeat(int(length.item()), 1)
                for index, length in enumerate(text_len_tensor)
            ]
            return torch.cat(repeated, dim=0)

        raise ValueError(f"unsupported multilabel target shape {tuple(label_tensor.shape)} for scores {tuple(scores.shape)}")
