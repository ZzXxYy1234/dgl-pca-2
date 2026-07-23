"""Shared training and evaluation helpers for DGL-PCA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a padded dialogue batch to the selected device."""
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def align_labels_for_predictions(
    label_tensor: torch.Tensor,
    predictions: torch.Tensor,
    lengths: torch.Tensor,
    class_num: int,
    multilabel: bool,
) -> torch.Tensor:
    """Return labels with the same sample axis as model predictions."""
    if not multilabel:
        labels = label_tensor.view(-1)
        if labels.numel() < predictions.numel():
            raise ValueError(
                f"Not enough labels for predictions: labels={labels.numel()}, predictions={predictions.numel()}"
            )
        return labels[: predictions.numel()]

    if tuple(label_tensor.shape) == tuple(predictions.shape):
        return label_tensor

    if label_tensor.dim() == 1 and label_tensor.numel() == predictions.numel():
        return label_tensor.view_as(predictions)

    batch_size = int(lengths.numel())
    if label_tensor.dim() == 2 and label_tensor.size(0) == batch_size and label_tensor.size(1) == class_num:
        sample_targets = label_tensor
    elif label_tensor.dim() == 1 and label_tensor.numel() == batch_size * class_num:
        sample_targets = label_tensor.view(batch_size, class_num)
    else:
        raise ValueError(
            "Unsupported multilabel target shape: "
            f"labels={tuple(label_tensor.shape)}, predictions={tuple(predictions.shape)}"
        )

    repeated = [sample_targets[i].unsqueeze(0).repeat(int(lengths[i].item()), 1) for i in range(batch_size)]
    return torch.cat(repeated, dim=0)


def compute_metrics(
    predictions: list[torch.Tensor],
    labels: list[torch.Tensor],
    multilabel: bool,
) -> dict[str, float]:
    """Compute evaluation metrics from accumulated prediction and label tensors."""
    if not predictions:
        return {"loss": float("nan")}

    y_pred = torch.cat([item.detach().cpu() for item in predictions], dim=0).numpy()
    y_true = torch.cat([item.detach().cpu() for item in labels], dim=0).numpy()

    if multilabel:
        y_pred = np.asarray(y_pred).astype(int)
        y_true = np.asarray(y_true).astype(int)
        return {
            "f1_micro": _multilabel_f1(y_true, y_pred, average="micro"),
            "f1_macro": _multilabel_f1(y_true, y_pred, average="macro"),
            "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        }

    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    per_class = _single_label_f1_per_class(y_true, y_pred)
    total = max(int(y_true.size), 1)
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "f1_macro": float(np.mean([item["f1"] for item in per_class])),
        "f1_weighted": float(sum(item["f1"] * item["support"] for item in per_class) / total),
    }
    if len(np.unique(y_true)) <= 2:
        labels_present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        positive_label = 1 if 1 in labels_present else labels_present[-1]
        metrics["f1_binary"] = _single_label_f1_for_label(y_true, y_pred, positive_label)
    return metrics


def _single_label_f1_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float]]:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return [
        {
            "label": int(label),
            "f1": _single_label_f1_for_label(y_true, y_pred, label),
            "support": int(np.sum(y_true == label)),
        }
        for label in labels
    ]


def _single_label_f1_for_label(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    tp = int(np.sum((y_true == label) & (y_pred == label)))
    fp = int(np.sum((y_true != label) & (y_pred == label)))
    fn = int(np.sum((y_true == label) & (y_pred != label)))
    denominator = (2 * tp) + fp + fn
    return 0.0 if denominator == 0 else float((2 * tp) / denominator)


def _multilabel_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> float:
    if average == "micro":
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        denominator = (2 * tp) + fp + fn
        return 0.0 if denominator == 0 else float((2 * tp) / denominator)

    if average == "macro":
        scores = []
        for class_index in range(y_true.shape[1]):
            scores.append(_binary_f1(y_true[:, class_index], y_pred[:, class_index]))
        return float(np.mean(scores)) if scores else 0.0

    raise ValueError(f"Unsupported multilabel F1 average: {average}")


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denominator = (2 * tp) + fp + fn
    return 0.0 if denominator == 0 else float((2 * tp) / denominator)


def evaluate_model(
    model: torch.nn.Module,
    dataset: Any,
    runtime_args: Any,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate a model on a DialogueDataset split."""
    model.eval()
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        batch_count = len(dataset) if max_batches is None else min(len(dataset), int(max_batches))
        for batch_index in range(batch_count):
            batch = move_batch_to_device(dataset[batch_index], runtime_args.device)
            loss = model.get_loss(batch)
            batch_predictions = model(batch)
            batch_labels = align_labels_for_predictions(
                batch["label_tensor"],
                batch_predictions,
                batch["text_len_tensor"],
                int(runtime_args.class_num),
                bool(runtime_args.multilabel),
            )
            item_count = int(batch_predictions.shape[0])
            total_loss += float(loss.item()) * max(item_count, 1)
            total_items += item_count
            predictions.append(batch_predictions.detach())
            labels.append(batch_labels.detach())

    metrics = compute_metrics(predictions, labels, bool(runtime_args.multilabel))
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def build_optimizer(model: torch.nn.Module, training_config: dict[str, Any]) -> torch.optim.Optimizer:
    """Create the optimizer declared in the YAML config."""
    learning_rate = float(training_config.get("learning_rate", 2e-4))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    optimizer_name = str(training_config.get("optimizer", "adamw")).lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def checkpoint_path_from_config(config: dict[str, Any], seed: int) -> Path:
    """Resolve the configured checkpoint path for a seed."""
    pattern = config.get("checkpoint", {}).get("path")
    if not pattern:
        experiment_name = config["experiment"]["name"]
        pattern = f"checkpoints/{experiment_name}_seed{{seed}}.pt"
    return Path(str(pattern).format(seed=seed))
