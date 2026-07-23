"""Training entry point for DGL-PCA."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from dgl_pca.data.dataset import DialogueDataset
from dgl_pca.data.loader import load_dialogue_splits
from dgl_pca.model import DGLPCA
from dgl_pca.training.common import (
    build_optimizer,
    checkpoint_path_from_config,
    evaluate_model,
    move_batch_to_device,
)
from dgl_pca.utils.config import load_config
from dgl_pca.utils.args import build_args_from_config
from dgl_pca.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DGL-PCA from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a DGL-PCA YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint output path.")
    parser.add_argument("--metrics", default=None, help="Optional JSON metrics output path.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs from the config.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches for sanity checks.")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Limit dev/test batches for sanity checks.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training.")
    return parser.parse_args()


def train_one_epoch(model, dataset, optimizer, device: torch.device, max_batches: int | None = None) -> float:
    """Train one epoch over a DialogueDataset split."""
    model.train()
    dataset.shuffle()
    total_loss = 0.0
    total_items = 0

    batch_count = len(dataset) if max_batches is None else min(len(dataset), int(max_batches))
    for batch_index in range(batch_count):
        batch = move_batch_to_device(dataset[batch_index], device)
        optimizer.zero_grad()
        loss = model.get_loss(batch)
        loss.backward()
        optimizer.step()

        item_count = int(batch["text_len_tensor"].sum().item())
        total_loss += float(loss.item()) * max(item_count, 1)
        total_items += item_count

    return total_loss / max(total_items, 1)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    config: dict,
    seed: int,
    epoch: int,
    metrics: dict,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "seed": seed,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime_args = build_args_from_config(config, seed=args.seed, device=args.device)
    set_global_seed(runtime_args.seed)

    name = config["experiment"]["name"]
    data_file = Path(config["data"]["file"])
    if args.dry_run:
        print(
            "DGL-PCA dry run passed: "
            f"config={args.config}, experiment={name}, seed={runtime_args.seed}, "
            f"hidden_size={runtime_args.hidden_size}, graph_branch=HAN"
        )
        return

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {data_file}. "
            "Use --dry-run to validate the configuration without loading data. "
            "For real training, prepare the dataset under data/."
        )

    train_set, dev_set, test_set = load_dialogue_splits(data_file, runtime_args)
    model = DGLPCA(runtime_args).to(runtime_args.device)
    optimizer = build_optimizer(model, config.get("training", {}))

    training_config = config.get("training", {})
    epochs = int(args.epochs if args.epochs is not None else training_config.get("epochs", 40))
    patience = int(training_config.get("early_stop_patience", 8))
    min_delta = float(training_config.get("early_stop_min_delta", 0.0))
    monitor_metric = str(training_config.get("monitor_metric", "f1_weighted"))
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else checkpoint_path_from_config(config, runtime_args.seed)

    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_set, optimizer, runtime_args.device, args.max_train_batches)
        dev_metrics = evaluate_model(model, dev_set, runtime_args, max_batches=args.max_eval_batches)
        score = float(dev_metrics.get(monitor_metric, dev_metrics.get("accuracy", 0.0)))
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev": dev_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))

        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                runtime_args.seed,
                epoch,
                {"dev": dev_metrics, "monitor_metric": monitor_metric, "monitor_score": score},
            )
        else:
            stale_epochs += 1

        if patience > 0 and stale_epochs >= patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=runtime_args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_args = copy.copy(runtime_args)
    test_args.batch_size = len(test_set.samples)
    full_test_set = DialogueDataset(test_set.samples, test_args)
    test_metrics = evaluate_model(model, full_test_set, runtime_args, max_batches=args.max_eval_batches)
    summary = {
        "experiment": name,
        "seed": runtime_args.seed,
        "best_epoch": best_epoch,
        "checkpoint": checkpoint_path.as_posix(),
        "best_dev": checkpoint["metrics"]["dev"],
        "test": test_metrics,
    }
    if args.max_train_batches is not None or args.max_eval_batches is not None:
        summary["sanity_limits"] = {
            "max_train_batches": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
        }

    if args.metrics:
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"summary": summary, "history": history}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
