"""Evaluation and smoke-test entry point for DGL-PCA."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from dgl_pca.data.dataset import DialogueDataset
from dgl_pca.data.loader import load_dialogue_splits
from dgl_pca.model import CrossmodalNet, DGLPCA, GraphModel, HAN, UnimodalEncoder
from dgl_pca.training.common import evaluate_model
from dgl_pca.utils.args import build_args_from_config
from dgl_pca.utils.config import load_config
from dgl_pca.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DGL-PCA checkpoints.")
    parser.add_argument("--config", help="Path to a DGL-PCA YAML config.")
    parser.add_argument("--checkpoint", help="Path to a checkpoint.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used by the evaluated run.")
    parser.add_argument("--split", choices=("dev", "test"), default="test", help="Dataset split to evaluate.")
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--output", default=None, help="Optional JSON metrics output path.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a synthetic forward pass.")
    return parser.parse_args()


def run_smoke_test() -> None:
    set_global_seed(24)
    args = SimpleNamespace(
        dataset="mosei",
        task_type="sentiment",
        class_num=2,
        modalities="atv",
        hidden_size=12,
        graph_hidden_size=None,
        graph_weight=1.0,
        no_gnn=False,
        use_graph_transformer=False,
        graph_transformer_nheads=1,
        use_crossmodal=False,
        use_speaker=True,
        dataset_embedding_dims={"mosei": {"a": 8, "t": 8, "v": 8}},
        wp=1,
        wf=1,
        edge_type="temp_multi",
        enable_full_relations=False,
        drop_rate=0.1,
        graph_drop_rate=None,
        use_soft_mask=True,
        temporal_decay_beta=0.0,
        use_relative_time_encoding=False,
        no_time_node_embedding=False,
        edge_prune_ratio=0.0,
        han_layers=1,
        use_han_residual=True,
        strict_edge_weighting=False,
        rnn="ffn",
        use_highway=False,
        loss_type="nll",
        label_smoothing=0.0,
        multilabel=False,
        multilabel_threshold=0.5,
        class_weight=False,
        use_class_weights=False,
        binary_threshold=0.0,
        device=torch.device("cpu"),
    )
    model = DGLPCA(args)
    data = {
        "audio_tensor": torch.randn(2, 2, 8),
        "text_tensor": torch.randn(2, 2, 8),
        "visual_tensor": torch.randn(2, 2, 8),
        "text_len_tensor": torch.tensor([2, 2], dtype=torch.long),
        "speaker_tensor": torch.zeros(2, 2, dtype=torch.long),
        "timestamps": torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
    }
    predictions = model(data)
    if tuple(predictions.shape) != (4,):
        raise RuntimeError(f"Unexpected smoke-test output shape: {tuple(predictions.shape)}")
    _ = (CrossmodalNet, GraphModel, HAN, UnimodalEncoder)
    print("DGL-PCA smoke test passed")


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_smoke_test()
        return

    if not args.config or not args.checkpoint:
        raise ValueError("--config and --checkpoint are required unless --smoke-test is used.")

    config = load_config(args.config)
    runtime_args = build_args_from_config(config, seed=args.seed, device=args.device)
    set_global_seed(runtime_args.seed)

    data_file = Path(config["data"]["file"])
    checkpoint_path = Path(args.checkpoint)
    if not data_file.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {data_file}. "
            "Prepare the dataset under data/ before evaluation."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}. "
            "Train a model first or provide an existing local checkpoint."
        )

    train_set, dev_set, test_set = load_dialogue_splits(data_file, runtime_args)
    test_args = copy.copy(runtime_args)
    test_args.batch_size = len(test_set.samples)
    full_test_set = DialogueDataset(test_set.samples, test_args)
    split_map = {"dev": dev_set, "test": full_test_set}
    _ = train_set

    model = DGLPCA(runtime_args).to(runtime_args.device)
    checkpoint = torch.load(checkpoint_path, map_location=runtime_args.device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    metrics = evaluate_model(model, split_map[args.split], runtime_args)
    result = {
        "config": Path(args.config).as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "split": args.split,
        "metrics": metrics,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
