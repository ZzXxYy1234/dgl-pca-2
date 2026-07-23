"""Convert YAML configs into runtime namespaces."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch


def build_args_from_config(config: dict[str, Any], seed: int | None = None, device: str | None = None) -> SimpleNamespace:
    """Build the runtime argument namespace used by DGL-PCA model components."""
    experiment = config["experiment"]
    data = config["data"]
    model = config["model"]
    training = config.get("training", {})
    graph = config.get("graph", {})

    dataset = experiment["dataset"]
    feature_dims = data.get("feature_dims")
    if not feature_dims:
        raise ValueError("data.feature_dims is required to instantiate DGL-PCA")

    seeds = experiment.get("seeds", [24])
    selected_seed = int(seed if seed is not None else seeds[0])
    selected_device = torch.device(device or "cpu")

    return SimpleNamespace(
        dataset=dataset,
        task_type=experiment.get("task_type", "sentiment"),
        class_num=int(experiment.get("class_num", 2)),
        modalities=experiment.get("modalities", "atv"),
        multilabel=bool(experiment.get("multilabel", training.get("multilabel", False))),
        seed=selected_seed,
        device=selected_device,
        data=data.get("file"),
        data_root=data.get("root", "data"),
        batch_size=int(training.get("batch_size", 10)),
        dataset_embedding_dims={dataset: {"a": int(feature_dims["a"]), "t": int(feature_dims["t"]), "v": int(feature_dims["v"])}},
        hidden_size=int(model.get("hidden_size", 100)),
        graph_hidden_size=model.get("graph_hidden_size"),
        graph_weight=float(model.get("graph_weight", 1.0)),
        no_gnn=bool(model.get("no_graph", False)),
        use_graph_transformer=bool(model.get("use_graph_transformer", False)),
        graph_transformer_nheads=int(model.get("graph_transformer_nheads", 1)),
        use_crossmodal=bool(model.get("use_crossmodal", False)),
        use_speaker=bool(model.get("use_speaker", True)),
        rnn=model.get("rnn", "transformer"),
        encoder_nlayers=int(model.get("encoder_nlayers", 1)),
        encoder_nheads=int(model.get("encoder_nheads", 4)),
        edge_type=model.get("edge_type", "temp_multi"),
        enable_full_relations=bool(model.get("enable_full_relations", False)),
        drop_rate=float(training.get("drop_rate", 0.5)),
        graph_drop_rate=model.get("graph_drop_rate"),
        use_soft_mask=bool(graph.get("use_soft_mask", True)),
        temporal_decay_beta=float(graph.get("temporal_decay_beta", 0.0)),
        use_relative_time_encoding=bool(graph.get("use_relative_time_encoding", False)),
        no_time_node_embedding=bool(graph.get("no_time_node_embedding", False)),
        edge_prune_ratio=float(graph.get("edge_prune_ratio", 0.0)),
        wp=int(graph.get("wp", 5)),
        wf=int(graph.get("wf", 3)),
        han_layers=int(model.get("han_layers", 1)),
        use_han_residual=bool(model.get("use_han_residual", True)),
        strict_edge_weighting=bool(model.get("strict_edge_weighting", False)),
        crossmodal_nheads=int(model.get("crossmodal_nheads", 2)),
        self_att_nheads=int(model.get("self_att_nheads", 2)),
        num_crossmodal=int(model.get("num_crossmodal", 2)),
        num_self_att=int(model.get("num_self_att", 3)),
        use_highway=bool(model.get("use_highway", False)),
        loss_type=training.get("loss_type", "nll"),
        label_smoothing=float(training.get("label_smoothing", 0.0)),
        focal_gamma=float(training.get("focal_gamma", 2.0)),
        class_weight=bool(training.get("class_weight", False)),
        use_class_weights=bool(training.get("use_class_weights", False)),
        class_weight_values=training.get("class_weight_values"),
        multilabel_threshold=float(training.get("multilabel_threshold", 0.5)),
        binary_threshold=float(training.get("binary_threshold", 0.0)),
        pcm_gate_map=model.get("pcm_gate_map"),
        monitor_graph_pcm_norms=bool(model.get("monitor_graph_pcm_norms", False)),
        consistency_lambda=float(training.get("consistency_lambda", 0.0)),
        consistency_beta=float(training.get("consistency_beta", 0.5)),
        consistency_conf_thr=float(training.get("consistency_conf_thr", 0.8)),
        consistency_window=int(training.get("consistency_window", 1)),
    )
