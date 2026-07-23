"""Graph construction and HAN dispatch for the DGL-PCA graph branch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .han import HAN


class GraphModel(nn.Module):
    """Build weighted temporal dialogue graphs and run the hand-written HAN branch."""

    def __init__(self, g_dim: int, h1_dim: int, h2_dim: int, num_modals: int, device, args) -> None:
        super().__init__()
        self.args = args
        self.device = torch.device(device)
        self.n_modals = int(num_modals)
        self.input_dim = int(g_dim)
        self.hidden_dim = int(h1_dim)

        self.window_past = int(getattr(args, "wp", 0))
        self.window_future = int(getattr(args, "wf", 0))
        self.edge_type_mode = str(getattr(args, "edge_type", "temp_multi"))
        self.use_soft_mask = bool(getattr(args, "use_soft_mask", True))
        self.temporal_beta = float(getattr(args, "temporal_decay_beta", 0.0))
        self.use_relative_time_encoding = bool(getattr(args, "use_relative_time_encoding", False))
        self.edge_prune_ratio = float(getattr(args, "edge_prune_ratio", 0.0))

        self.node_embedding = nn.Embedding(self.n_modals, self.input_dim)
        self.time_embedding = nn.Linear(1, self.input_dim)
        self.query = nn.Linear(self.input_dim, self.input_dim)
        self.key = nn.Linear(self.input_dim, self.input_dim)
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.scale = math.sqrt(self.input_dim)
        self.relative_time_bias = nn.Linear(1, 1) if self.use_relative_time_encoding else None

        self.edge_type_to_idx = self._build_relation_index()
        self.num_relations = max(1, len(self.edge_type_to_idx))
        dropout = getattr(args, "graph_drop_rate", None)
        if dropout is None:
            dropout = getattr(args, "drop_rate", 0.0)

        self.graph_branch = HAN(
            self.input_dim,
            self.hidden_dim,
            int(h2_dim),
            num_relations=self.num_relations,
            num_modals=self.n_modals,
            args=args,
            dropout=float(dropout),
        )

    def _build_relation_index(self) -> dict[tuple[int, int, int], int]:
        """Return relation ids keyed by (time_delta_sign, source_modal, target_modal)."""
        use_temp = "temp" in self.edge_type_mode
        use_multi = "multi" in self.edge_type_mode
        enable_full = bool(getattr(self.args, "enable_full_relations", False))
        relation_keys: list[tuple[int, int, int]] = []

        if enable_full and use_temp and use_multi:
            relation_keys = [
                (delta, source, target)
                for delta in (-1, 0, 1)
                for source in range(self.n_modals)
                for target in range(self.n_modals)
            ]
        elif use_temp:
            relation_keys = [
                (delta, modal, modal)
                for delta in (-1, 1)
                for modal in range(self.n_modals)
            ]
        elif use_multi:
            relation_keys = [
                (0, source, target)
                for source in range(self.n_modals)
                for target in range(self.n_modals)
                if source != target
            ]
        else:
            relation_keys = [(0, modal, modal) for modal in range(self.n_modals)]

        return {key: index for index, key in enumerate(relation_keys)}

    def generate_temporal_edges(self, text_len_tensor: torch.Tensor) -> torch.Tensor:
        """Create modality-major temporal-window edge candidates."""
        lengths = text_len_tensor.to(device=self.device, dtype=torch.long).reshape(-1)
        total_length = int(lengths.sum().item())
        if total_length <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        starts = torch.cat(
            [
                torch.zeros((1,), dtype=torch.long, device=self.device),
                torch.cumsum(lengths, dim=0)[:-1],
            ]
        )
        edges: list[tuple[int, int]] = []
        for modal_index in range(self.n_modals):
            modal_offset = modal_index * total_length
            for dialogue_index, length_tensor in enumerate(lengths):
                length = int(length_tensor.item())
                dialogue_offset = int((modal_offset + starts[dialogue_index]).item())
                for source_pos in range(length):
                    first = max(0, source_pos - self.window_past)
                    last = min(length, source_pos + self.window_future + 1)
                    for target_pos in range(first, last):
                        if target_pos != source_pos:
                            edges.append((dialogue_offset + source_pos, dialogue_offset + target_pos))

        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        return torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()

    def generate_edges(
        self,
        node_types: torch.Tensor,
        edge_candidates: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Assign each candidate edge to a HAN relation id."""
        edge_count = int(edge_candidates.size(1))
        if edge_count == 0:
            return torch.empty((0,), dtype=torch.long, device=edge_candidates.device)

        source = edge_candidates[0]
        target = edge_candidates[1]
        source_type = node_types.index_select(0, source).to(torch.long)
        target_type = node_types.index_select(0, target).to(torch.long)
        delta = self._time_delta_sign(timestamps, source, target)

        relation_ids = []
        for index in range(edge_count):
            key = (
                int(delta[index].item()),
                int(source_type[index].item()),
                int(target_type[index].item()),
            )
            relation_ids.append(self._lookup_relation(key))
        return torch.tensor(relation_ids, dtype=torch.long, device=edge_candidates.device)

    def _lookup_relation(self, key: tuple[int, int, int]) -> int:
        if key in self.edge_type_to_idx:
            return self.edge_type_to_idx[key]

        delta, source, target = key
        if "temp" in self.edge_type_mode and source == target:
            return self.edge_type_to_idx.get((1 if delta == 0 else delta, source, target), 0)
        if "multi" in self.edge_type_mode and source != target:
            return self.edge_type_to_idx.get((0, source, target), 0)
        return 0

    @staticmethod
    def _time_delta_sign(timestamps: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if timestamps.numel() == 0:
            return torch.zeros_like(source)
        time_vector = timestamps.reshape(-1)
        diff = time_vector.index_select(0, target) - time_vector.index_select(0, source)
        return torch.sign(diff).clamp(min=-1, max=1).to(torch.long)

    def _edge_time_distance(
        self,
        timestamps: torch.Tensor,
        edge_candidates: torch.Tensor,
    ) -> torch.Tensor:
        if edge_candidates.numel() == 0 or timestamps.numel() == 0:
            return torch.zeros((edge_candidates.size(1), 1), dtype=torch.float32, device=self.device)
        time_vector = timestamps.reshape(-1)
        distance = torch.abs(
            time_vector.index_select(0, edge_candidates[1])
            - time_vector.index_select(0, edge_candidates[0])
        )
        return distance.unsqueeze(-1)

    def _edge_weights(
        self,
        x: torch.Tensor,
        edge_candidates: torch.Tensor,
        time_distance: torch.Tensor,
    ) -> torch.Tensor:
        if edge_candidates.numel() == 0:
            return x.new_empty((0,))

        source = edge_candidates[0]
        target = edge_candidates[1]
        scores = (
            self.query(x).index_select(0, target)
            * self.key(x).index_select(0, source)
        ).sum(dim=-1) / self.scale

        if self.relative_time_bias is not None:
            scores = scores + self.relative_time_bias(time_distance.to(dtype=x.dtype)).squeeze(-1)

        weights = torch.sigmoid(scores)
        if self.temporal_beta > 0.0:
            weights = weights * torch.exp(-self.temporal_beta * time_distance.squeeze(-1).to(dtype=x.dtype))
        return weights

    def _select_edges(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return edge_index, edge_type, edge_weight

        if not self.use_soft_mask:
            keep_mask = edge_weight > torch.sigmoid(self.threshold)
            edge_index = edge_index[:, keep_mask]
            edge_type = edge_type[keep_mask]
            edge_weight = edge_weight[keep_mask]

        if self.edge_prune_ratio <= 0.0 or edge_index.size(1) <= 1:
            return edge_index, edge_type, edge_weight

        keep_count = max(1, int(edge_index.size(1) * (1.0 - self.edge_prune_ratio)))
        keep_count = min(keep_count, int(edge_index.size(1)))
        if keep_count == edge_index.size(1):
            return edge_index, edge_type, edge_weight

        keep_index = torch.topk(edge_weight, k=keep_count, largest=True).indices
        return edge_index[:, keep_index], edge_type[keep_index], edge_weight[keep_index]

    @staticmethod
    def _node_attention_mask(
        num_nodes: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros((int(num_nodes),), dtype=dtype, device=device)
        degree = torch.zeros_like(mask)
        if edge_index.numel() > 0:
            source = edge_index[0]
            target = edge_index[1]
            mask.scatter_add_(0, source, edge_weight)
            mask.scatter_add_(0, target, edge_weight)
            ones = torch.ones_like(edge_weight)
            degree.scatter_add_(0, source, ones)
            degree.scatter_add_(0, target, ones)
        return (mask / degree.clamp_min(1.0)).unsqueeze(-1)

    def forward(
        self,
        multimodal_features: torch.Tensor,
        node_types: torch.Tensor,
        timestamps: torch.Tensor,
        edge_candidates: torch.Tensor,
    ) -> torch.Tensor:
        if multimodal_features.dim() != 2:
            raise ValueError(f"multimodal_features must be 2D, got {tuple(multimodal_features.shape)}")
        if node_types.numel() != multimodal_features.size(0):
            raise ValueError("node_types length must match the number of graph nodes")
        if timestamps.reshape(-1).numel() != multimodal_features.size(0):
            raise ValueError("timestamps length must match the number of graph nodes")

        features = multimodal_features.to(self.device)
        node_types = node_types.to(device=self.device, dtype=torch.long)
        timestamps = timestamps.to(device=self.device, dtype=features.dtype).reshape(-1, 1)
        edge_candidates = edge_candidates.to(device=self.device, dtype=torch.long)

        node_emb = self.node_embedding(node_types)
        if bool(getattr(self.args, "no_time_node_embedding", False)):
            time_emb = torch.zeros_like(features)
        else:
            time_emb = self.time_embedding(timestamps)
        x = features + node_emb + time_emb

        edge_type = self.generate_edges(node_types, edge_candidates, timestamps)
        time_distance = self._edge_time_distance(timestamps, edge_candidates).to(device=self.device, dtype=features.dtype)
        edge_weight = self._edge_weights(x, edge_candidates, time_distance)
        edge_index, edge_type, edge_weight = self._select_edges(edge_candidates, edge_type, edge_weight)
        attn_mask = self._node_attention_mask(x.size(0), edge_index, edge_weight, x.dtype, x.device)

        return self.graph_branch(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            attn_mask=attn_mask,
            edge_weight=edge_weight,
        )
