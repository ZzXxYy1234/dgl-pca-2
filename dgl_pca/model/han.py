"""Hand-written heterogeneous attention layers used by DGL-PCA."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _target_softmax(scores: torch.Tensor, targets: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Normalize edge scores over incoming edges of each target node."""
    if scores.numel() == 0:
        return scores

    max_per_target = torch.full(
        (int(num_nodes),),
        float("-inf"),
        dtype=scores.dtype,
        device=scores.device,
    )
    max_per_target.scatter_reduce_(0, targets, scores, reduce="amax", include_self=True)

    stabilized = scores - max_per_target[targets]
    exp_scores = torch.exp(stabilized)
    denom = torch.zeros((int(num_nodes),), dtype=scores.dtype, device=scores.device)
    denom.scatter_add_(0, targets, exp_scores)
    return exp_scores / denom[targets].clamp_min(1e-12)


class RelationAttention(nn.Module):
    """Node-level attention for one HAN relation/meta-path."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.activation = nn.LeakyReLU(0.2)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0 or edge_index.size(1) == 0:
            return torch.zeros_like(x), x.new_empty((0,))

        source = edge_index[0]
        target = edge_index[1]
        query = self.query(x).index_select(0, target)
        key = self.key(x).index_select(0, source)
        value = self.value(x).index_select(0, source)

        scores = self.activation((query * key).sum(dim=-1) / self.scale)
        weights = self.dropout(_target_softmax(scores, target, x.size(0)))

        out = torch.zeros_like(x)
        out.index_add_(0, target, value * weights.unsqueeze(-1))
        return out, weights


class SemanticAttention(nn.Module):
    """Attention over relation-specific node embeddings."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(int(hidden_dim), 1)

    def forward(self, relation_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(relation_outputs).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        return torch.sum(relation_outputs * weights.unsqueeze(-1), dim=0), weights


class HANLayer(nn.Module):
    """One DGL-PCA HAN layer with relation-level and semantic-level attention."""

    def __init__(self, in_dim: int, hidden_dim: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        self.num_relations = int(num_relations)
        self.input_projection = nn.Linear(int(in_dim), int(hidden_dim))
        self.relation_attention = RelationAttention(int(hidden_dim), float(dropout))
        self.semantic_attention = SemanticAttention(int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        relation_edges: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        base = self.activation(self.input_projection(self.dropout(x)))
        if not relation_edges or all(edge_index.numel() == 0 for edge_index in relation_edges):
            empty = base.new_empty((0, base.size(0)))
            return base, [], empty

        relation_outputs = []
        relation_weights = []
        for edge_index in relation_edges:
            out, weights = self.relation_attention(base, edge_index)
            relation_outputs.append(out)
            relation_weights.append(weights)

        stacked = torch.stack(relation_outputs, dim=0)
        out, semantic_weights = self.semantic_attention(stacked)
        return out, relation_weights, semantic_weights


class WeightedMessageProjection(nn.Module):
    """Edge-weighted projection used when the graph branch is configured with heads."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.message_layers = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(self.heads)
        )
        self.root_layers = nn.ModuleList(nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.heads))
        self.dropout = nn.Dropout(float(dropout))

    @property
    def output_dim(self) -> int:
        return self.hidden_dim * self.heads

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        source = edge_index[0] if edge_index.numel() else edge_index.new_empty((0,))
        target = edge_index[1] if edge_index.numel() else edge_index.new_empty((0,))
        if edge_weight is None:
            weights = x.new_ones((source.numel(),))
        else:
            weights = edge_weight.to(device=x.device, dtype=x.dtype).reshape(-1)

        head_outputs = []
        for message_layer, root_layer in zip(self.message_layers, self.root_layers):
            out = root_layer(x)
            if source.numel() > 0:
                messages = message_layer(x).index_select(0, source)
                messages = messages * weights.unsqueeze(-1)
                out.index_add_(0, target, messages)
            head_outputs.append(out)
        return self.dropout(torch.cat(head_outputs, dim=-1))


class HAN(nn.Module):
    """DGL-PCA graph branch: a pure PyTorch implementation of HAN."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_relations: int,
        num_modals: int,
        args,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        del out_dim, num_modals
        self.num_relations = int(num_relations)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(getattr(args, "han_layers", 1)))
        self.use_residual = bool(getattr(args, "use_han_residual", True))
        self.dropout_rate = float(dropout if dropout is not None else getattr(args, "drop_rate", 0.0))

        layers = []
        current_dim = int(in_dim)
        for _ in range(self.num_layers):
            layers.append(HANLayer(current_dim, self.hidden_dim, self.num_relations, self.dropout_rate))
            current_dim = self.hidden_dim
        self.layers = nn.ModuleList(layers)

        self.output_projection = None
        heads = int(getattr(args, "graph_transformer_nheads", 1))
        if bool(getattr(args, "use_graph_transformer", False)):
            self.output_projection = WeightedMessageProjection(self.hidden_dim, max(1, heads), self.dropout_rate)
            self.output_norm = nn.LayerNorm(self.output_projection.output_dim)
            self.final_dim = self.output_projection.output_dim
        else:
            self.output_norm = nn.Identity()
            self.final_dim = self.hidden_dim

    def build_relation_edges(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_relations: int | None = None,
    ) -> list[torch.Tensor]:
        """Split a batched edge index into relation-specific edge indices."""
        relation_count = self.num_relations if num_relations is None else int(num_relations)
        if edge_index.numel() == 0:
            return [
                torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
                for _ in range(relation_count)
            ]

        return [
            edge_index[:, edge_type == relation_id]
            for relation_id in range(relation_count)
        ]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        attn_mask: torch.Tensor | None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        relation_edges = self.build_relation_edges(edge_index, edge_type)
        hidden = x
        for layer in self.layers:
            previous = hidden
            hidden, _, _ = layer(hidden, relation_edges)
            if self.use_residual and previous.shape == hidden.shape:
                hidden = hidden + previous

        if self.output_projection is not None:
            hidden = self.output_projection(hidden, edge_index, edge_weight)
            hidden = self.output_norm(hidden)

        if attn_mask is not None:
            hidden = hidden * attn_mask.to(device=hidden.device, dtype=hidden.dtype)
        return hidden
