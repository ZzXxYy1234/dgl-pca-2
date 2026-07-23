"""Pairwise cross-modal attention branch for DGL-PCA."""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder_modules import choose_attention_heads


class AttentionBlock(nn.Module):
    """Residual multi-head attention followed by a feed-forward block."""

    def __init__(self, embed_dim: int, num_heads: int, layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_AttentionLayer(embed_dim, num_heads, dropout=dropout) for _ in range(max(1, int(layers)))]
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = query
        key = output if key is None else key
        value = key if value is None else value
        for layer in self.layers:
            output = layer(output, key, value)
            key = output if key is query else key
            value = output if value is query else value
        return output


class _AttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        heads = choose_attention_heads(int(embed_dim), int(num_heads))
        self.attention = nn.MultiheadAttention(int(embed_dim), heads, dropout=float(dropout), batch_first=True)
        self.attn_norm = nn.LayerNorm(int(embed_dim))
        self.ffn = nn.Sequential(
            nn.Linear(int(embed_dim), int(embed_dim) * 4),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(embed_dim) * 4, int(embed_dim)),
        )
        self.ffn_norm = nn.LayerNorm(int(embed_dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(query, key, value, need_weights=False)
        hidden = self.attn_norm(query + self.dropout(attended))
        return self.ffn_norm(hidden + self.dropout(self.ffn(hidden)))


class CrossmodalNet(nn.Module):
    """Compute pairwise cross-modal context and per-modality memory refinement."""

    def __init__(self, inchannels: int, args) -> None:
        super().__init__()
        self.modalities = str(args.modalities)
        self.hidden_dim = int(inchannels)
        self.n_modals = len(self.modalities)
        if self.n_modals < 2:
            raise ValueError("CrossmodalNet requires at least two modalities")

        dropout = float(getattr(args, "drop_rate", 0.0))
        cross_heads = choose_attention_heads(self.hidden_dim, int(getattr(args, "crossmodal_nheads", 2)))
        cross_layers = int(getattr(args, "num_crossmodal", 1))
        memory_dim = self.hidden_dim * (self.n_modals - 1)
        memory_heads = choose_attention_heads(memory_dim, int(getattr(args, "self_att_nheads", 2)))
        memory_layers = int(getattr(args, "num_self_att", 1))

        self.cross_layers = nn.ModuleDict()
        self.memory_layers = nn.ModuleDict()
        for target in self.modalities:
            for source in self.modalities:
                if target != source:
                    self.cross_layers[f"{target}_{source}"] = AttentionBlock(
                        self.hidden_dim,
                        cross_heads,
                        layers=cross_layers,
                        dropout=dropout,
                    )
            self.memory_layers[target] = AttentionBlock(
                memory_dim,
                memory_heads,
                layers=memory_layers,
                dropout=dropout,
            )

    def forward(self, modal_features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Return sequence-first fused cross-modal features."""
        if len(modal_features) != self.n_modals:
            raise ValueError(f"Expected {self.n_modals} modalities, got {len(modal_features)}")

        features = {modal: tensor for modal, tensor in zip(self.modalities, modal_features)}
        batch_size, seq_len = _shared_batch_shape(features)
        _ = (batch_size, seq_len)

        refined = []
        for target in self.modalities:
            pair_outputs = []
            for source in self.modalities:
                if target == source:
                    continue
                pair_outputs.append(
                    self.cross_layers[f"{target}_{source}"](
                        features[target],
                        features[source],
                        features[source],
                    )
                )
            memory_input = torch.cat(pair_outputs, dim=-1)
            refined.append(self.memory_layers[target](memory_input))

        fused = torch.cat(refined, dim=-1)
        return fused.transpose(0, 1).contiguous()


def _shared_batch_shape(features: dict[str, torch.Tensor]) -> tuple[int, int]:
    shapes = {name: tuple(tensor.shape) for name, tensor in features.items()}
    first_name = next(iter(features))
    first = features[first_name]
    if first.dim() != 3:
        raise ValueError(f"Modal feature {first_name!r} must be 3D, got {tuple(first.shape)}")
    batch_size, seq_len, hidden_dim = first.shape
    for name, tensor in features.items():
        if tensor.dim() != 3:
            raise ValueError(f"Modal feature {name!r} must be 3D, got {tuple(tensor.shape)}")
        if tensor.shape[:2] != (batch_size, seq_len):
            raise ValueError(f"Modal feature shapes disagree: {shapes}")
        if tensor.size(-1) != hidden_dim:
            raise ValueError(f"Modal feature hidden dimensions disagree: {shapes}")
    return int(batch_size), int(seq_len)
