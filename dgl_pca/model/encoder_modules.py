"""Reusable unimodal sequence encoders for DGL-PCA."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def choose_attention_heads(embed_dim: int, requested: int | None = None) -> int:
    """Choose a valid number of attention heads for a transformer layer."""
    if requested is not None and requested > 0 and embed_dim % requested == 0:
        return int(requested)
    if requested is None:
        for heads in (8, 4, 2, 1):
            if heads <= embed_dim and embed_dim % heads == 0:
                return heads
    upper = min(embed_dim, requested or 8)
    for heads in range(upper, 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1


class FeedForwardEncoder(nn.Module):
    """Position-wise two-layer projection for padded utterance features."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        _ = lengths
        return self.net(x)


class SinusoidalPositionEncoding(nn.Module):
    """Batch-first sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        encoding = torch.zeros(max_len, dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        if dim > 1:
            encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1), :].to(dtype=x.dtype, device=x.device)


class SequenceTransformerEncoder(nn.Module):
    """Transformer encoder with padding masks derived from utterance lengths."""

    def __init__(self, input_dim: int, hidden_dim: int, args) -> None:
        super().__init__()
        dropout = float(getattr(args, "drop_rate", 0.0))
        layers = int(getattr(args, "encoder_nlayers", 1))
        requested_heads = getattr(args, "encoder_nheads", None)
        heads = choose_attention_heads(int(hidden_dim), int(requested_heads) if requested_heads is not None else None)

        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.position = SinusoidalPositionEncoding(int(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=heads,
            dim_feedforward=max(int(hidden_dim) * 4, 4),
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.position(self.input_proj(x))
        padding_mask = _padding_mask(lengths, max_len=x.size(1), device=x.device) if lengths is not None else None
        return self.encoder(hidden, src_key_padding_mask=padding_mask)


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder with projection to the requested hidden size."""

    def __init__(self, input_dim: int, hidden_dim: int, args) -> None:
        super().__init__()
        layers = int(getattr(args, "encoder_nlayers", 1))
        dropout = float(getattr(args, "drop_rate", 0.0)) if layers > 1 else 0.0
        recurrent_dim = max(1, int(hidden_dim) // 2)
        self.lstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=recurrent_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.output_proj = nn.Linear(recurrent_dim * 2, int(hidden_dim))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            out, _ = self.lstm(x)
        else:
            packed = pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        return self.output_proj(out)


def _padding_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    return positions >= lengths.to(device=device).unsqueeze(1)


# Compatibility aliases for older imports.
SeqTransfomer = SequenceTransformerEncoder
LSTM_Layer = LSTMEncoder
FC_with_PE = FeedForwardEncoder
PositionalEncoder = SinusoidalPositionEncoding
