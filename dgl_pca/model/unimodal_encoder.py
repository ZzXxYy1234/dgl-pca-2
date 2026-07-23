"""Audio/text/visual encoders for DGL-PCA."""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder_modules import FeedForwardEncoder, LSTMEncoder, SequenceTransformerEncoder


class UnimodalEncoder(nn.Module):
    """Encode each enabled modality into the shared hidden dimension."""

    def __init__(self, a_dim: int, t_dim: int, v_dim: int, h_dim: int, args) -> None:
        super().__init__()
        self.hidden_dim = int(h_dim)
        self.text_mode = str(getattr(args, "rnn", "transformer")).lower()
        dropout = float(getattr(args, "drop_rate", 0.0))

        self.audio_encoder = FeedForwardEncoder(int(a_dim), self.hidden_dim, dropout=dropout)
        self.text_encoder = self._build_text_encoder(int(t_dim), self.hidden_dim, args)
        self.visual_encoder = FeedForwardEncoder(int(v_dim), self.hidden_dim, dropout=dropout)

    def forward(
        self,
        audio: torch.Tensor | None,
        text: torch.Tensor | None,
        visual: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Return encoded tensors in the same order as the inputs: audio, text, visual."""
        audio_out = self.audio_encoder(audio) if audio is not None else None
        text_out = self.text_encoder(text, lengths) if text is not None else None
        visual_out = self.visual_encoder(visual) if visual is not None else None
        return audio_out, text_out, visual_out

    def _build_text_encoder(self, input_dim: int, hidden_dim: int, args) -> nn.Module:
        if self.text_mode == "transformer":
            return SequenceTransformerEncoder(input_dim, hidden_dim, args)
        if self.text_mode == "lstm":
            return LSTMEncoder(input_dim, hidden_dim, args)
        if self.text_mode == "ffn":
            dropout = float(getattr(args, "drop_rate", 0.0))
            return FeedForwardEncoder(input_dim, hidden_dim, dropout=dropout)
        raise ValueError(f"Unsupported text encoder mode: {self.text_mode}")
