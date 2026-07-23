"""Model-ready dialogue batching for DGL-PCA."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


MODALITY_TO_FIELD = {"a": "audio", "t": "text", "v": "visual"}


@dataclass(frozen=True)
class DialogueSampleView:
    """Normalized view over a dict-like or object-like dialogue sample."""

    audio: Sequence[Any]
    text: Sequence[Any]
    visual: Sequence[Any]
    speakers: Sequence[Any]
    labels: Any
    sentence: Any
    timestamps: Sequence[Any] | None


class DialogueDataset:
    """Batch dialogue samples and pad utterance-level multimodal features."""

    def __init__(self, samples: Sequence[Any], args: Any) -> None:
        self.samples = list(samples)
        self.modalities = str(args.modalities)
        self.batch_size = int(args.batch_size)
        self.dataset = str(args.dataset)
        self.embedding_dim = args.dataset_embedding_dims[self.dataset]
        self.multilabel = bool(getattr(args, "multilabel", False))
        self.class_num = int(getattr(args, "class_num", 0) or 0)
        self.speaker_to_idx = {"M": 0, "F": 1, "0": 0, "1": 1, 0: 0, 1: 1}
        self.num_batches = math.ceil(len(self.samples) / self.batch_size) if self.batch_size > 0 else 0

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.padding(self.raw_batch(index))

    def raw_batch(self, index: int) -> list[Any]:
        if index < 0 or index >= self.num_batches:
            raise IndexError(f"batch index {index} is outside [0, {self.num_batches})")
        start = index * self.batch_size
        return self.samples[start : start + self.batch_size]

    def padding(self, samples: Sequence[Any]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot pad an empty batch")

        views = [self._view_sample(sample) for sample in samples]
        lengths = [self._validate_lengths(view) for view in views]
        max_len = max(lengths)
        batch_size = len(views)

        text_tensor = torch.zeros((batch_size, max_len, int(self.embedding_dim["t"])), dtype=torch.float32)
        audio_tensor = torch.zeros((batch_size, max_len, int(self.embedding_dim["a"])), dtype=torch.float32)
        visual_tensor = torch.zeros((batch_size, max_len, int(self.embedding_dim["v"])), dtype=torch.float32)
        speaker_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
        timestamps_tensor = torch.zeros((batch_size, max_len), dtype=torch.float32)
        utterance_texts = []
        label_tensors = []

        for row, (view, length) in enumerate(zip(views, lengths)):
            text_tensor[row, :length] = self._feature_tensor(view.text, "t", length)
            audio_tensor[row, :length] = self._feature_tensor(view.audio, "a", length)
            visual_tensor[row, :length] = self._feature_tensor(view.visual, "v", length)
            speaker_tensor[row, :length] = self._speaker_tensor(view.speakers, length)
            timestamps_tensor[row, :length] = self._timestamp_tensor(view.timestamps, length)
            utterance_texts.append(view.sentence)
            label_tensors.append(self._label_tensor(view.labels, length))

        label_tensor = self._merge_labels(label_tensors)
        return {
            "text_len_tensor": torch.tensor(lengths, dtype=torch.long),
            "text_tensor": text_tensor,
            "audio_tensor": audio_tensor,
            "visual_tensor": visual_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "utterance_texts": utterance_texts,
            "timestamps": timestamps_tensor,
        }

    def shuffle(self) -> None:
        random.shuffle(self.samples)

    def _view_sample(self, sample: Any) -> DialogueSampleView:
        text = self._get(sample, "text")
        speakers = self._get(sample, "speakers", default=None)
        length = len(text)
        if speakers is None:
            speakers = ["M"] * length
        return DialogueSampleView(
            audio=self._get(sample, "audio"),
            text=text,
            visual=self._get(sample, "visual"),
            speakers=speakers,
            labels=self._get(sample, "labels"),
            sentence=self._get(sample, "sentence", default=None),
            timestamps=self._get(sample, "timestamps", default=None),
        )

    @staticmethod
    def _get(sample: Any, field: str, default: Any = ...):
        if isinstance(sample, dict):
            if field in sample:
                return sample[field]
        elif hasattr(sample, field):
            return getattr(sample, field)

        if default is ...:
            raise KeyError(f"Dialogue sample is missing required field: {field}")
        return default

    def _validate_lengths(self, view: DialogueSampleView) -> int:
        length = len(view.text)
        if length == 0:
            raise ValueError("Dialogue sample contains no utterances")

        for modality in self.modalities:
            field = MODALITY_TO_FIELD.get(modality)
            if field is None:
                raise ValueError(f"Unsupported modality: {modality}")
            value = getattr(view, field)
            if len(value) != length:
                raise ValueError(f"{field} length {len(value)} does not match text length {length}")

        if len(view.speakers) != length:
            raise ValueError(f"speakers length {len(view.speakers)} does not match text length {length}")
        if view.timestamps is not None and len(view.timestamps) != length:
            raise ValueError(f"timestamps length {len(view.timestamps)} does not match text length {length}")
        return length

    def _feature_tensor(self, values: Sequence[Any], modality: str, length: int) -> torch.Tensor:
        expected_dim = int(self.embedding_dim[modality])
        tensor = torch.as_tensor(np.asarray(values), dtype=torch.float32)
        if tensor.shape != (length, expected_dim):
            raise ValueError(
                f"{MODALITY_TO_FIELD[modality]} feature shape {tuple(tensor.shape)} "
                f"does not match expected {(length, expected_dim)}"
            )
        return tensor

    def _speaker_tensor(self, speakers: Sequence[Any], length: int) -> torch.Tensor:
        indices = []
        for speaker in speakers:
            if speaker in self.speaker_to_idx:
                indices.append(self.speaker_to_idx[speaker])
            elif self.dataset == "mosei":
                indices.append(0)
            else:
                raise ValueError(f"Unsupported speaker label for {self.dataset}: {speaker!r}")
        tensor = torch.tensor(indices, dtype=torch.long)
        if tensor.numel() != length:
            raise ValueError("speaker tensor length mismatch")
        return tensor

    @staticmethod
    def _timestamp_tensor(timestamps: Sequence[Any] | None, length: int) -> torch.Tensor:
        if timestamps is None:
            return torch.arange(length, dtype=torch.float32)
        tensor = torch.as_tensor(timestamps, dtype=torch.float32).reshape(-1)
        if tensor.numel() != length:
            raise ValueError("timestamp tensor length mismatch")
        return tensor

    def _label_tensor(self, labels: Any, length: int) -> torch.Tensor:
        tensor = torch.as_tensor(labels, dtype=torch.float32 if self.multilabel else torch.long)
        if not self.multilabel:
            tensor = tensor.reshape(-1)
            if tensor.numel() != length:
                raise ValueError(f"single-label sample has {tensor.numel()} labels for {length} utterances")
            return tensor.long()

        if tensor.dim() == 1:
            if self.class_num and tensor.numel() == self.class_num:
                return tensor.float().reshape(1, -1)
            if self.class_num and tensor.numel() == length * self.class_num:
                return tensor.float().reshape(length, self.class_num)
        elif tensor.dim() == 2:
            if tensor.size(0) in (1, length):
                return tensor.float()

        raise ValueError(f"unsupported multilabel shape {tuple(tensor.shape)} for dialogue length {length}")

    def _merge_labels(self, labels: Sequence[torch.Tensor]) -> torch.Tensor:
        if not self.multilabel:
            return torch.cat([label.reshape(-1).long() for label in labels], dim=0)

        if all(label.dim() == 2 and label.size(0) == 1 for label in labels):
            return torch.cat(labels, dim=0).float()
        return torch.cat(labels, dim=0).float()
