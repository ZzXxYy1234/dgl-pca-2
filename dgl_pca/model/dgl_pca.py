"""Main DGL-PCA model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl_pca.utils.logging import get_logger

from .classifier import Classifier
from .crossmodal import CrossmodalNet
from .functions import feature_packing, multi_concat
from .graph_model import GraphModel
from .unimodal_encoder import UnimodalEncoder


log = get_logger(__name__)


class DGLPCA(nn.Module):
    """DGL-PCA for utterance-level multimodal conversation classification."""

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = args.device if isinstance(args.device, torch.device) else torch.device(args.device)
        self.args.device = self.device
        self.modalities = str(args.modalities)
        self.n_modals = len(self.modalities)
        if not self.modalities:
            raise ValueError("DGL-PCA requires at least one modality")

        self.hidden_size = int(args.hidden_size)
        self.graph_hidden_size = int(getattr(args, "graph_hidden_size", None) or self.hidden_size)
        self.graph_weight = float(getattr(args, "graph_weight", 1.0))
        self.use_graph = not bool(getattr(args, "no_gnn", False))
        self.use_crossmodal = bool(getattr(args, "use_crossmodal", False)) and self.n_modals > 1
        self.use_speaker = bool(getattr(args, "use_speaker", True))

        feature_dims = args.dataset_embedding_dims[args.dataset]
        self.encoder = UnimodalEncoder(
            feature_dims["a"],
            feature_dims["t"],
            feature_dims["v"],
            self.hidden_size,
            args,
        )

        self.n_speakers = self._speaker_count(str(args.dataset))
        self.speaker_embedding = nn.Embedding(self.n_speakers, self.hidden_size) if self.use_speaker else None

        input_dim = self._classifier_input_dim(args)
        self.graph_model = None
        if self.use_graph:
            self.graph_model = GraphModel(
                self.hidden_size,
                self.graph_hidden_size,
                self.graph_hidden_size,
                self.n_modals,
                self.device,
                args,
            )
            log.info("DGL-PCA graph branch: hand-written HAN")

        self.crossmodal = CrossmodalNet(self.hidden_size, args) if self.use_crossmodal else None
        if bool(getattr(args, "use_crossmodal", False)) and not self.use_crossmodal:
            log.info("Cross-modal branch disabled because only one modality is active")

        self.clf = Classifier(input_dim, self.hidden_size, self._label_count(args), args)
        self.rlog: dict[str, object] = {}
        log.info(
            "Initialized DGL-PCA: dataset=%s, task=%s, labels=%s, speakers=%s, modalities=%s",
            args.dataset,
            getattr(args, "task_type", "unknown"),
            self.clf.output_dim,
            self.n_speakers,
            self.modalities,
        )

    @staticmethod
    def _speaker_count(dataset: str) -> int:
        return 2 if dataset.lower() == "iemocap" else 1

    @staticmethod
    def _label_count(args) -> int:
        if hasattr(args, "class_num"):
            return int(args.class_num)
        if getattr(args, "task_type", "sentiment") == "emotion":
            return 6
        return 2

    def _classifier_input_dim(self, args) -> int:
        input_dim = 0
        if self.use_graph:
            graph_dim = self.graph_hidden_size
            if bool(getattr(args, "use_graph_transformer", False)):
                graph_dim *= int(getattr(args, "graph_transformer_nheads", 1))
            input_dim += graph_dim * self.n_modals

        if self.use_crossmodal:
            input_dim += self.hidden_size * self.n_modals * (self.n_modals - 1)

        if not self.use_graph and not self.use_crossmodal:
            input_dim = self.hidden_size * self.n_modals
        if input_dim <= 0:
            raise ValueError("DGL-PCA classifier input dimension resolved to zero")
        return input_dim

    def represent(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return fused utterance representations for the enabled branches."""
        lengths = data["text_len_tensor"].to(self.device)
        encoded = self._encode_modalities(data, lengths)
        parts: list[torch.Tensor] = []

        if self.use_graph:
            if self.graph_model is None:
                raise RuntimeError("graph branch is enabled but graph_model was not initialized")
            graph_nodes, node_types, timestamps = self._pack_graph_nodes(encoded, lengths, data["timestamps"])
            edge_candidates = self.graph_model.generate_temporal_edges(lengths)
            graph_out = self.graph_model(
                multimodal_features=graph_nodes,
                node_types=node_types,
                timestamps=timestamps,
                edge_candidates=edge_candidates,
            )
            graph_feat = multi_concat(graph_out, lengths, self.n_modals)
            if self.graph_weight != 1.0:
                graph_feat = graph_feat * self.graph_weight
            parts.append(graph_feat)
            self._record_scalar("graph_feat_abs_mean", graph_feat.abs().mean())
            self._record_scalar("graph_weight", torch.tensor(self.graph_weight, device=self.device))

        if self.use_crossmodal:
            if self.crossmodal is None:
                raise RuntimeError("cross-modal branch is enabled but crossmodal was not initialized")
            modal_inputs = [encoded[modal] for modal in self.modalities]
            crossmodal_seq = self.crossmodal(modal_inputs).transpose(0, 1).contiguous()
            crossmodal_feat = self._pack_batch_major(crossmodal_seq, lengths)
            pre_mean = crossmodal_feat.abs().mean()
            crossmodal_feat = self.apply_pcm_gates(crossmodal_feat)
            self.rlog["pcm_cr_feat_mean"] = {
                "pre": float(pre_mean.item()),
                "post": float(crossmodal_feat.abs().mean().item()),
            }
            self._record_scalar("pcm_feat_abs_mean", crossmodal_feat.abs().mean())
            parts.append(crossmodal_feat)

        if not self.use_graph and not self.use_crossmodal:
            modal_inputs = [encoded[modal] for modal in self.modalities]
            packed = feature_packing(modal_inputs, lengths)
            parts.append(multi_concat(packed, lengths, self.n_modals))

        fused = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        self._record_scalar("fused_feat_abs_mean", fused.abs().mean())
        return fused

    def _encode_modalities(
        self,
        data: dict[str, torch.Tensor],
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        audio = data["audio_tensor"].to(self.device) if "a" in self.modalities else None
        text = data["text_tensor"].to(self.device) if "t" in self.modalities else None
        visual = data["visual_tensor"].to(self.device) if "v" in self.modalities else None

        audio, text, visual = self.encoder(audio, text, visual, lengths)
        encoded = {"a": audio, "t": text, "v": visual}
        encoded = {modal: tensor.to(self.device) for modal, tensor in encoded.items() if tensor is not None}

        if self.speaker_embedding is not None:
            speaker = data["speaker_tensor"].to(self.device)
            speaker_emb = self.speaker_embedding(speaker)
            encoded = {modal: tensor + speaker_emb for modal, tensor in encoded.items()}

        missing = [modal for modal in self.modalities if modal not in encoded]
        if missing:
            raise ValueError(f"Missing encoded modalities: {missing}")
        return encoded

    def _pack_graph_nodes(
        self,
        encoded: dict[str, torch.Tensor],
        lengths: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(lengths.numel())
        timestamp_tensor = timestamps.to(self.device)
        node_features = []
        node_types = []
        node_times = []
        for modal_index, modal in enumerate(self.modalities):
            feature = encoded[modal]
            for batch_index in range(batch_size):
                length = int(lengths[batch_index].item())
                node_features.append(feature[batch_index, :length])
                node_types.append(
                    torch.full((length,), modal_index, dtype=torch.long, device=self.device)
                )
                node_times.append(timestamp_tensor[batch_index, :length])

        return (
            torch.cat(node_features, dim=0),
            torch.cat(node_types, dim=0),
            torch.cat(node_times, dim=0).reshape(-1, 1),
        )

    def _pack_batch_major(self, sequence_tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = []
        for batch_index, length_tensor in enumerate(lengths):
            packed.append(sequence_tensor[batch_index, : int(length_tensor.item())])
        return torch.cat(packed, dim=0).to(self.device)

    def apply_pcm_gates(self, crossmodal_features: torch.Tensor) -> torch.Tensor:
        """Apply optional per-modality gates to the cross-modal feature blocks."""
        gate_map = getattr(self.args, "pcm_gate_map", None)
        if not gate_map:
            return crossmodal_features

        block_count = self.n_modals
        feature_dim = int(crossmodal_features.size(-1))
        if feature_dim % block_count != 0:
            raise ValueError("cross-modal feature dimension must be divisible by the number of modalities")

        block_size = feature_dim // block_count
        gated = crossmodal_features.clone()
        debug = {"modalities": self.modalities, "weights": {}, "block_size": block_size}
        for index, modal in enumerate(self.modalities):
            weight = float(gate_map.get(modal, 1.0))
            debug["weights"][modal] = weight
            start = index * block_size
            end = start + block_size
            gated[:, start:end] = gated[:, start:end] * weight
        self.rlog["pcm_gate_debug"] = debug
        return gated

    def _record_scalar(self, key: str, value: torch.Tensor) -> None:
        if bool(getattr(self.args, "monitor_graph_pcm_norms", False)):
            self.rlog[key] = float(value.detach().item())

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.clf(self.represent(data), data["text_len_tensor"])

    def get_loss(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        representations = self.represent(data)
        loss = self.clf.get_loss(representations, data["label_tensor"], data["text_len_tensor"])
        return loss + self._consistency_loss(representations, data)

    def _consistency_loss(self, representations: torch.Tensor, data: dict[str, torch.Tensor]) -> torch.Tensor:
        coefficient = float(getattr(self.args, "consistency_lambda", 0.0))
        if coefficient <= 0.0 or bool(getattr(self.args, "multilabel", False)):
            return torch.tensor(0.0, device=self.device)

        _, scores = self.clf.get_prob(representations, data["text_len_tensor"])
        probabilities = torch.softmax(scores, dim=-1)
        lengths = data["text_len_tensor"].to(self.device)
        speakers = data["speaker_tensor"].to(self.device)
        timestamps = data["timestamps"].to(self.device)

        beta = float(getattr(self.args, "consistency_beta", 0.5))
        confidence_threshold = float(getattr(self.args, "consistency_conf_thr", 0.8))
        window = int(getattr(self.args, "consistency_window", 1))
        total = torch.tensor(0.0, device=self.device)
        offset = 0
        for batch_index, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            dialogue_probs = probabilities[offset : offset + length]
            dialogue_speakers = speakers[batch_index, :length]
            dialogue_times = timestamps[batch_index, :length]
            for source in range(length):
                for step in range(1, window + 1):
                    target = source + step
                    if target >= length:
                        break
                    if dialogue_speakers[source] != dialogue_speakers[target]:
                        continue
                    distance = torch.abs(dialogue_times[target] - dialogue_times[source])
                    time_weight = torch.exp(-beta * distance)
                    confidence_weight = (
                        1.0
                        if dialogue_probs[source].max() < confidence_threshold
                        and dialogue_probs[target].max() < confidence_threshold
                        else 0.5
                    )
                    total = total + self._symmetric_kl(
                        dialogue_probs[source],
                        dialogue_probs[target],
                    ) * time_weight * confidence_weight
            offset += length

        return coefficient * total

    @staticmethod
    def _symmetric_kl(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first_log = torch.log(first.clamp_min(1e-8))
        second_log = torch.log(second.clamp_min(1e-8))
        return F.kl_div(first_log, second, reduction="sum") + F.kl_div(second_log, first, reduction="sum")

    def get_log(self) -> dict[str, object]:
        return self.rlog
