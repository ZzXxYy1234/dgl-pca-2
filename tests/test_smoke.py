from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from dgl_pca.data import DialogueDataset
from dgl_pca.data.io import save_pickle
from dgl_pca.data.loader import load_dialogue_splits
from dgl_pca.model.classifier import Classifier
from dgl_pca.model.crossmodal import CrossmodalNet
from dgl_pca.model.functions import feature_packing, multi_concat
from dgl_pca.model import DGLPCA
from dgl_pca.model.graph_model import GraphModel
from dgl_pca.model.unimodal_encoder import UnimodalEncoder
from dgl_pca.training.common import align_labels_for_predictions, compute_metrics
from dgl_pca.training.evaluate import run_smoke_test
from dgl_pca.utils.args import build_args_from_config
from dgl_pca.utils.config import load_config


class DGLPCASmokeTests(unittest.TestCase):
    def test_model_smoke_forward(self):
        run_smoke_test()

    def test_dialogue_dataset_padding(self):
        args = SimpleNamespace(
            modalities="atv",
            batch_size=2,
            dataset="mosei",
            dataset_embedding_dims={"mosei": {"a": 2, "t": 3, "v": 4}},
        )
        samples = [
            {
                "text": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "audio": [[0.1, 0.2], [0.3, 0.4]],
                "visual": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                "speakers": ["M", "M"],
                "labels": [1, 0],
                "sentence": ["u1", "u2"],
                "timestamps": [0.0, 1.0],
            },
            {
                "text": [[0.0, 0.0, 1.0]],
                "audio": [[0.5, 0.6]],
                "visual": [[0.9, 1.0, 1.1, 1.2]],
                "speakers": ["M"],
                "labels": [1],
                "sentence": ["u3"],
                "timestamps": [0.0],
            },
        ]
        dataset = DialogueDataset(samples, args)
        batch = dataset[0]
        self.assertEqual(tuple(batch["text_tensor"].shape), (2, 2, 3))
        self.assertEqual(tuple(batch["audio_tensor"].shape), (2, 2, 2))
        self.assertEqual(tuple(batch["visual_tensor"].shape), (2, 2, 4))
        self.assertTrue(torch.equal(batch["text_len_tensor"], torch.tensor([2, 1])))
        self.assertEqual(tuple(batch["label_tensor"].shape), (3,))

    def test_dialogue_dataset_object_sample_and_multilabel_dialogue_label(self):
        args = SimpleNamespace(
            modalities="atv",
            batch_size=1,
            dataset="mosei",
            class_num=3,
            multilabel=True,
            dataset_embedding_dims={"mosei": {"a": 2, "t": 2, "v": 2}},
        )
        sample = SimpleNamespace(
            text=[[1.0, 0.0], [0.0, 1.0]],
            audio=[[0.1, 0.2], [0.3, 0.4]],
            visual=[[0.4, 0.3], [0.2, 0.1]],
            speakers=["unknown", "unknown"],
            labels=[1, 0, 1],
            sentence=["u1", "u2"],
            timestamps=None,
        )
        batch = DialogueDataset([sample], args)[0]
        self.assertEqual(tuple(batch["label_tensor"].shape), (1, 3))
        self.assertTrue(torch.equal(batch["speaker_tensor"], torch.zeros((1, 2), dtype=torch.long)))
        self.assertTrue(torch.equal(batch["timestamps"], torch.tensor([[0.0, 1.0]], dtype=torch.float32)))

    def test_load_dialogue_splits_validates_required_splits(self):
        args = SimpleNamespace(
            modalities="atv",
            batch_size=1,
            dataset="mosei",
            dataset_embedding_dims={"mosei": {"a": 1, "t": 1, "v": 1}},
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pkl"
            save_pickle({"train": [], "dev": []}, path)
            with self.assertRaisesRegex(ValueError, "missing required splits"):
                load_dialogue_splits(path, args)

    def test_feature_packing_and_multi_concat(self):
        lengths = torch.tensor([2, 1])
        first = torch.tensor([[[1.0], [2.0]], [[3.0], [0.0]]])
        second = torch.tensor([[[10.0], [20.0]], [[30.0], [0.0]]])
        packed = feature_packing([first, second], lengths)
        self.assertTrue(torch.equal(packed.reshape(-1), torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])))
        fused = multi_concat(packed, lengths, n_modals=2)
        self.assertTrue(torch.equal(fused, torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])))

    def test_classifier_binary_threshold_and_loss(self):
        args = SimpleNamespace(
            class_num=2,
            loss_type="nll",
            label_smoothing=0.0,
            multilabel=False,
            binary_threshold=0.6,
            use_highway=False,
            drop_rate=0.0,
            class_weight=False,
            use_class_weights=False,
        )
        classifier = Classifier(input_dim=2, hidden_size=2, tag_size=2, args=args)
        with torch.no_grad():
            classifier.lin1.weight.copy_(torch.eye(2))
            classifier.lin1.bias.zero_()
            classifier.lin2.weight.copy_(torch.eye(2))
            classifier.lin2.bias.zero_()
        h = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
        labels = torch.tensor([1, 0])
        loss = classifier.get_loss(h, labels, torch.tensor([2]))
        pred = classifier(h, torch.tensor([2]))
        self.assertGreater(float(loss.item()), 0.0)
        self.assertTrue(torch.equal(pred, labels))

    def test_classifier_multilabel_dialogue_targets(self):
        args = SimpleNamespace(
            class_num=3,
            loss_type="nll",
            label_smoothing=0.0,
            multilabel=True,
            multilabel_threshold=0.5,
            use_highway=False,
            drop_rate=0.0,
            class_weight=False,
            use_class_weights=False,
        )
        classifier = Classifier(input_dim=2, hidden_size=2, tag_size=3, args=args)
        h = torch.randn(3, 2)
        labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        loss = classifier.get_loss(h, labels, torch.tensor([2, 1]))
        pred = classifier(h, torch.tensor([2, 1]))
        self.assertEqual(tuple(pred.shape), (3, 3))
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_unimodal_encoder_preserves_audio_text_visual_order(self):
        class ConstantEncoder(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = float(value)

            def forward(self, x, lengths=None):
                _ = lengths
                return torch.full((x.size(0), x.size(1), 2), self.value)

        args = SimpleNamespace(rnn="ffn", drop_rate=0.0, encoder_nlayers=1)
        encoder = UnimodalEncoder(a_dim=1, t_dim=1, v_dim=1, h_dim=2, args=args)
        encoder.audio_encoder = ConstantEncoder(1.0)
        encoder.text_encoder = ConstantEncoder(2.0)
        encoder.visual_encoder = ConstantEncoder(3.0)
        audio, text, visual = encoder(
            torch.zeros(1, 2, 1),
            torch.zeros(1, 2, 1),
            torch.zeros(1, 2, 1),
            torch.tensor([2]),
        )
        self.assertAlmostEqual(float(audio.mean().item()), 1.0)
        self.assertAlmostEqual(float(text.mean().item()), 2.0)
        self.assertAlmostEqual(float(visual.mean().item()), 3.0)

    def test_unimodal_encoder_modes_return_hidden_shapes(self):
        for mode in ("ffn", "transformer", "lstm"):
            args = SimpleNamespace(rnn=mode, drop_rate=0.0, encoder_nlayers=1, encoder_nheads=2)
            encoder = UnimodalEncoder(a_dim=2, t_dim=4, v_dim=3, h_dim=6, args=args)
            audio, text, visual = encoder(
                torch.randn(2, 3, 2),
                torch.randn(2, 3, 4),
                torch.randn(2, 3, 3),
                torch.tensor([3, 2]),
            )
            self.assertEqual(tuple(audio.shape), (2, 3, 6))
            self.assertEqual(tuple(text.shape), (2, 3, 6))
            self.assertEqual(tuple(visual.shape), (2, 3, 6))

    def test_crossmodal_net_shape_and_no_input_mutation(self):
        args = SimpleNamespace(
            modalities="atv",
            crossmodal_nheads=2,
            self_att_nheads=2,
            num_crossmodal=1,
            num_self_att=1,
            drop_rate=0.0,
        )
        model = CrossmodalNet(4, args)
        features = [torch.randn(2, 3, 4) for _ in range(3)]
        original_shapes = [tuple(feature.shape) for feature in features]
        out = model(features)
        self.assertEqual(tuple(out.shape), (3, 2, 24))
        self.assertEqual([tuple(feature.shape) for feature in features], original_shapes)

    def test_dglpca_crossmodal_forward_without_graph(self):
        args = SimpleNamespace(
            dataset="mosei",
            task_type="sentiment",
            class_num=2,
            modalities="atv",
            hidden_size=4,
            graph_hidden_size=None,
            graph_weight=1.0,
            no_gnn=True,
            use_graph_transformer=False,
            graph_transformer_nheads=1,
            use_crossmodal=True,
            use_speaker=True,
            dataset_embedding_dims={"mosei": {"a": 2, "t": 2, "v": 2}},
            wp=1,
            wf=1,
            edge_type="temp_multi",
            enable_full_relations=False,
            drop_rate=0.0,
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
            crossmodal_nheads=2,
            self_att_nheads=2,
            num_crossmodal=1,
            num_self_att=1,
            device=torch.device("cpu"),
        )
        model = DGLPCA(args)
        batch = {
            "audio_tensor": torch.randn(1, 2, 2),
            "text_tensor": torch.randn(1, 2, 2),
            "visual_tensor": torch.randn(1, 2, 2),
            "text_len_tensor": torch.tensor([2], dtype=torch.long),
            "speaker_tensor": torch.zeros(1, 2, dtype=torch.long),
            "timestamps": torch.zeros(1, 2, dtype=torch.float32),
            "label_tensor": torch.tensor([0, 1], dtype=torch.long),
        }
        predictions = model(batch)
        loss = model.get_loss(batch)
        self.assertEqual(tuple(predictions.shape), (2,))
        self.assertGreater(float(loss.item()), 0.0)

    def test_graph_model_temporal_edges_and_relation_ids(self):
        args = SimpleNamespace(
            hidden_size=4,
            wp=1,
            wf=1,
            edge_type="temp_multi",
            enable_full_relations=False,
            drop_rate=0.0,
            graph_drop_rate=None,
            use_soft_mask=True,
            temporal_decay_beta=0.0,
            use_relative_time_encoding=False,
            no_time_node_embedding=False,
            edge_prune_ratio=0.0,
            han_layers=1,
            use_han_residual=True,
            use_graph_transformer=False,
            graph_transformer_nheads=1,
        )
        graph = GraphModel(4, 4, 4, num_modals=2, device=torch.device("cpu"), args=args)
        edges = graph.generate_temporal_edges(torch.tensor([3], dtype=torch.long))
        node_types = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        timestamps = torch.tensor([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]])
        edge_type = graph.generate_edges(node_types, edges, timestamps)
        self.assertEqual(tuple(edges.shape), (2, 8))
        self.assertEqual(set(edge_type.tolist()), {0, 1, 2, 3})

    def test_dglpca_full_graph_and_crossmodal_forward_with_heads(self):
        args = SimpleNamespace(
            dataset="mosei",
            task_type="sentiment",
            class_num=2,
            modalities="atv",
            hidden_size=4,
            graph_hidden_size=None,
            graph_weight=1.0,
            no_gnn=False,
            use_graph_transformer=True,
            graph_transformer_nheads=2,
            use_crossmodal=True,
            use_speaker=True,
            dataset_embedding_dims={"mosei": {"a": 2, "t": 2, "v": 2}},
            wp=1,
            wf=1,
            edge_type="temp_multi",
            enable_full_relations=False,
            drop_rate=0.0,
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
            crossmodal_nheads=2,
            self_att_nheads=2,
            num_crossmodal=1,
            num_self_att=1,
            device=torch.device("cpu"),
        )
        model = DGLPCA(args)
        batch = {
            "audio_tensor": torch.randn(1, 2, 2),
            "text_tensor": torch.randn(1, 2, 2),
            "visual_tensor": torch.randn(1, 2, 2),
            "text_len_tensor": torch.tensor([2], dtype=torch.long),
            "speaker_tensor": torch.zeros(1, 2, dtype=torch.long),
            "timestamps": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            "label_tensor": torch.tensor([0, 1], dtype=torch.long),
        }
        predictions = model(batch)
        loss = model.get_loss(batch)
        self.assertEqual(tuple(predictions.shape), (2,))
        self.assertGreater(float(loss.item()), 0.0)

    def test_config_builds_runtime_args(self):
        config = load_config("configs/mosei_sentiment_2.yaml")
        args = build_args_from_config(config, seed=42)
        self.assertEqual(args.dataset, "mosei")
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.dataset_embedding_dims["mosei"]["a"], 80)
        legacy_graph_arg = "gcn" + "_conv"
        self.assertFalse(hasattr(args, legacy_graph_arg))
        self.assertEqual(args.hidden_size, 200)
        self.assertEqual(args.han_layers, 12)
        self.assertEqual(args.graph_transformer_nheads, 4)
        self.assertEqual(args.crossmodal_nheads, 2)
        self.assertEqual((args.wp, args.wf), (5, 8))
        self.assertTrue(args.use_relative_time_encoding)
        self.assertEqual(args.loss_type, "focal")
        self.assertEqual(args.encoder_nlayers, 1)
        self.assertEqual(args.encoder_nheads, 4)

    def test_iemocap_config_supports_two_speaker_forward(self):
        config = load_config("configs/iemocap_4.yaml")
        args = build_args_from_config(config)
        args.hidden_size = 4
        args.rnn = "ffn"
        args.no_gnn = True
        args.use_crossmodal = False
        args.dataset_embedding_dims["iemocap"] = {"a": 2, "t": 2, "v": 2}
        model = DGLPCA(args)
        batch = {
            "audio_tensor": torch.randn(1, 2, 2),
            "text_tensor": torch.randn(1, 2, 2),
            "visual_tensor": torch.randn(1, 2, 2),
            "text_len_tensor": torch.tensor([2], dtype=torch.long),
            "speaker_tensor": torch.tensor([[0, 1]], dtype=torch.long),
            "timestamps": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            "label_tensor": torch.tensor([0, 1], dtype=torch.long),
        }
        self.assertEqual(model.n_speakers, 2)
        self.assertEqual(tuple(model(batch).shape), (2,))

    def test_iemocap_configs_match_manuscript_settings(self):
        for config_path, class_num, han_layers in (
            ("configs/iemocap_4.yaml", 4, 15),
            ("configs/iemocap_6.yaml", 6, 8),
        ):
            config = load_config(config_path)
            args = build_args_from_config(config)
            self.assertEqual(args.class_num, class_num)
            self.assertEqual(args.dataset_embedding_dims["iemocap"], {"a": 100, "t": 768, "v": 512})
            self.assertEqual(args.hidden_size, 200)
            self.assertEqual(args.han_layers, han_layers)
            self.assertTrue(args.use_graph_transformer)
            self.assertEqual(args.graph_transformer_nheads, 7)
            self.assertEqual((args.wp, args.wf), (6, 8))
            self.assertAlmostEqual(args.temporal_decay_beta, 0.3)
            self.assertTrue(args.use_relative_time_encoding)
            self.assertEqual(args.loss_type, "cross_entropy")

    def test_multilabel_runtime_instantiates_with_explicit_class_weights(self):
        config = load_config("configs/mosei_sentiment_2.yaml")
        config["experiment"].update({"task_type": "emotion", "class_num": 6, "multilabel": True})
        config["training"].update(
            {"class_weight": True, "class_weight_values": [1.0, 1.2, 1.3, 1.5, 1.8, 1.4]}
        )
        args = build_args_from_config(config)
        model = DGLPCA(args)
        self.assertEqual(model.clf.output_dim, 6)
        self.assertIsNotNone(model.clf.class_weights)
        self.assertEqual(model.clf.class_weights.numel(), 6)

    def test_single_label_metrics(self):
        metrics = compute_metrics(
            [torch.tensor([0, 1, 1, 0])],
            [torch.tensor([0, 1, 0, 0])],
            multilabel=False,
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["f1_binary"], 2 / 3)

    def test_multilabel_alignment_repeats_dialogue_labels(self):
        labels = align_labels_for_predictions(
            label_tensor=torch.tensor([[1, 0, 0], [0, 1, 1]]),
            predictions=torch.zeros((3, 3), dtype=torch.long),
            lengths=torch.tensor([2, 1]),
            class_num=3,
            multilabel=True,
        )
        expected = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 1]])
        self.assertTrue(torch.equal(labels, expected))

if __name__ == "__main__":
    unittest.main()
