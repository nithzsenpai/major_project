from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from vanet_detector.constants import BEHAVIOR_FEATURES, RELATION_FEATURES
from vanet_detector.features import build_receiver_frame, iter_sequences
from vanet_detector.io import flatten_message, infer_source, iter_receiver_messages
from vanet_detector.model import VANETDetector


def message(
    sender_id: str = "veh_1",
    alias: str = "alias_1",
    message_id: str = "1",
    send_time: float = 1_000_000_000,
    x: float = 10.0,
    rssi: float | None = None,
    attacker: int = 0,
) -> dict:
    row = {
        "sender_id": sender_id,
        "sender_alias": alias,
        "messageID": message_id,
        "sendTime": str(send_time),
        "rcvTime": str(send_time + 1_000_000),
        "sender": {"pos": [x, 0.0, 0.0], "spd": "10", "acl": "0", "hed": "0"},
        "receiver": {"pos": [0.0, 0.0, 0.0]},
        "attacker": attacker,
    }
    if rssi is not None:
        row["rssi"] = rssi
    return row


class PipelineTests(unittest.TestCase):
    def test_parser_accepts_numeric_strings_and_labels_attacks(self) -> None:
        normal = flatten_message(message(), "rx", "highway_2", "normal", "train")
        attack = flatten_message(
            message(attacker=1), "rx", "highway_2", "sybil", "train"
        )
        self.assertEqual(normal["speed"], 10.0)
        self.assertEqual(normal["label"], 0)
        self.assertEqual(attack["label"], 1)

    def test_cold_start_sequence_is_masked_not_discarded(self) -> None:
        frame = build_receiver_frame(
            [message(attacker=1)], "rx", "highway_2", "sybil", "train"
        )
        behavior, relation, label = next(iter_sequences(frame, 4, 1, 3.0))
        self.assertEqual(behavior.shape, (4, len(BEHAVIOR_FEATURES)))
        self.assertEqual(relation.shape, (4, len(RELATION_FEATURES)))
        np.testing.assert_array_equal(
            behavior[:, BEHAVIOR_FEATURES.index("history_valid")], [0, 0, 0, 1]
        )
        np.testing.assert_array_equal(
            relation[:, RELATION_FEATURES.index("relation_history_valid")],
            [0, 0, 0, 1],
        )
        self.assertEqual(label, 1)

    def test_rssi_similarity_requires_two_real_measurements(self) -> None:
        absent = build_receiver_frame(
            [
                message("a", "aa", "1", x=10.0),
                message("b", "bb", "2", x=10.1),
            ],
            "rx",
            "highway_2",
            "normal",
            "train",
        )
        self.assertTrue((absent["max_nearby_rssi_similarity"] == 0).all())

        present = build_receiver_frame(
            [
                message("a", "aa", "1", x=10.0, rssi=-61.0),
                message("b", "bb", "2", x=10.1, rssi=-61.2),
            ],
            "rx",
            "highway_2",
            "normal",
            "train",
        )
        self.assertTrue((present["rssi_available"] == 1).all())
        self.assertTrue((present["max_nearby_rssi_similarity"] > 0.9).all())

    def test_nested_archive_reader_preserves_official_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inner_path = root / "Train.zip"
            with zipfile.ZipFile(inner_path, "w") as inner:
                inner.writestr("Train/veh_9.json", json.dumps([message()]))
            outer_path = root / "InTAS_highway_2.zip"
            with zipfile.ZipFile(outer_path, "w") as outer:
                outer.write(inner_path, "Train/Train.zip")
            rows = list(iter_receiver_messages(outer_path, root / "temporary"))
            self.assertEqual(rows[0][0], "train")
            self.assertEqual(rows[0][1], "veh_9")

    def test_model_forward_shapes_and_attention(self) -> None:
        model = VANETDetector(
            behavior_features=len(BEHAVIOR_FEATURES),
            relation_features=len(RELATION_FEATURES),
            tcn_channels=[16, 16],
            gru_hidden=8,
            relation_hidden=8,
            attention_heads=4,
            fusion_hidden=24,
            dropout=0.0,
        )
        output = model(
            torch.randn(3, 8, len(BEHAVIOR_FEATURES)),
            torch.randn(3, 8, len(RELATION_FEATURES)),
        )
        self.assertEqual(tuple(output["logits"].shape), (3, 3))
        torch.testing.assert_close(
            output["behavior_attention"].sum(dim=1), torch.ones(3)
        )
        torch.testing.assert_close(
            output["relation_attention"].sum(dim=1), torch.ones(3)
        )

    def test_supported_archive_names(self) -> None:
        self.assertEqual(infer_source(Path("InTAS_urban_2.zip")), ("urban_2", "normal"))
        self.assertEqual(
            infer_source(Path("InTAS_urban_2_trafficCongestionSybil.zip")),
            ("urban_2", "sybil"),
        )


if __name__ == "__main__":
    unittest.main()
