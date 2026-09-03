from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import BEHAVIOR_FEATURES, CLASS_NAMES, RELATION_FEATURES
from .features import build_receiver_frame
from .io import load_rssi_lookup
from .metrics import predictions_from_probabilities, softmax
from .model import build_model


def _csv_to_messages(path: Path) -> list[dict]:
    frame = pd.read_csv(path)

    def choose(row: pd.Series, *names: str, default: object = 0.0) -> object:
        for name in names:
            if name in row and pd.notna(row[name]):
                return row[name]
        return default

    messages: list[dict] = []
    for _, row in frame.iterrows():
        messages.append(
            {
                "rcvTime": choose(row, "rcvTime", "receive_time_ns"),
                "sendTime": choose(row, "sendTime", "send_time_ns", "timestamp"),
                "sender_id": str(choose(row, "sender_id", "vehicle_id", default="unknown")),
                "sender_alias": str(choose(row, "sender_alias", default="unknown")),
                "messageID": str(choose(row, "messageID", "message_id", default="")),
                "rssi": choose(row, "rssi_dbm", "rssi", default=np.nan),
                "receiver": {
                    "pos": [
                        choose(row, "receiver_x"),
                        choose(row, "receiver_y"),
                        choose(row, "receiver_z"),
                    ]
                },
                "sender": {
                    "pos": [
                        choose(row, "sender_x", "pos_0", "x"),
                        choose(row, "sender_y", "pos_1", "y"),
                        choose(row, "sender_z", "pos_2", "z"),
                    ],
                    "spd": choose(row, "speed", "spd", "spd_0"),
                    "acl": choose(row, "acceleration", "acl", "acl_0"),
                    "hed": choose(row, "heading", "hed", "hed_0"),
                },
            }
        )
    return messages


def load_receiver_inputs(path: Path) -> list[tuple[str, list[dict]]]:
    if path.suffix.lower() == ".json":
        messages = json.loads(path.read_text(encoding="utf-8"))
        return [(path.stem, messages)]
    if path.suffix.lower() == ".csv":
        return [(path.stem, _csv_to_messages(path))]
    if path.suffix.lower() == ".zip":
        result: list[tuple[str, list[dict]]] = []
        with zipfile.ZipFile(path) as archive:
            for name in sorted(item for item in archive.namelist() if item.endswith(".json")):
                result.append((Path(name).stem, json.loads(archive.read(name))))
        if not result:
            raise ValueError("Prediction ZIP must directly contain receiver JSON files")
        return result
    raise ValueError("Input must be a receiver JSON, flat CSV, or ZIP of receiver JSON files")


def _inference_windows(
    frame: pd.DataFrame,
    seq_len: int,
    max_gap_seconds: float,
) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    windows: list[tuple[np.ndarray, np.ndarray, dict]] = []
    relation_history_index = list(RELATION_FEATURES).index("relation_history_valid")
    for _, sender_frame in frame.groupby("sender_id", sort=False):
        sender_frame = sender_frame.sort_values("send_time_ns", kind="stable")
        times = sender_frame["send_time_ns"].to_numpy(np.float64) / 1e9
        boundaries = np.flatnonzero(np.diff(times) > max_gap_seconds) + 1
        for indices in np.split(np.arange(len(sender_frame)), boundaries):
            if len(indices) == 0:
                continue
            segment = sender_frame.iloc[indices]
            behavior = segment[list(BEHAVIOR_FEATURES)].to_numpy(np.float32)
            relation = segment[list(RELATION_FEATURES)].to_numpy(np.float32)
            history_valid_index = list(BEHAVIOR_FEATURES).index("history_valid")
            observation_index = list(BEHAVIOR_FEATURES).index("sender_observation_log")
            for end in range(0, len(segment)):
                row = segment.iloc[end]
                start = max(0, end - seq_len + 1)
                metadata = {
                    "receiver_id": str(row["receiver_id"]),
                    "sender_id": str(row["sender_id"]),
                    "sender_alias": str(row["sender_alias"]),
                    "message_id": str(row["message_id"]),
                    "send_time_ns": int(row["send_time_ns"]),
                }
                behavior_window = behavior[start : end + 1].copy()
                relation_window = relation[start : end + 1].copy()
                missing = seq_len - len(behavior_window)
                if missing > 0:
                    behavior_padding = np.repeat(behavior_window[:1], missing, axis=0)
                    behavior_padding[:, history_valid_index] = 0.0
                    behavior_padding[:, observation_index] = 0.0
                    relation_padding = np.repeat(relation_window[:1], missing, axis=0)
                    relation_padding[:, relation_history_index] = 0.0
                    behavior_window = np.concatenate([behavior_padding, behavior_window], axis=0)
                    relation_window = np.concatenate([relation_padding, relation_window], axis=0)
                windows.append((behavior_window, relation_window, metadata))
    return windows


def predict(
    checkpoint_path: Path,
    input_path: Path,
    output_path: Path,
    rssi_csv: Path | None = None,
    batch_size: int = 512,
    alert_window: int = 5,
    alert_min_messages: int = 3,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    manifest = checkpoint["manifest"]
    model = build_model(
        checkpoint["config"],
        len(manifest["behavior_features"]),
        len(manifest["relation_features"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scaler = checkpoint["scaler"]
    behavior_mean = np.asarray(scaler["behavior_mean"], np.float32)
    behavior_scale = np.asarray(scaler["behavior_scale"], np.float32)
    relation_mean = np.asarray(scaler["relation_mean"], np.float32)
    relation_scale = np.asarray(scaler["relation_scale"], np.float32)
    rssi_lookup = load_rssi_lookup(rssi_csv)

    all_windows: list[tuple[np.ndarray, np.ndarray, dict]] = []
    for receiver_id, messages in load_receiver_inputs(input_path):
        frame = build_receiver_frame(
            messages,
            receiver_id=receiver_id,
            scenario="highway_2",
            source_class="normal",
            split="inference",
            relation_bucket_seconds=float(manifest.get("relation_bucket_seconds", 1.0)),
            rssi_lookup=rssi_lookup,
        )
        if int(manifest.get("rssi_measurements", 0)) == 0 and bool(
            frame["rssi_available"].any()
        ):
            raise ValueError(
                "This checkpoint was trained without RSSI. Rebuild the training data with "
                "prepare_data.py --rssi-csv ... and retrain before using RSSI at inference."
            )
        all_windows.extend(
            _inference_windows(
                frame,
                seq_len=int(manifest["seq_len"]),
                max_gap_seconds=float(manifest.get("max_gap_seconds", 3.0)),
            )
        )
    if not all_windows:
        raise ValueError("No usable messages were found in the input")

    logits: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(all_windows), batch_size):
            batch = all_windows[start : start + batch_size]
            behavior = np.stack([item[0] for item in batch])
            relation = np.stack([item[1] for item in batch])
            behavior = (behavior - behavior_mean) / behavior_scale
            relation = (relation - relation_mean) / relation_scale
            output = model(
                torch.from_numpy(behavior).to(device),
                torch.from_numpy(relation).to(device),
            )
            logits.append(output["logits"].cpu().numpy())
    probabilities = softmax(
        np.concatenate(logits), float(checkpoint.get("temperature", 1.0))
    )
    threshold = float(checkpoint.get("attack_threshold", 0.5))
    predictions = predictions_from_probabilities(probabilities, threshold)

    rows: list[dict] = []
    for (_, _, metadata), probability, prediction in zip(
        all_windows, probabilities, predictions, strict=True
    ):
        risk = float(1.0 - probability[0])
        predicted_class = CLASS_NAMES[int(prediction)]
        action = {
            "normal": "accept_message",
            "sybil": "quarantine_identity_group_and_reduce_trust",
            "illusion": "reject_position_claim_and_request_reverification",
        }[predicted_class]
        rows.append(
            metadata
            | {
                "probability_normal": float(probability[0]),
                "probability_sybil": float(probability[1]),
                "probability_illusion": float(probability[2]),
                "malicious_risk": risk,
                "prediction": predicted_class,
                "recommended_action": action,
            }
        )
    result = pd.DataFrame(rows).sort_values(["receiver_id", "sender_id", "send_time_ns"])
    result["smoothed_risk"] = (
        result.groupby(["receiver_id", "sender_id"])["malicious_risk"]
        .transform(lambda values: values.rolling(alert_window, min_periods=1).mean())
    )
    result["recent_suspicious_count"] = (
        result.assign(_suspicious=result["malicious_risk"] >= threshold)
        .groupby(["receiver_id", "sender_id"])["_suspicious"]
        .transform(lambda values: values.rolling(alert_window, min_periods=1).sum())
        .astype(int)
    )
    result["vehicle_alert"] = result["recent_suspicious_count"] >= alert_min_messages
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Normal/Sybil/Illusion from messages")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--rssi-csv", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--alert-window", type=int, default=5)
    parser.add_argument("--alert-min-messages", type=int, default=3)
    args = parser.parse_args()
    result = predict(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        output_path=args.output,
        rssi_csv=args.rssi_csv,
        batch_size=args.batch_size,
        alert_window=args.alert_window,
        alert_min_messages=args.alert_min_messages,
    )
    print(result.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
