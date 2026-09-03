from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .constants import ATTACK_TO_CLASS, CLASS_TO_INDEX, SCENARIOS


def as_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def as_vector(value: object, length: int = 3) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple, np.ndarray)):
        parts = list(value)
    else:
        parts = []
    numbers = [as_float(part) for part in parts[:length]]
    numbers.extend([0.0] * (length - len(numbers)))
    return tuple(numbers)


def infer_source(path: Path) -> tuple[str, str]:
    stem = path.stem
    for scenario in SCENARIOS:
        prefix = f"InTAS_{scenario}"
        if stem == prefix:
            return scenario, "normal"
        if stem.startswith(prefix + "_"):
            attack = stem[len(prefix) + 1 :]
            if attack in ATTACK_TO_CLASS:
                return scenario, ATTACK_TO_CLASS[attack]
    raise ValueError(f"Unsupported archive name: {path.name}")


def flatten_message(
    message: dict,
    receiver_id: str,
    scenario: str,
    source_class: str,
    split: str,
) -> dict[str, object]:
    sender = message.get("sender") or {}
    receiver = message.get("receiver") or {}
    sx, sy, sz = as_vector(sender.get("pos"))
    rx, ry, rz = as_vector(receiver.get("pos"))
    attacker = int(as_float(message.get("attacker", 0)))
    label_name = source_class if attacker == 1 and source_class != "normal" else "normal"
    return {
        "receiver_id": receiver_id,
        "sender_id": str(message.get("sender_id", "unknown")),
        "sender_alias": str(message.get("sender_alias", "unknown")),
        "message_id": str(message.get("messageID", "")),
        "send_time_ns": as_float(message.get("sendTime")),
        "receive_time_ns": as_float(message.get("rcvTime")),
        "sender_x": sx,
        "sender_y": sy,
        "sender_z": sz,
        "receiver_x": rx,
        "receiver_y": ry,
        "receiver_z": rz,
        "speed": as_float(sender.get("spd")),
        "acceleration": as_float(sender.get("acl")),
        "heading": as_float(sender.get("hed")) % 360.0,
        "rssi_dbm": as_float(
            message.get("rssi", message.get("RSSI", receiver.get("rssi"))),
            default=np.nan,
        ),
        "scenario": scenario,
        "source_class": source_class,
        "split": split.lower(),
        "attacker": attacker,
        "label": CLASS_TO_INDEX[label_name],
    }


def load_rssi_lookup(path: Path | None) -> dict[tuple[str, str], float]:
    """Load receiver-side RSSI measurements keyed by receiver_id and message_id."""
    if path is None:
        return {}
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"receiver_id", "message_id", "rssi_dbm"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"RSSI CSV is missing columns: {sorted(missing)}")
    return {
        (str(row.receiver_id), str(row.message_id)): as_float(row.rssi_dbm, np.nan)
        for row in frame.itertuples(index=False)
    }


def _split_from_inner_name(name: str) -> str:
    components = {part.lower() for part in Path(name).parts}
    for split in ("train", "validation", "test"):
        if split in components:
            return split
    raise ValueError(f"Could not determine split from nested archive path: {name}")


def iter_receiver_messages(
    outer_archive: Path,
    temporary_dir: Path,
    max_files_per_split: int = 0,
) -> Iterator[tuple[str, str, list[dict]]]:
    """Yield (split, receiver_id, messages) from VeReMi's nested ZIP layout."""
    temporary_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outer_archive) as outer:
        nested_names = sorted(name for name in outer.namelist() if name.lower().endswith(".zip"))
        if not nested_names:
            raise RuntimeError(f"No nested Train/Validation/Test ZIPs in {outer_archive}")
        for nested_name in nested_names:
            split = _split_from_inner_name(nested_name)
            with tempfile.NamedTemporaryFile(
                suffix=".zip", dir=temporary_dir, delete=False
            ) as temporary:
                with outer.open(nested_name) as source:
                    shutil.copyfileobj(source, temporary, length=1024 * 1024)
                temporary_path = Path(temporary.name)
            try:
                with zipfile.ZipFile(temporary_path) as inner:
                    json_names = sorted(
                        name for name in inner.namelist() if name.lower().endswith(".json")
                    )
                    if max_files_per_split > 0:
                        json_names = json_names[:max_files_per_split]
                    for json_name in json_names:
                        receiver_id = Path(json_name).stem
                        messages = json.loads(inner.read(json_name))
                        if not isinstance(messages, list):
                            raise TypeError(f"Expected a JSON array in {json_name}")
                        yield split, receiver_id, messages
            finally:
                temporary_path.unlink(missing_ok=True)
