from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .constants import BEHAVIOR_FEATURES, CLASS_NAMES, RELATION_FEATURES
from .features import build_receiver_frame, iter_sequences
from .io import infer_source, iter_receiver_messages, load_rssi_lookup


class ShardWriter:
    def __init__(self, output_dir: Path, split: str, shard_size: int) -> None:
        self.directory = output_dir / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.behavior: list[np.ndarray] = []
        self.relation: list[np.ndarray] = []
        self.labels: list[int] = []
        self.shard_index = 0
        self.counts: Counter[int] = Counter()

    def add(self, behavior: np.ndarray, relation: np.ndarray, label: int) -> None:
        self.behavior.append(behavior)
        self.relation.append(relation)
        self.labels.append(label)
        self.counts[label] += 1
        if len(self.labels) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.labels:
            return
        prefix = self.directory / f"shard_{self.shard_index:05d}"
        np.save(prefix.with_name(prefix.name + "_behavior.npy"), np.stack(self.behavior))
        np.save(prefix.with_name(prefix.name + "_relation.npy"), np.stack(self.relation))
        np.save(prefix.with_name(prefix.name + "_labels.npy"), np.asarray(self.labels, np.int64))
        self.behavior.clear()
        self.relation.clear()
        self.labels.clear()
        self.shard_index += 1


def prepare_dataset(
    raw_dir: Path,
    output_dir: Path,
    seq_len: int = 16,
    stride: int = 2,
    max_gap_seconds: float = 3.0,
    relation_bucket_seconds: float = 1.0,
    shard_size: int = 10_000,
    max_files_per_split: int = 0,
    keep_attack_source_normals: bool = False,
    overwrite: bool = False,
    rssi_csv: Path | None = None,
) -> dict:
    archives: list[tuple[Path, str, str]] = []
    for path in sorted(raw_dir.glob("*.zip")):
        try:
            scenario, source_class = infer_source(path)
        except ValueError:
            continue
        archives.append((path, scenario, source_class))
    if not archives:
        raise FileNotFoundError(f"No supported VeReMi archives found in {raw_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} is not empty. Use --overwrite or choose another directory."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir / "_temporary_nested_zips"

    writers = {
        split: ShardWriter(output_dir, split, shard_size)
        for split in ("train", "validation", "test")
    }
    processed_receivers = Counter()
    rssi_measurements = 0
    rssi_lookup = load_rssi_lookup(rssi_csv)

    try:
        for archive, scenario, source_class in archives:
            description = f"{scenario}/{source_class}"
            iterator = iter_receiver_messages(
                archive, temporary_dir, max_files_per_split=max_files_per_split
            )
            for split, receiver_id, messages in tqdm(iterator, desc=description):
                processed_receivers[(split, scenario, source_class)] += 1
                frame = build_receiver_frame(
                    messages,
                    receiver_id=receiver_id,
                    scenario=scenario,
                    source_class=source_class,
                    split=split,
                    relation_bucket_seconds=relation_bucket_seconds,
                    rssi_lookup=rssi_lookup,
                )
                rssi_measurements += int(frame["rssi_available"].sum())
                for behavior, relation, label in iter_sequences(
                    frame,
                    seq_len=seq_len,
                    stride=stride,
                    max_gap_seconds=max_gap_seconds,
                    keep_attack_source_normals=keep_attack_source_normals,
                ):
                    writers[split].add(behavior, relation, label)
    finally:
        for writer in writers.values():
            writer.flush()
        shutil.rmtree(temporary_dir, ignore_errors=True)

    manifest = {
        "version": 1,
        "seq_len": seq_len,
        "stride": stride,
        "max_gap_seconds": max_gap_seconds,
        "relation_bucket_seconds": relation_bucket_seconds,
        "behavior_features": list(BEHAVIOR_FEATURES),
        "relation_features": list(RELATION_FEATURES),
        "classes": list(CLASS_NAMES),
        "normal_target_policy": (
            "baseline_once_plus_attack_subset_normals"
            if keep_attack_source_normals
            else "baseline_only"
        ),
        "rssi_source": str(rssi_csv) if rssi_csv else None,
        "rssi_measurements": rssi_measurements,
        "archives": [path.name for path, _, _ in archives],
        "splits": {
            split: {
                "sequences": int(sum(writer.counts.values())),
                "class_counts": {
                    CLASS_NAMES[index]: int(writer.counts.get(index, 0))
                    for index in range(len(CLASS_NAMES))
                },
                "shards": writer.shard_index,
            }
            for split, writer in writers.items()
        },
        "processed_receiver_files": {
            "|".join(key): value for key, value in sorted(processed_receivers.items())
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["splits"], indent=2))
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build leakage-safe VeReMi sequence shards")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-gap-seconds", type=float, default=3.0)
    parser.add_argument("--relation-bucket-seconds", type=float, default=1.0)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--max-files-per-split", type=int, default=0)
    parser.add_argument("--keep-attack-source-normals", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rssi-csv",
        type=Path,
        help="Optional receiver_id,message_id,rssi_dbm measurements from a simulator or hardware",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    prepare_dataset(**vars(args))


if __name__ == "__main__":
    main()
