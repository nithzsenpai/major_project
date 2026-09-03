from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _shard_paths(split_dir: Path) -> list[tuple[Path, Path, Path]]:
    behavior_paths = sorted(split_dir.glob("shard_*_behavior.npy"))
    result: list[tuple[Path, Path, Path]] = []
    for behavior in behavior_paths:
        stem = behavior.name.removesuffix("_behavior.npy")
        relation = split_dir / f"{stem}_relation.npy"
        labels = split_dir / f"{stem}_labels.npy"
        if not relation.exists() or not labels.exists():
            raise FileNotFoundError(f"Incomplete shard: {stem}")
        result.append((behavior, relation, labels))
    if not result:
        raise FileNotFoundError(f"No sequence shards in {split_dir}")
    return result


class NpyShardDataset(Dataset):
    def __init__(
        self,
        processed_dir: Path,
        split: str,
        scaler: dict[str, list[float]] | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.split = split
        paths = _shard_paths(self.processed_dir / split)
        self.behavior = [np.load(item[0], mmap_mode="r") for item in paths]
        self.relation = [np.load(item[1], mmap_mode="r") for item in paths]
        self.labels = [np.load(item[2], mmap_mode="r") for item in paths]
        self.lengths = [len(array) for array in self.labels]
        self.cumulative = np.cumsum(self.lengths).tolist()
        self.scaler = scaler

        for behavior, relation, labels in zip(
            self.behavior, self.relation, self.labels, strict=True
        ):
            if not (len(behavior) == len(relation) == len(labels)):
                raise ValueError("Behavior, relation and label shard lengths differ")

        if scaler is not None:
            self.behavior_mean = np.asarray(scaler["behavior_mean"], dtype=np.float32)
            self.behavior_scale = np.asarray(scaler["behavior_scale"], dtype=np.float32)
            self.relation_mean = np.asarray(scaler["relation_mean"], dtype=np.float32)
            self.relation_scale = np.asarray(scaler["relation_scale"], dtype=np.float32)

    def __len__(self) -> int:
        return self.cumulative[-1]

    def _location(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.cumulative, index)
        start = 0 if shard == 0 else self.cumulative[shard - 1]
        return shard, index - start

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shard, local = self._location(index)
        behavior = np.array(self.behavior[shard][local], dtype=np.float32, copy=True)
        relation = np.array(self.relation[shard][local], dtype=np.float32, copy=True)
        if self.scaler is not None:
            behavior = (behavior - self.behavior_mean) / self.behavior_scale
            relation = (relation - self.relation_mean) / self.relation_scale
        label = int(self.labels[shard][local])
        return torch.from_numpy(behavior), torch.from_numpy(relation), torch.tensor(label)

    def all_labels(self) -> np.ndarray:
        return np.concatenate([np.asarray(labels) for labels in self.labels]).astype(np.int64)


def _running_moments(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, int]:
    total_sum: np.ndarray | None = None
    total_square: np.ndarray | None = None
    count = 0
    for path in paths:
        array = np.load(path, mmap_mode="r")
        flattened = np.asarray(array, dtype=np.float64).reshape(-1, array.shape[-1])
        batch_sum = flattened.sum(axis=0)
        batch_square = np.square(flattened).sum(axis=0)
        total_sum = batch_sum if total_sum is None else total_sum + batch_sum
        total_square = batch_square if total_square is None else total_square + batch_square
        count += len(flattened)
    if total_sum is None or total_square is None or count == 0:
        raise ValueError("Cannot calculate scaler for an empty dataset")
    mean = total_sum / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-8)
    return mean, np.sqrt(variance), count


def compute_training_scaler(processed_dir: Path) -> dict[str, list[float]]:
    train_dir = Path(processed_dir) / "train"
    behavior_paths = sorted(train_dir.glob("shard_*_behavior.npy"))
    relation_paths = sorted(train_dir.glob("shard_*_relation.npy"))
    behavior_mean, behavior_scale, behavior_count = _running_moments(behavior_paths)
    relation_mean, relation_scale, relation_count = _running_moments(relation_paths)
    return {
        "behavior_mean": behavior_mean.tolist(),
        "behavior_scale": behavior_scale.tolist(),
        "relation_mean": relation_mean.tolist(),
        "relation_scale": relation_scale.tolist(),
        "behavior_values": behavior_count,
        "relation_values": relation_count,
    }


def save_scaler(scaler: dict, path: Path) -> None:
    path.write_text(json.dumps(scaler, indent=2), encoding="utf-8")


def load_manifest(processed_dir: Path) -> dict:
    return json.loads((Path(processed_dir) / "manifest.json").read_text(encoding="utf-8"))

