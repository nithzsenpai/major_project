from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import NpyShardDataset
from .metrics import calculate_metrics, softmax
from .model import build_model
from .reporting import save_evaluation


def evaluate_checkpoint(
    checkpoint_path: Path,
    processed_dir: Path,
    split: str,
    output_dir: Path,
    batch_size: int = 512,
) -> dict:
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
    dataset = NpyShardDataset(processed_dir, split, checkpoint["scaler"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for behavior, relation, target in loader:
            output = model(behavior.to(device), relation.to(device))
            logits.append(output["logits"].cpu().numpy())
            labels.append(target.numpy())
    all_logits = np.concatenate(logits)
    all_labels = np.concatenate(labels)
    probabilities = softmax(all_logits, float(checkpoint.get("temperature", 1.0)))
    metrics, matrix, report = calculate_metrics(
        all_labels, probabilities, float(checkpoint.get("attack_threshold", 0.5))
    )
    save_evaluation(output_dir, split, metrics, matrix, report)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained VANET checkpoint")
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    metrics = evaluate_checkpoint(**vars(args))
    print(metrics)


if __name__ == "__main__":
    main()
