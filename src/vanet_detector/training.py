from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import NpyShardDataset, compute_training_scaler, load_manifest, save_scaler
from .metrics import (
    calculate_metrics,
    fit_temperature,
    softmax,
    tune_attack_threshold,
)
from .model import build_model
from .reporting import save_evaluation, save_history
from .utils import load_config, parameter_count, set_seed, write_json


class FocalCrossEntropy(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None,
        gamma: float,
        label_smoothing: float,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probability = torch.softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
        focal = torch.pow(1.0 - probability, self.gamma)
        return torch.mean(focal * cross_entropy)


def _class_statistics(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training split is missing classes {missing}; download all required subsets")
    weights = counts.sum() / (len(counts) * counts)
    weights = np.clip(weights, 0.25, 8.0)
    weights = weights / weights.mean()
    return counts, weights


def _build_loader(
    dataset: NpyShardDataset,
    batch_size: int,
    workers: int,
    training: bool,
    balanced_sampling: bool,
    pin_memory: bool,
) -> DataLoader:
    sampler = None
    shuffle = training
    if training and balanced_sampling:
        labels = dataset.all_labels()
        counts = np.bincount(labels, minlength=3).astype(np.float64)
        sample_weights = 1.0 / np.maximum(counts[labels], 1.0)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(labels),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        drop_last=training,
    )


def _combined_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    main_loss: nn.Module,
    auxiliary_weight: float,
    sybil_pos_weight: torch.Tensor,
    illusion_pos_weight: torch.Tensor,
) -> torch.Tensor:
    total = main_loss(outputs["logits"], labels)
    if auxiliary_weight <= 0:
        return total
    sybil_targets = (labels == 1).float()
    illusion_targets = (labels == 2).float()
    sybil_loss = F.binary_cross_entropy_with_logits(
        outputs["sybil_logit"], sybil_targets, pos_weight=sybil_pos_weight
    )
    illusion_loss = F.binary_cross_entropy_with_logits(
        outputs["illusion_logit"], illusion_targets, pos_weight=illusion_pos_weight
    )
    return total + auxiliary_weight * (sybil_loss + illusion_loss)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    main_loss: nn.Module,
    auxiliary_weight: float,
    sybil_pos_weight: torch.Tensor,
    illusion_pos_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    maximum_gradient_norm: float = 1.0,
    amp_enabled: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for behavior, relation, labels in loader:
        behavior = behavior.to(device, non_blocking=True)
        relation = relation.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(behavior, relation)
                loss = _combined_loss(
                    outputs,
                    labels,
                    main_loss,
                    auxiliary_weight,
                    sybil_pos_weight,
                    illusion_pos_weight,
                )
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), maximum_gradient_norm)
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        all_logits.append(outputs["logits"].detach().float().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    return float(np.mean(losses)), np.concatenate(all_logits), np.concatenate(all_labels)


def train(config: dict, run_dir: Path, overwrite: bool = False) -> Path:
    seed = int(config.get("seed", 42))
    set_seed(seed, deterministic=bool(config.get("deterministic", True)))
    data_dir = Path(config["data_dir"])
    manifest = load_manifest(data_dir)

    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{run_dir} is not empty; pass --overwrite to replace it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "resolved_config.json", config)

    scaler_values = compute_training_scaler(data_dir)
    save_scaler(scaler_values, run_dir / "scaler.json")
    train_dataset = NpyShardDataset(data_dir, "train", scaler_values)
    validation_dataset = NpyShardDataset(data_dir, "validation", scaler_values)
    test_dataset = NpyShardDataset(data_dir, "test", scaler_values)

    training_config = config.get("training", {})
    batch_size = int(training_config.get("batch_size", 256))
    workers = int(training_config.get("num_workers", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = _build_loader(
        train_dataset,
        batch_size,
        workers,
        True,
        bool(training_config.get("balanced_sampling", False)),
        pin_memory,
    )
    validation_loader = _build_loader(
        validation_dataset, batch_size, workers, False, False, pin_memory
    )
    test_loader = _build_loader(test_dataset, batch_size, workers, False, False, pin_memory)

    labels = train_dataset.all_labels()
    counts, class_weights_array = _class_statistics(labels)
    use_class_weights = bool(training_config.get("class_weighting", True))
    class_weights = (
        torch.tensor(class_weights_array, dtype=torch.float32, device=device)
        if use_class_weights
        else None
    )
    model = build_model(
        config,
        behavior_features=len(manifest["behavior_features"]),
        relation_features=len(manifest["relation_features"]),
    ).to(device)
    total_parameters, trainable_parameters = parameter_count(model)
    (run_dir / "model_summary.txt").write_text(
        f"{model}\n\nTotal parameters: {total_parameters:,}\n"
        f"Trainable parameters: {trainable_parameters:,}\n",
        encoding="utf-8",
    )

    main_loss = FocalCrossEntropy(
        class_weights=class_weights,
        gamma=float(training_config.get("focal_gamma", 1.5)),
        label_smoothing=float(training_config.get("label_smoothing", 0.02)),
    )
    sybil_positive = torch.tensor(
        min(float((counts.sum() - counts[1]) / counts[1]), 10.0), device=device
    )
    illusion_positive = torch.tensor(
        min(float((counts.sum() - counts[2]) / counts[2]), 10.0), device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training_config.get("lr_factor", 0.5)),
        patience=int(training_config.get("lr_patience", 3)),
        min_lr=float(training_config.get("minimum_lr", 1e-6)),
    )
    amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    gradient_scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    epochs = int(training_config.get("epochs", 50))
    patience = int(training_config.get("early_stopping_patience", 10))
    auxiliary_weight = float(training_config.get("auxiliary_weight", 0.2))
    maximum_gradient_norm = float(training_config.get("max_gradient_norm", 1.0))
    best_score = -math.inf
    epochs_without_improvement = 0
    history: list[dict] = []
    checkpoint_path = run_dir / "best_model.pt"
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, _, _ = _run_epoch(
            model,
            train_loader,
            device,
            main_loss,
            auxiliary_weight,
            sybil_positive,
            illusion_positive,
            optimizer=optimizer,
            scaler=gradient_scaler,
            maximum_gradient_norm=maximum_gradient_norm,
            amp_enabled=amp_enabled,
        )
        with torch.no_grad():
            validation_loss, validation_logits, validation_labels = _run_epoch(
                model,
                validation_loader,
                device,
                main_loss,
                auxiliary_weight,
                sybil_positive,
                illusion_positive,
                amp_enabled=amp_enabled,
            )
        validation_predictions = np.argmax(validation_logits, axis=1)
        validation_macro_f1 = float(
            f1_score(validation_labels, validation_predictions, average="macro", zero_division=0)
        )
        scheduler.step(validation_macro_f1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_macro_f1": validation_macro_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train {train_loss:.5f} | val {validation_loss:.5f} "
            f"| macro-F1 {validation_macro_f1:.4f}"
        )

        if validation_macro_f1 > best_score + 1e-5:
            best_score = validation_macro_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "manifest": manifest,
                    "scaler": scaler_values,
                    "epoch": epoch,
                    "validation_macro_f1": validation_macro_f1,
                    "class_counts": counts.tolist(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}")
                break

    save_history(run_dir, history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        _, validation_logits, validation_labels = _run_epoch(
            model,
            validation_loader,
            device,
            main_loss,
            auxiliary_weight,
            sybil_positive,
            illusion_positive,
            amp_enabled=amp_enabled,
        )
        _, test_logits, test_labels = _run_epoch(
            model,
            test_loader,
            device,
            main_loss,
            auxiliary_weight,
            sybil_positive,
            illusion_positive,
            amp_enabled=amp_enabled,
        )

    temperature = fit_temperature(validation_logits, validation_labels)
    validation_probabilities = softmax(validation_logits, temperature)
    threshold_config = config.get("threshold", {})
    threshold, threshold_details = tune_attack_threshold(
        validation_probabilities,
        validation_labels,
        maximum_fpr=float(threshold_config.get("maximum_false_positive_rate", 0.05)),
    )
    validation_metrics, validation_matrix, validation_report = calculate_metrics(
        validation_labels, validation_probabilities, threshold
    )
    test_probabilities = softmax(test_logits, temperature)
    test_metrics, test_matrix, test_report = calculate_metrics(
        test_labels, test_probabilities, threshold
    )
    save_evaluation(
        run_dir, "validation", validation_metrics, validation_matrix, validation_report
    )
    save_evaluation(run_dir, "test", test_metrics, test_matrix, test_report)

    checkpoint.update(
        {
            "temperature": temperature,
            "attack_threshold": threshold,
            "threshold_details": threshold_details,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "training_seconds": time.perf_counter() - started,
        }
    )
    torch.save(checkpoint, checkpoint_path)
    write_json(run_dir / "final_summary.json", checkpoint | {"model_state": "stored in checkpoint"})
    print(f"Saved calibrated checkpoint: {checkpoint_path}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    return checkpoint_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train optimized VANET detector")
    parser.add_argument("--config", type=Path, default=Path("configs/recommended.yaml"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    run_dir = args.run_dir or Path(config.get("run_dir", "runs/recommended"))
    train(config, run_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

