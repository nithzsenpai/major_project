from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .constants import CLASS_NAMES
from .utils import write_json


def save_evaluation(
    output_dir: Path,
    split: str,
    metrics: dict,
    matrix: np.ndarray,
    report: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{split}_metrics.json", metrics)
    write_json(output_dir / f"{split}_classification_report.json", report)
    np.savetxt(
        output_dir / f"{split}_confusion_matrix.csv",
        matrix,
        fmt="%d",
        delimiter=",",
        header=",".join(CLASS_NAMES),
        comments="",
    )

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(f"{split.title()} confusion matrix")
    maximum = matrix.max() if matrix.size else 1
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > maximum / 2 else "black"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
    figure.tight_layout()
    figure.savefig(output_dir / f"{split}_confusion_matrix.png", dpi=180)
    plt.close(figure)


def save_history(output_dir: Path, history: list[dict]) -> None:
    if not history:
        return
    columns = list(history[0].keys())
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)

    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["validation_loss"] for item in history], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [item["validation_macro_f1"] for item in history])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Validation macro F1")
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=180)
    plt.close(figure)

