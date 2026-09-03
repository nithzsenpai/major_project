from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .constants import CLASS_NAMES


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probabilities = softmax(logits, temperature)
    chosen = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.linspace(0.5, 4.0, 141)
    losses = [negative_log_likelihood(logits, labels, float(value)) for value in candidates]
    return float(candidates[int(np.argmin(losses))])


def predictions_from_probabilities(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    attack_risk = 1.0 - probabilities[:, 0]
    attack_type = np.argmax(probabilities[:, 1:], axis=1) + 1
    return np.where(attack_risk >= threshold, attack_type, 0).astype(np.int64)


def false_positive_rate(labels: np.ndarray, predictions: np.ndarray) -> float:
    normal = labels == 0
    return float(np.mean(predictions[normal] != 0)) if np.any(normal) else math.nan


def tune_attack_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    maximum_fpr: float = 0.05,
) -> tuple[float, dict[str, float]]:
    candidates = np.linspace(0.05, 0.95, 181)
    scored: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predictions = predictions_from_probabilities(probabilities, float(threshold))
        macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        fpr = false_positive_rate(labels, predictions)
        scored.append((float(threshold), float(macro_f1), float(fpr)))
    feasible = [item for item in scored if np.isnan(item[2]) or item[2] <= maximum_fpr]
    pool = feasible or scored
    best = max(pool, key=lambda item: (item[1], -item[2], -abs(item[0] - 0.5)))
    return best[0], {"validation_macro_f1": best[1], "validation_fpr": best[2]}


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict, np.ndarray, dict]:
    predictions = predictions_from_probabilities(probabilities, threshold)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        zero_division=0,
    )
    normal = labels == 0
    attack = labels != 0
    metrics: dict[str, object] = {
        "samples": int(len(labels)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(labels, predictions)),
        "false_positive_rate": false_positive_rate(labels, predictions),
        "attack_recall": float(np.mean(predictions[attack] != 0)) if np.any(attack) else math.nan,
        "normal_specificity": float(np.mean(predictions[normal] == 0)) if np.any(normal) else math.nan,
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }
    one_hot = label_binarize(labels, classes=np.arange(len(CLASS_NAMES)))
    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = math.nan
    try:
        metrics["pr_auc_macro"] = float(
            average_precision_score(one_hot, probabilities, average="macro")
        )
    except ValueError:
        metrics["pr_auc_macro"] = math.nan

    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(CLASS_NAMES)))
    report = classification_report(
        labels,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return metrics, matrix, report

