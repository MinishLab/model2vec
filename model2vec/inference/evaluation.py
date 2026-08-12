from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np


def _is_multi_label_shaped(y: list[int] | list[str] | list[list[int]] | list[list[str]]) -> bool:
    """Check if the labels are in a multi-label shape."""
    return isinstance(y, (list, tuple)) and len(y) > 0 and isinstance(y[0], (list, tuple, set))


def _one_hot(labels: Sequence[Any], classes: Sequence[Any]) -> np.ndarray:
    """One-hot encode a flat sequence of labels against a fixed set of classes."""
    index = {label: position for position, label in enumerate(classes)}
    encoded = np.zeros((len(labels), len(classes)), dtype=int)
    for row, label in enumerate(labels):
        encoded[row, index[label]] = 1
    return encoded


def _multi_hot(label_lists: Iterable[Iterable[Any]], classes: Sequence[Any]) -> np.ndarray:
    """Multi-hot encode a sequence of label lists against a fixed set of classes."""
    index = {label: position for position, label in enumerate(classes)}
    label_lists = list(label_lists)
    encoded = np.zeros((len(label_lists), len(classes)), dtype=int)
    for row, labels in enumerate(label_lists):
        for label in labels:
            encoded[row, index[label]] = 1
    return encoded


def _precision_recall_f1_support(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class precision, recall, f1, and support from one-hot / multi-hot encoded labels."""
    true_positive = ((y_true == 1) & (y_pred == 1)).sum(axis=0).astype(float)
    false_positive = ((y_true == 0) & (y_pred == 1)).sum(axis=0).astype(float)
    false_negative = ((y_true == 1) & (y_pred == 0)).sum(axis=0).astype(float)
    support = y_true.sum(axis=0)

    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    precision = np.divide(
        true_positive, predicted_positive, out=np.zeros_like(true_positive), where=predicted_positive > 0
    )
    recall = np.divide(true_positive, actual_positive, out=np.zeros_like(true_positive), where=actual_positive > 0)
    precision_plus_recall = precision + recall
    f1 = np.divide(
        2 * precision * recall, precision_plus_recall, out=np.zeros_like(precision), where=precision_plus_recall > 0
    )

    return precision, recall, f1, support


def evaluate_single_or_multi_label(
    predictions: np.ndarray,
    y: list[int] | list[str] | list[list[int]] | list[list[str]],
) -> dict[str, dict[str, float]]:
    """Evaluate the classifier on a given dataset using a classification report.

    This function computes per-class precision, recall and f1-score (via one-vs-rest / multi-hot encoding), plus
    overall accuracy, macro average, and weighted average.

    :param predictions: The predictions.
    :param y: The ground truth labels.
    :return: A classification report, as a dictionary.
    """
    if _is_multi_label_shaped(y):
        y = cast(list[list[str]] | list[list[int]], y)
        predictions = cast(np.ndarray, predictions)
        y_labels = {label for labels in y for label in labels}
        predicted_labels = {label for labels in predictions for label in labels}
        classes = sorted(y_labels | predicted_labels)
        y_transformed = _multi_hot(y, classes)
        predictions_transformed = _multi_hot(predictions, classes)
    else:
        y = cast(list[str] | list[int], y)
        classes = sorted(set(y) | set(predictions.tolist()))
        y_transformed = _one_hot(y, classes)
        predictions_transformed = _one_hot(predictions.tolist(), classes)

    target_names = [str(c) for c in classes]
    precision, recall, f1, support = _precision_recall_f1_support(y_transformed, predictions_transformed)
    total_support = float(support.sum())
    accuracy = float(np.all(y_transformed == predictions_transformed, axis=1).mean())

    report: dict[str, Any] = {
        name: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1-score": float(f1[idx]),
            "support": float(support[idx]),
        }
        for idx, name in enumerate(target_names)
    }
    report["accuracy"] = accuracy
    report["macro avg"] = {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1-score": float(f1.mean()),
        "support": total_support,
    }
    weights = support / total_support if total_support > 0 else np.zeros_like(support, dtype=float)
    report["weighted avg"] = {
        "precision": float((precision * weights).sum()),
        "recall": float((recall * weights).sum()),
        "f1-score": float((f1 * weights).sum()),
        "support": total_support,
    }

    return report
