"""Naive baselines, to make the model scores interpretable.

A micro F1 of 0.71 means nothing on its own. On a long-tailed multi-label
problem, always predicting the few most frequent labels can score
surprisingly well, and any model that fails to beat that has learned nothing
useful. These strategies use no features at all — only label frequencies
counted on the training split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecg.evaluation.metrics import multilabel_metrics


def predict_constant(y_train: np.ndarray, n_test: int, k: int) -> np.ndarray:
    """Always predict the ``k`` most frequent training labels, for every record."""
    if k < 0:
        raise ValueError("k must be non-negative")
    n_labels = y_train.shape[1]
    prediction = np.zeros((n_test, n_labels), dtype=int)
    if k == 0:
        return prediction
    top = np.argsort(y_train.sum(axis=0))[::-1][:k]
    prediction[:, top] = 1
    return prediction


def predict_all(n_test: int, n_labels: int) -> np.ndarray:
    """Predict every label for every record — the maximum-recall extreme."""
    return np.ones((n_test, n_labels), dtype=int)


def evaluate_baselines(
    y_train: np.ndarray, y_test: np.ndarray, k_values: tuple[int, ...] = (1, 2, 3, 5, 10)
) -> pd.DataFrame:
    """Score each naive strategy on the test split.

    Returns one row per strategy with the same metrics the models report, so
    the numbers drop straight into the comparison table.
    """
    n_test, n_labels = y_test.shape
    rows = []

    rows.append({"strategy": "predict nothing", "labels predicted": 0,
                 **multilabel_metrics(y_test, predict_constant(y_train, n_test, 0))})

    for k in k_values:
        if k > n_labels:
            continue
        rows.append({"strategy": f"top-{k} most frequent", "labels predicted": k,
                     **multilabel_metrics(y_test, predict_constant(y_train, n_test, k))})

    rows.append({"strategy": "predict everything", "labels predicted": n_labels,
                 **multilabel_metrics(y_test, predict_all(n_test, n_labels))})

    return pd.DataFrame(rows)


def label_statistics(y: np.ndarray) -> dict[str, float]:
    """Density and cardinality of a multi-label target — the shape summary."""
    n_records, n_labels = y.shape
    per_record = y.sum(axis=1)
    per_label = y.sum(axis=0)
    return {
        "records": int(n_records),
        "labels": int(n_labels),
        "density": float(y.mean()),
        "mean_labels_per_record": float(per_record.mean()),
        "labels_seen_once_or_less": int((per_label <= 1).sum()),
        "labels_under_10": int((per_label < 10).sum()),
        "top_label_prevalence": float(per_label.max() / n_records),
    }
