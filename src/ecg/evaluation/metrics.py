"""Multi-label metrics, computed once and shared by every model.

Micro and macro are both reported throughout because they answer different
questions on this dataset: micro is dominated by the handful of common
rhythms, macro weights the rare conditions equally. A model can improve one
while degrading the other, so neither is reported alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

AVERAGES = ("micro", "macro")


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Precision, recall and F1 at both averaging strategies."""
    scores: dict[str, float] = {}
    for average in AVERAGES:
        kwargs = {"average": average, "zero_division": 0}
        scores[f"precision_{average}"] = float(precision_score(y_true, y_pred, **kwargs))
        scores[f"recall_{average}"] = float(recall_score(y_true, y_pred, **kwargs))
        scores[f"f1_{average}"] = float(f1_score(y_true, y_pred, **kwargs))
    return scores


def per_label_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray
) -> pd.DataFrame:
    """One row per condition, sorted by support.

    Aggregate scores hide that most labels in this dataset are extremely rare;
    this is the table that shows which conditions the model never predicts.
    """
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)

    rows = []
    for idx, name in enumerate(class_names):
        truth, pred = y_true[:, idx], y_pred[:, idx]
        rows.append(
            {
                "label": str(name),
                "support": int(truth.sum()),
                "predicted": int(pred.sum()),
                "precision": float(precision_score(truth, pred, zero_division=0)),
                "recall": float(recall_score(truth, pred, zero_division=0)),
                "f1": float(f1_score(truth, pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)
