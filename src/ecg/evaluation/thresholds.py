"""Probability cutoff selection, shared by all probabilistic models.

Default 0.5 thresholding is wrong for this problem. Labels are heavily
imbalanced, and in a screening context a missed arrhythmia costs more than a
false alarm, so the operating point is chosen explicitly against cross-
validated predictions rather than left at the library default.

Crucially the sweep runs on *training-fold* predictions only. Choosing a
cutoff on the test set would leak it, and the resulting test score would be
optimistically biased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from ecg.evaluation.metrics import multilabel_metrics

#: 0.05 to 0.95 in 0.05 steps.
CUTOFF_GRID = np.round(np.linspace(0.05, 0.95, 19), 2)

DEFAULT_CV_FOLDS = 5


def sweep_cutoffs(
    y_true: np.ndarray, y_proba: np.ndarray, cutoffs: np.ndarray = CUTOFF_GRID
) -> pd.DataFrame:
    """Score every cutoff on one set of predicted probabilities."""
    y_true = np.atleast_2d(np.asarray(y_true))
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        y_proba = y_proba[:, None]

    rows = []
    for cutoff in cutoffs:
        y_pred = (y_proba >= cutoff).astype(int)
        rows.append({"cutoff": float(cutoff), **multilabel_metrics(y_true, y_pred)})
    return pd.DataFrame(rows)


def sweep_from_cv(
    model,
    X,
    y,
    cv: int = DEFAULT_CV_FOLDS,
    cutoffs: np.ndarray = CUTOFF_GRID,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Cross-validate ``model``, then sweep cutoffs on the held-out folds.

    Returns the sweep table and the out-of-fold probability matrix, so callers
    can reuse the probabilities without paying for cross-validation twice.
    """
    X_arr = (
        X.to_numpy(dtype=np.float32)
        if hasattr(X, "to_numpy")
        else np.asarray(X, dtype=np.float32)
    )
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    y_proba = np.asarray(
        cross_val_predict(model, X_arr, y_arr, cv=cv, method="predict_proba", n_jobs=n_jobs)
    )
    if y_proba.ndim == 1:
        y_proba = y_proba[:, None]

    return sweep_cutoffs(y_arr, y_proba, cutoffs), y_proba


def best_cutoff(sweep: pd.DataFrame, metric: str = "f1_micro") -> float:
    """Pick the cutoff maximising ``metric``, breaking ties toward recall.

    Ties are common because the grid is coarse; preferring the lower cutoff
    keeps sensitivity, which is the clinically safer direction to err.
    """
    if metric not in sweep.columns:
        raise KeyError(f"{metric!r} not in sweep columns: {list(sweep.columns)}")
    best = sweep[metric].max()
    return float(sweep.loc[sweep[metric] == best, "cutoff"].min())
