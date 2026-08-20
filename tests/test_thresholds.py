import numpy as np
import pandas as pd

from ecg.evaluation.metrics import multilabel_metrics, per_label_metrics
from ecg.evaluation.thresholds import CUTOFF_GRID, best_cutoff, sweep_cutoffs


def test_sweep_covers_every_cutoff():
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6]])
    sweep = sweep_cutoffs(y_true, y_proba)
    assert len(sweep) == len(CUTOFF_GRID)
    assert {"cutoff", "f1_micro", "f1_macro"} <= set(sweep.columns)


def test_perfect_separation_scores_one_at_the_right_cutoff():
    y_true = np.array([[1, 0], [0, 1]])
    y_proba = np.array([[0.95, 0.05], [0.05, 0.95]])
    assert sweep_cutoffs(y_true, y_proba)["f1_micro"].max() == 1.0


def test_best_cutoff_prefers_recall_on_ties():
    # Two cutoffs tie on f1_micro; the lower one must win.
    sweep = pd.DataFrame({"cutoff": [0.2, 0.5, 0.8], "f1_micro": [0.9, 0.9, 0.4]})
    assert best_cutoff(sweep, "f1_micro") == 0.2


def test_best_cutoff_rejects_unknown_metric():
    sweep = pd.DataFrame({"cutoff": [0.5], "f1_micro": [0.5]})
    try:
        best_cutoff(sweep, "not_a_metric")
    except KeyError:
        return
    raise AssertionError("expected KeyError for an unknown metric")


def test_metrics_report_micro_and_macro():
    y = np.array([[1, 0], [0, 1]])
    scores = multilabel_metrics(y, y)
    assert scores["f1_micro"] == 1.0 and scores["f1_macro"] == 1.0


def test_per_label_metrics_sorted_by_support():
    y_true = np.array([[1, 1], [0, 1], [0, 1]])
    y_pred = np.array([[1, 1], [0, 1], [0, 0]])
    table = per_label_metrics(y_true, y_pred, np.array(["rare", "common"]))
    assert list(table["label"]) == ["common", "rare"]
    assert list(table["support"]) == [3, 1]
