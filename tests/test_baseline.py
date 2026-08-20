import numpy as np

from ecg.evaluation.baseline import (
    evaluate_baselines,
    label_statistics,
    predict_all,
    predict_constant,
)


def _toy():
    # Label 0 is common, label 2 never occurs.
    return np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]])


def test_predict_constant_picks_most_frequent():
    y = _toy()
    pred = predict_constant(y, n_test=2, k=1)
    assert pred.shape == (2, 3)
    assert pred[:, 0].all() and not pred[:, 1].any()


def test_predict_constant_k_zero_predicts_nothing():
    assert predict_constant(_toy(), n_test=3, k=0).sum() == 0


def test_predict_constant_rejects_negative_k():
    try:
        predict_constant(_toy(), n_test=1, k=-1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for negative k")


def test_predict_all_is_all_ones():
    assert predict_all(3, 4).sum() == 12


def test_evaluate_baselines_covers_each_strategy():
    y = _toy()
    results = evaluate_baselines(y, y, k_values=(1, 2))
    assert list(results["strategy"]) == [
        "predict nothing", "top-1 most frequent", "top-2 most frequent", "predict everything"
    ]
    assert (results.loc[results["strategy"] == "predict nothing", "f1_micro"] == 0).all()


def test_evaluate_baselines_skips_k_above_label_count():
    results = evaluate_baselines(_toy(), _toy(), k_values=(1, 99))
    assert "top-99 most frequent" not in set(results["strategy"])


def test_predict_everything_reaches_full_recall():
    results = evaluate_baselines(_toy(), _toy(), k_values=(1,))
    row = results[results["strategy"] == "predict everything"].iloc[0]
    assert row["recall_micro"] == 1.0


def test_label_statistics_reports_shape():
    stats = label_statistics(_toy())
    assert stats["records"] == 4
    assert stats["labels"] == 3
    assert stats["mean_labels_per_record"] == 1.25
    assert stats["top_label_prevalence"] == 0.75
