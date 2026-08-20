"""End-to-end smoke test on synthetic signals.

The real dataset is 5.3 GB, so CI cannot run the actual pipeline. This builds
a tiny two-rhythm dataset instead — 60 bpm against 100 bpm — and drives the
whole path: feature extraction, preprocessing, cutoff sweep, fit, predict.

It asserts the pipeline *connects*, not that the published numbers reproduce.
The synthetic task is deliberately trivial, so a high score here says nothing
about real performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from ecg.config import load_config
from ecg.data.wfdb import EcgDataset
from ecg.evaluation import best_cutoff, multilabel_metrics, per_label_metrics, sweep_from_cv
from ecg.features import extract_feature_table
from ecg.models import get_model_spec
from ecg.preprocessing import prepare

FS = 500
N_RECORDS = 60
N_LEADS = 12
N_SAMPLES = 5000


@pytest.fixture(scope="module")
def synthetic_dataset() -> tuple[EcgDataset, pd.Series]:
    rng = np.random.default_rng(0)
    t = np.arange(N_SAMPLES) / FS
    signals = np.zeros((N_RECORDS, N_LEADS, N_SAMPLES), dtype=np.float32)
    labels = []

    for i in range(N_RECORDS):
        bpm = 60.0 if i % 2 == 0 else 100.0
        beat_hz = bpm / 60.0
        wave = np.zeros_like(t)
        for beat in range(int(N_SAMPLES / FS * beat_hz)):
            wave += np.exp(-(((t - beat / beat_hz) / 0.01) ** 2))
        signals[i] = (wave + rng.normal(0, 0.01, N_SAMPLES)).astype(np.float32)
        labels.append(["brady"] if bpm == 60.0 else ["tachy"])

    metadata = pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(N_RECORDS)],
            "n_samples": [N_SAMPLES] * N_RECORDS,
            "n_leads": [N_LEADS] * N_RECORDS,
        }
    )
    return EcgDataset(signals=signals, metadata=metadata), pd.Series(labels)


def test_full_pipeline_runs(synthetic_dataset):
    dataset, labels = synthetic_dataset

    features = extract_feature_table(dataset, lead_index=1, fs_hz=FS, progress=False)
    assert len(features) == N_RECORDS

    data = prepare(features, labels)
    assert not data.X.isna().any().any()
    assert list(data.classes) == ["brady", "tachy"]

    spec = get_model_spec("sgd_18f")
    frozen = {
        k: v for k, v in load_config("models/sgd_18f")["best_config"].items() if k != "trial_id"
    }
    model = spec.build(frozen)

    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.3, random_state=42
    )
    sweep, proba = sweep_from_cv(model, X_train, y_train, cv=3)
    assert proba.shape[0] == len(X_train)

    cutoff = best_cutoff(sweep, "f1_micro")
    model.fit(X_train.to_numpy(dtype=np.float32), y_train)
    proba = np.asarray(model.predict_proba(X_test.to_numpy(dtype=np.float32)))
    y_pred = (proba >= cutoff).astype(int)

    scores = multilabel_metrics(y_test, y_pred)
    # A rate-based split this clean must be learnable; a low score here means
    # the pipeline is mis-wired, not that the model is weak.
    assert scores["f1_micro"] > 0.8

    per_label = per_label_metrics(y_test, y_pred, data.classes)
    assert set(per_label["label"]) == {"brady", "tachy"}


def test_feature_extraction_separates_the_two_rates(synthetic_dataset):
    """Heart rate must actually differ between the two synthetic classes."""
    dataset, labels = synthetic_dataset
    features = extract_feature_table(dataset, lead_index=1, fs_hz=FS, progress=False)
    is_brady = labels.apply(lambda names: names == ["brady"]).to_numpy()

    brady_hr = features.loc[is_brady, "mean_heart_rate"].mean()
    tachy_hr = features.loc[~is_brady, "mean_heart_rate"].mean()
    assert brady_hr == pytest.approx(60, abs=3)
    assert tachy_hr == pytest.approx(100, abs=3)
