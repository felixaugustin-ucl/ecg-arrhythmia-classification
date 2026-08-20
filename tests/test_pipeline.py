"""Structural checks on the registry, preprocessing and resampling."""

import numpy as np
import pandas as pd
import pytest

from ecg.data.resample import resample_to
from ecg.models import available_models, get_model_spec
from ecg.preprocessing import prepare


def test_every_registered_model_builds_from_its_frozen_config():
    from ecg.config import load_config

    for name in available_models():
        spec = get_model_spec(name)
        config = load_config(f"models/{name}")
        frozen = {k: v for k, v in config["best_config"].items() if k != "trial_id"}
        assert spec.build(frozen) is not None, name


def test_registry_rejects_duplicate_names():
    from ecg.models.registry import ModelSpec, register

    spec = ModelSpec(
        name="sgd_18f", description="dup", feature_source="metrics_18f", build=lambda c: None
    )
    with pytest.raises(ValueError):
        register(spec)


def test_model_spec_rejects_unknown_feature_source():
    from ecg.models.registry import ModelSpec

    with pytest.raises(ValueError):
        ModelSpec(name="x", description="", feature_source="nope", build=lambda c: None)


def test_prepare_imputes_and_binarizes():
    metrics = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.inf, 2.0, 4.0]})
    labels = pd.Series([["x"], ["x", "y"], ["y"]])
    data = prepare(metrics, labels, drop_columns=())

    assert not data.X.isna().any().any()
    assert data.y.shape == (3, 2)
    assert list(data.classes) == ["x", "y"]


def test_prepare_drops_requested_columns():
    metrics = pd.DataFrame({"keep": [1.0, 2.0], "drop_me": [3.0, 4.0]})
    data = prepare(metrics, pd.Series([["a"], ["b"]]), drop_columns=("drop_me",))
    assert data.feature_names == ["keep"]


def test_resample_halves_length_and_preserves_shape():
    signals = np.random.default_rng(0).normal(size=(4, 12, 1000)).astype(np.float32)
    out = resample_to(signals, 500, 250)
    assert out.shape == (4, 12, 500)


def test_resample_is_identity_at_matching_rate():
    signals = np.ones((2, 3, 100), dtype=np.float32)
    assert np.array_equal(resample_to(signals, 250, 250), signals)


def test_resample_tolerates_nan_padding():
    signals = np.full((1, 2, 1000), np.nan, dtype=np.float32)
    signals[0, :, :500] = 1.0
    out = resample_to(signals, 500, 250)
    assert np.isfinite(out).all()
