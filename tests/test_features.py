import numpy as np

from ecg.features import FEATURE_NAMES, extract_features
from ecg.features.complexity import approximate_entropy, dfa_alpha, higuchi_fd, sample_entropy
from ecg.features.spectral import band_power, spectral_entropy
from ecg.features.waveform import detect_r_peaks, rr_intervals


def test_detects_expected_beat_count(synthetic_ecg):
    signal, fs, bpm = synthetic_ecg
    peaks = detect_r_peaks(signal - np.median(signal), fs)
    assert abs(len(peaks) - int(10 * bpm / 60)) <= 1


def test_rr_intervals_match_known_rate(synthetic_ecg):
    signal, fs, bpm = synthetic_ecg
    rr = rr_intervals(detect_r_peaks(signal - np.median(signal), fs), fs)
    assert np.allclose(rr, 60.0 / bpm, atol=0.02)


def test_extract_returns_every_named_feature(synthetic_ecg):
    signal, fs, _ = synthetic_ecg
    features = extract_features(signal, fs)
    assert set(features) == set(FEATURE_NAMES)


def test_short_record_yields_all_nan():
    features = extract_features(np.zeros(100), 500)
    assert all(np.isnan(v) for v in features.values())


def test_complexity_measures_return_nan_when_too_short():
    tiny = np.array([1.0, 2.0])
    assert np.isnan(approximate_entropy(tiny))
    assert np.isnan(sample_entropy(tiny))
    assert np.isnan(dfa_alpha(tiny))
    assert np.isnan(higuchi_fd(tiny))


def test_regular_series_is_less_complex_than_noise():
    rng = np.random.default_rng(0)
    regular = np.tile([0.8, 0.82], 60)
    noisy = rng.uniform(0.6, 1.0, 120)
    assert sample_entropy(regular) < sample_entropy(noisy)


def test_spectral_entropy_peaks_for_flat_spectrum():
    flat = np.ones(64)
    peaked = np.zeros(64)
    peaked[0] = 1.0
    assert spectral_entropy(flat) > spectral_entropy(peaked)


def test_band_power_outside_range_is_nan():
    freqs = np.linspace(0.0, 1.0, 32)
    psd = np.ones_like(freqs)
    assert np.isnan(band_power(freqs, psd, 5.0, 6.0))
