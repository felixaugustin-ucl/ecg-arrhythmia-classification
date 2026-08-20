"""Assemble the per-record feature vector from the domain modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

from ecg.features import complexity, spectral, waveform

#: Column order of the extracted feature table. ``mean_rr`` is intermediate
#: only — it feeds the Bazett correction and is not itself a model input.
FEATURE_NAMES = [
    "mean_heart_rate",
    "sdnn",
    "rmssd",
    "pnn50",
    "pr_interval",
    "qrs_duration",
    "qt_interval",
    "qtc_bazett",
    "st_deviation",
    "r_wave_amplitude",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "total_power",
    "qrs_area",
    "t_wave_symmetry_index",
    "wavelet_energy_scale_j",
    "spectral_entropy",
    "approximate_entropy",
    "sample_entropy",
    "dfa_scaling_exponent",
    "higuchi_fractal_dimension",
]


def extract_features(signal_lead: np.ndarray, fs_hz: float) -> dict[str, float]:
    """Compute every feature for a single lead of a single record.

    Returns an all-NaN vector for records too short to analyse, so the output
    table stays aligned with the input regardless of record quality.
    """
    x = np.asarray(signal_lead, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < waveform.MIN_SAMPLES:
        return dict.fromkeys(FEATURE_NAMES, np.nan)

    centered = x - float(np.median(x))
    r_peaks = waveform.detect_r_peaks(centered, fs_hz)
    rr = waveform.rr_intervals(r_peaks, fs_hz)

    hrv = waveform.hrv_time_domain(rr)
    morphology = waveform.beat_morphology(centered, r_peaks, fs_hz)
    bands = spectral.hrv_band_powers(rr)

    _, psd_signal = welch(centered, fs=fs_hz, nperseg=min(1024, centered.size))

    features = {
        "mean_heart_rate": hrv["mean_heart_rate"],
        "sdnn": hrv["sdnn"],
        "rmssd": hrv["rmssd"],
        "pnn50": hrv["pnn50"],
        "qtc_bazett": waveform.qtc_bazett(morphology["qt_interval"], hrv["mean_rr"]),
        "wavelet_energy_scale_j": waveform.wavelet_energy(centered),
        "spectral_entropy": spectral.spectral_entropy(psd_signal),
        "approximate_entropy": complexity.approximate_entropy(rr, m=2),
        "sample_entropy": complexity.sample_entropy(rr, m=2),
        "dfa_scaling_exponent": complexity.dfa_alpha(rr),
        "higuchi_fractal_dimension": complexity.higuchi_fd(rr, kmax=6),
        **morphology,
        **bands,
    }
    return {name: features[name] for name in FEATURE_NAMES}


def extract_feature_table(
    dataset, lead_index: int, fs_hz: float, progress: bool = True
) -> pd.DataFrame:
    """Run :func:`extract_features` across every record in a dataset."""
    rows = []
    indices = range(len(dataset))
    if progress:
        try:
            from tqdm import tqdm

            indices = tqdm(indices, desc="Extracting features", unit="rec")
        except ImportError:
            pass

    for i in indices:
        rows.append(extract_features(dataset.lead(i, lead_index), fs_hz))

    return pd.DataFrame(rows, columns=FEATURE_NAMES).astype(np.float32)
