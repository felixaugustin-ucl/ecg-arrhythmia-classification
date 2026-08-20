"""Time-domain waveform features: R-peak detection, intervals, HRV statistics.

Fiducial points are located by fixed physiological offsets from each detected
R peak rather than by a full delineation algorithm. That is a deliberate
simplification: it is fast enough for ~45k records and robust to noise, at the
cost of per-beat precision. Intervals should be read as population-level
descriptors, not as clinical measurements on any individual record.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ecg.features._compat import trapezoid

#: Physiologically plausible RR bounds (seconds) — 24 to 240 bpm.
RR_MIN_S, RR_MAX_S = 0.25, 2.5

#: Offsets from the R peak, in seconds, used to bracket each wave.
QRS_ONSET_S, QRS_OFFSET_S = 0.04, 0.06
P_WINDOW_S = (0.28, 0.06)
P_ONSET_S = 0.04
ST_OFFSET_S = 0.08
T_WINDOW_S = (0.10, 0.50)
T_ONSET_S, T_OFFSET_S = 0.12, 0.16

#: Records shorter than this carry too few beats to describe.
MIN_SAMPLES = 500


def nanmean_or_nan(values) -> float:
    """Mean that returns NaN for empty or all-NaN input instead of warning."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def detect_r_peaks(centered: np.ndarray, fs_hz: float) -> np.ndarray:
    """Locate R peaks by prominence, with a refractory-period minimum spacing."""
    prominence = max(1e-6, 0.6 * np.nanstd(centered))
    peaks, _ = find_peaks(
        centered, distance=max(1, int(0.25 * fs_hz)), prominence=prominence
    )
    return peaks


def rr_intervals(r_peaks: np.ndarray, fs_hz: float) -> np.ndarray:
    """RR intervals in seconds, filtered to physiologically plausible values."""
    if r_peaks.size < 2:
        return np.array([], dtype=float)
    rr = np.diff(r_peaks) / fs_hz
    return rr[(rr > RR_MIN_S) & (rr < RR_MAX_S)]


def hrv_time_domain(rr: np.ndarray) -> dict[str, float]:
    """Standard time-domain HRV statistics from an RR series."""
    mean_rr = np.nan if rr.size == 0 else float(np.mean(rr))
    mean_hr = np.nan if not np.isfinite(mean_rr) or mean_rr <= 0 else 60.0 / mean_rr
    sdnn = np.nan if rr.size < 2 else float(np.std(rr, ddof=1))
    drr = np.diff(rr) if rr.size >= 2 else np.array([], dtype=float)
    rmssd = np.nan if drr.size == 0 else float(np.sqrt(np.mean(drr**2)))
    pnn50 = np.nan if drr.size == 0 else float(np.mean(np.abs(drr) > 0.05))
    return {
        "mean_rr": mean_rr,
        "mean_heart_rate": mean_hr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
    }


def beat_morphology(
    centered: np.ndarray, r_peaks: np.ndarray, fs_hz: float
) -> dict[str, float]:
    """Per-beat intervals and amplitudes, averaged across all detected beats."""
    n = centered.size
    pr, qrs, qt, st, r_amp, qrs_area, symmetry = [], [], [], [], [], [], []

    for r in r_peaks:
        qrs_on = max(0, r - int(QRS_ONSET_S * fs_hz))
        qrs_off = min(n - 1, r + int(QRS_OFFSET_S * fs_hz))

        # P wave: search a window before the QRS onset for the dominant deflection.
        p_start = max(0, r - int(P_WINDOW_S[0] * fs_hz))
        p_end = max(p_start + 1, r - int(P_WINDOW_S[1] * fs_hz))
        p_segment = centered[p_start:p_end]
        if p_segment.size > 4:
            p_peak = p_start + int(np.argmax(np.abs(p_segment)))
            p_onset = max(p_start, p_peak - int(P_ONSET_S * fs_hz))
            pr.append((qrs_on - p_onset) / fs_hz)

        qrs.append((qrs_off - qrs_on) / fs_hz)
        qrs_area.append(float(trapezoid(np.abs(centered[qrs_on : qrs_off + 1])) / fs_hz))
        r_amp.append(float(centered[r]))
        st.append(float(centered[min(n - 1, qrs_off + int(ST_OFFSET_S * fs_hz))]))

        # T wave: dominant deflection in the repolarisation window after R.
        t_start = min(n - 2, r + int(T_WINDOW_S[0] * fs_hz))
        t_end = min(n - 1, r + int(T_WINDOW_S[1] * fs_hz))
        if t_end - t_start > 6:
            t_peak = t_start + int(np.argmax(np.abs(centered[t_start:t_end])))
            t_on = max(t_start, t_peak - int(T_ONSET_S * fs_hz))
            t_off = min(t_end, t_peak + int(T_OFFSET_S * fs_hz))
            qt.append((t_off - qrs_on) / fs_hz)
            rise = max((t_peak - t_on) / fs_hz, 1e-6)
            decay = max((t_off - t_peak) / fs_hz, 1e-6)
            symmetry.append(float(rise / decay))

    return {
        "pr_interval": nanmean_or_nan(pr),
        "qrs_duration": nanmean_or_nan(qrs),
        "qt_interval": nanmean_or_nan(qt),
        "st_deviation": nanmean_or_nan(st),
        "r_wave_amplitude": nanmean_or_nan(r_amp),
        "qrs_area": nanmean_or_nan(qrs_area),
        "t_wave_symmetry_index": nanmean_or_nan(symmetry),
    }


def qtc_bazett(qt_interval: float, mean_rr: float) -> float:
    """Rate-correct QT by Bazett's formula (QT divided by root RR)."""
    if not np.isfinite(qt_interval) or not np.isfinite(mean_rr) or mean_rr <= 0:
        return np.nan
    return float(qt_interval / np.sqrt(mean_rr))


def wavelet_energy(centered: np.ndarray, window: int = 8) -> float:
    """Energy in the high-frequency detail left after a moving-average smooth.

    A cheap stand-in for a single-scale wavelet detail coefficient, capturing
    sharp deflections that the smoothed trend discards.
    """
    kernel = np.ones(window, dtype=float) / window
    detail = centered - np.convolve(centered, kernel, mode="same")
    return float(np.mean(detail**2))
