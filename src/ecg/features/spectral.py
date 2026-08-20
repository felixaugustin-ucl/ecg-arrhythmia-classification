"""Frequency-domain features: HRV band powers and spectral entropy."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from ecg.features._compat import trapezoid

#: Standard HRV bands (Hz). LF/HF ratio is a sympathovagal balance proxy.
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)
TOTAL_BAND = (0.04, 0.40)

#: RR series are irregularly sampled; interpolate onto this grid before Welch.
RR_RESAMPLE_HZ = 4.0


def spectral_entropy(psd) -> float:
    """Shannon entropy of a power spectrum normalised to a distribution."""
    psd = np.asarray(psd, dtype=float)
    psd = np.where(np.isfinite(psd) & (psd > 0), psd, 0.0)
    total = psd.sum()
    if total <= 0:
        return np.nan
    p = psd / total
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    return float(-(p * np.log(p)).sum())


def band_power(freqs, psd, low: float, high: float) -> float:
    """Integrate a PSD over ``[low, high]`` by interpolating onto a dense grid.

    Direct bin summation is unreliable here: short RR series give very coarse
    frequency resolution, so a band can contain one bin or none. Interpolating
    first makes the estimate stable across record lengths.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)

    valid = np.isfinite(freqs) & np.isfinite(psd)
    freqs, psd = freqs[valid], psd[valid]
    if freqs.size < 2:
        return np.nan

    order = np.argsort(freqs)
    freqs, psd = freqs[order], psd[order]
    freqs, idx = np.unique(freqs, return_index=True)
    psd = psd[idx]
    if freqs.size < 2:
        return np.nan

    if high <= freqs[0] or low >= freqs[-1]:
        return np.nan
    left, right = max(low, freqs[0]), min(high, freqs[-1])
    if right <= left:
        return np.nan

    step = max(1e-4, float(np.median(np.diff(freqs))))
    grid = np.linspace(left, right, max(64, int(np.ceil((right - left) / step))))
    power = float(trapezoid(np.interp(grid, freqs, psd), grid))
    return power if np.isfinite(power) and power >= 0 else np.nan


def rr_spectrum(rr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of an RR-interval series, interpolated to a uniform grid."""
    empty = (np.array([]), np.array([]))
    if rr.size < 3:
        return empty

    rr_time = np.cumsum(rr)
    rr_time -= rr_time[0]
    if rr_time[-1] <= 0:
        return empty

    uniform_t = np.arange(0, rr_time[-1], 1.0 / RR_RESAMPLE_HZ)
    if uniform_t.size < 8:
        return empty

    rr_interp = np.interp(uniform_t, rr_time, rr)
    return welch(rr_interp, fs=RR_RESAMPLE_HZ, nperseg=min(256, rr_interp.size))


def hrv_band_powers(rr: np.ndarray) -> dict[str, float]:
    """LF, HF, total power and the LF/HF ratio from an RR series."""
    freqs, psd = rr_spectrum(rr)
    if psd.size == 0:
        return dict.fromkeys(("lf_power", "hf_power", "total_power", "lf_hf_ratio"), np.nan)

    lf = band_power(freqs, psd, *LF_BAND)
    hf = band_power(freqs, psd, *HF_BAND)
    total = band_power(freqs, psd, *TOTAL_BAND)
    ratio = np.nan if not (np.isfinite(lf) and np.isfinite(hf) and hf != 0) else float(lf / hf)
    return {"lf_power": lf, "hf_power": hf, "total_power": total, "lf_hf_ratio": ratio}
