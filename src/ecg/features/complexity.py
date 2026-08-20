"""Nonlinear complexity measures over RR-interval series.

These quantify how *irregular* a rhythm is, which separates conditions that
share a similar mean rate but differ in beat-to-beat structure — atrial
fibrillation versus sinus tachycardia being the obvious pair.

All functions return NaN rather than raising when the series is too short to
support the estimate, so a single unusable record cannot abort a batch run.
"""

from __future__ import annotations

import numpy as np


def _finite(series) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    return x[np.isfinite(x)]


def _default_tolerance(x: np.ndarray, r: float | None) -> float | None:
    """Entropy tolerance defaults to 0.2 SD, the conventional choice."""
    if r is None:
        r = 0.2 * np.std(x)
    return r if np.isfinite(r) and r > 0 else None


def approximate_entropy(series, m: int = 2, r: float | None = None) -> float:
    """Approximate entropy: lower means more regular and predictable."""
    u = _finite(series)
    n = u.size
    if n < m + 2:
        return np.nan
    r = _default_tolerance(u, r)
    if r is None:
        return np.nan

    def phi(mm: int) -> float:
        x = np.array([u[i : i + mm] for i in range(n - mm + 1)])
        counts = np.empty(x.shape[0], dtype=float)
        for i in range(x.shape[0]):
            counts[i] = np.mean(np.max(np.abs(x - x[i]), axis=1) <= r)
        return float(np.mean(np.log(np.where(counts > 0, counts, 1e-12))))

    return float(phi(m) - phi(m + 1))


def sample_entropy(series, m: int = 2, r: float | None = None) -> float:
    """Sample entropy: like ApEn but without self-match bias.

    Returns ``inf`` when no longer-template match exists — a degenerate but
    meaningful outcome that the caller filters out downstream.
    """
    u = _finite(series)
    n = u.size
    if n < m + 2:
        return np.nan
    r = _default_tolerance(u, r)
    if r is None:
        return np.nan

    def count(mm: int) -> int:
        x = np.array([u[i : i + mm] for i in range(n - mm + 1)])
        total = 0
        for i in range(x.shape[0] - 1):
            total += int(np.sum(np.max(np.abs(x[i + 1 :] - x[i]), axis=1) <= r))
        return total

    b, a = count(m), count(m + 1)
    if b == 0:
        return np.nan
    if a == 0:
        return np.inf
    return float(-np.log(a / b))


def dfa_alpha(series) -> float:
    """Detrended fluctuation analysis exponent.

    ~0.5 is uncorrelated noise, 0.5-1.0 persistent long-range correlation,
    >1.0 non-stationary. Needs at least 12 points to be worth reporting.
    """
    x = _finite(series)
    n = x.size
    if n < 12:
        return np.nan

    y = np.cumsum(x - np.mean(x))
    scales = np.unique(
        np.floor(np.logspace(np.log10(4), np.log10(max(5, n // 2)), num=6)).astype(int)
    )

    fluctuations, valid_scales = [], []
    for scale in scales:
        if scale < 4 or n // scale < 2:
            continue
        rms = []
        for start in range(0, n - scale + 1, scale):
            segment = y[start : start + scale]
            t = np.arange(scale, dtype=float)
            slope, intercept = np.polyfit(t, segment, 1)
            rms.append(np.sqrt(np.mean((segment - (slope * t + intercept)) ** 2)))
        if rms:
            fluctuations.append(np.mean(rms))
            valid_scales.append(scale)

    fluctuations = np.asarray(fluctuations, dtype=float)
    valid_scales = np.asarray(valid_scales, dtype=float)
    mask = (fluctuations > 0) & np.isfinite(fluctuations)
    if mask.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(valid_scales[mask]), np.log(fluctuations[mask]), 1)
    return float(slope)


def higuchi_fd(series, kmax: int = 6) -> float:
    """Higuchi fractal dimension: higher means more complex dynamics."""
    x = _finite(series)
    n = x.size
    if n < 16:
        return np.nan

    lengths, k_values = [], []
    for k in range(1, kmax + 1):
        per_offset = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if idx.size < 2:
                continue
            curve_length = np.sum(np.abs(np.diff(x[idx])))
            norm = (n - 1) / (((n - m - 1) // k) * k)
            per_offset.append((curve_length * norm) / k)
        if per_offset:
            lengths.append(np.mean(per_offset))
            k_values.append(k)

    lengths = np.asarray(lengths, dtype=float)
    k_values = np.asarray(k_values, dtype=float)
    mask = (lengths > 0) & np.isfinite(lengths)
    if mask.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(1.0 / k_values[mask]), np.log(lengths[mask]), 1)
    return float(slope)
