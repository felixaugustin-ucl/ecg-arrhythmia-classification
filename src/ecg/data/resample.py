"""Signal resampling.

The raw-signal models train at 250 Hz rather than the native 500 Hz: it halves
memory for the Incremental PCA pass and the CNN, and the diagnostic content of
a surface ECG sits well below the 125 Hz Nyquist limit that 250 Hz affords.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def resample_to(
    signals: np.ndarray, source_fs: int, target_fs: int, batch_size: int = 256
) -> np.ndarray:
    """Resample a ``(record, lead, sample)`` tensor along the time axis.

    Uses polyphase filtering, which anti-aliases as it decimates — plain slicing
    would fold high-frequency noise back into the band of interest.

    NaN padding is zero-filled before filtering, since ``resample_poly`` would
    otherwise propagate a single NaN across the whole output record.
    """
    if signals.ndim != 3:
        raise ValueError(f"Expected (record, lead, sample), got shape {signals.shape}")
    if source_fs == target_fs:
        return np.asarray(signals, dtype=np.float32)

    ratio = Fraction(target_fs, source_fs).limit_denominator()
    up, down = ratio.numerator, ratio.denominator

    n_records, n_leads, n_samples = signals.shape
    out_samples = int(np.ceil(n_samples * up / down))
    out = np.empty((n_records, n_leads, out_samples), dtype=np.float32)

    for start in range(0, n_records, batch_size):
        batch = np.nan_to_num(
            np.asarray(signals[start : start + batch_size], dtype=np.float32), nan=0.0
        )
        resampled = resample_poly(batch, up, down, axis=-1)
        out[start : start + batch.shape[0]] = resampled[..., :out_samples].astype(np.float32)

    return out
