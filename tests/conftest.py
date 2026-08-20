import numpy as np
import pytest


@pytest.fixture
def synthetic_ecg():
    """A clean 10-second, 500 Hz sinusoidal 'ECG' with known beat spacing.

    Not physiologically realistic, but the R-peak spacing is exact, which is
    what the interval and HRV assertions need.
    """
    fs = 500
    duration_s = 10
    bpm = 60.0
    t = np.arange(0, duration_s, 1 / fs)
    beat_hz = bpm / 60.0
    signal = np.zeros_like(t)
    for beat in range(int(duration_s * beat_hz)):
        centre = beat / beat_hz
        signal += np.exp(-(((t - centre) / 0.01) ** 2))
    return signal.astype(np.float32), fs, bpm
