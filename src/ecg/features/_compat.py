"""Small shims for API differences across supported NumPy versions."""

from __future__ import annotations

import numpy as np

# np.trapz was renamed to np.trapezoid in NumPy 2.0 and deprecated thereafter.
trapezoid = getattr(np, "trapezoid", None) or np.trapz

__all__ = ["trapezoid"]
