"""Dimensionality reduction for the two PCA-based feature sources.

Two different jobs share this module:

* ``metrics_pca_13c`` — ordinary PCA over the 18 handcrafted features, kept to
  the components explaining a target share of variance.
* ``signal_pca_100`` — Incremental PCA over every lead of every patient at
  250 Hz. The matrix is ~542k rows by 2500 columns and does not fit in memory,
  so both the scaler and the PCA are fitted in batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler

DEFAULT_BATCH_SIZE = 128
DEFAULT_SIGNAL_COMPONENTS = 100


@dataclass
class PCAResult:
    """A fitted PCA together with the diagnostics needed to justify its rank."""

    scaler: StandardScaler
    model: PCA | IncrementalPCA
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray

    @property
    def n_components(self) -> int:
        return int(self.model.n_components_)

    def components_for_variance(self, target: float) -> int:
        """Smallest component count reaching ``target`` cumulative variance."""
        reached = np.searchsorted(self.cumulative_explained_variance, target) + 1
        return int(min(reached, self.n_components))


def _summarise(scaler, model) -> PCAResult:
    ratio = np.asarray(model.explained_variance_ratio_, dtype=float)
    return PCAResult(
        scaler=scaler,
        model=model,
        explained_variance_ratio=ratio,
        cumulative_explained_variance=np.cumsum(ratio),
    )


def fit_feature_pca(X: np.ndarray, n_components: int | None = None) -> PCAResult:
    """Standardise then PCA the handcrafted feature matrix.

    Standardisation is not optional here: the features span heart rate in bpm
    and entropies near zero, so an unscaled PCA would simply rank features by
    unit magnitude.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.asarray(X, dtype=np.float64))
    model = PCA(n_components=n_components)
    model.fit(X_scaled)
    return _summarise(scaler, model)


def fit_signal_pca(
    signals: np.ndarray,
    n_components: int = DEFAULT_SIGNAL_COMPONENTS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[PCAResult, tuple[int, int]]:
    """Incremental PCA over a ``(record, lead, sample)`` tensor.

    Every lead of every record becomes one row, so the basis is learned across
    leads rather than per-lead. Returns the fit and the ``(rows, samples)``
    shape actually used.
    """
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 3:
        raise ValueError(f"Expected (record, lead, sample), got {signals.shape}")

    n_records, n_leads, n_samples = signals.shape
    n_rows = n_records * n_leads
    rows = signals.reshape(n_rows, n_samples)

    effective = min(n_components, n_rows, n_samples)
    if effective != n_components:
        print(f"Reduced n_components {n_components} -> {effective} to fit data shape.")

    scaler = StandardScaler(copy=False)
    for start in range(0, n_rows, batch_size):
        scaler.partial_fit(rows[start : start + batch_size].astype(np.float32, copy=False))

    model = IncrementalPCA(n_components=effective, batch_size=batch_size)
    for start in range(0, n_rows, batch_size):
        batch = rows[start : start + batch_size].astype(np.float32, copy=False)
        model.partial_fit(scaler.transform(batch))

    return _summarise(scaler, model), (n_rows, n_samples)


def transform_signals(
    signals: np.ndarray, result: PCAResult, batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    """Project signals into the PCA basis and pool leads back per record.

    Lead components are averaged so each record yields one feature vector,
    which is what the record-level classifier expects.
    """
    signals = np.asarray(signals, dtype=np.float32)
    n_records, n_leads, n_samples = signals.shape
    rows = signals.reshape(n_records * n_leads, n_samples)

    projected = np.empty((rows.shape[0], result.n_components), dtype=np.float32)
    for start in range(0, rows.shape[0], batch_size):
        batch = rows[start : start + batch_size].astype(np.float32, copy=False)
        projected[start : start + batch.shape[0]] = result.model.transform(
            result.scaler.transform(batch)
        )

    return projected.reshape(n_records, n_leads, result.n_components).mean(axis=1)
