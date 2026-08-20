#!/usr/bin/env python3
"""Resample to 250 Hz and fit Incremental PCA over every lead of every record.

    python scripts/04_signal_pca.py --components 100

Memory-bounded: the ~542k x 2500 matrix is never materialised in full.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from ecg.config import ARTIFACT_DIR, SOURCE_FS_HZ, TABLE_DIR, TARGET_FS_HZ, ensure_output_dirs
from ecg.data import resample_to
from ecg.decomposition import fit_signal_pca, transform_signals


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="dataset.npz")
    parser.add_argument("--components", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    ensure_output_dirs()
    path = ARTIFACT_DIR / args.dataset
    if not path.exists():
        raise SystemExit(f"{path} not found — run scripts/01_build_dataset.py first.")

    signals = np.load(path, allow_pickle=True)["signals"]
    print(f"Loaded signals {signals.shape} at {SOURCE_FS_HZ} Hz")

    resampled = resample_to(signals, SOURCE_FS_HZ, TARGET_FS_HZ)
    print(f"Resampled to {TARGET_FS_HZ} Hz: {resampled.shape}")
    np.save(ARTIFACT_DIR / "signals_250hz.npy", resampled)

    result, (rows, samples) = fit_signal_pca(
        resampled, n_components=args.components, batch_size=args.batch_size
    )
    print(f"Incremental PCA over ({rows}, {samples})")
    print(f"  cumulative variance at {result.n_components} components: "
          f"{result.cumulative_explained_variance[-1]:.4f}")
    for target in (0.80, 0.90, 0.95):
        reached = result.components_for_variance(target)
        print(f"  {target:.0%} variance reached at {reached} components")

    projected = transform_signals(resampled, result)
    np.save(ARTIFACT_DIR / "signal_pca_features.npy", projected)
    np.savetxt(
        TABLE_DIR / "signal_pca_variance.csv",
        np.column_stack([
            np.arange(1, result.n_components + 1),
            result.explained_variance_ratio,
            result.cumulative_explained_variance,
        ]),
        delimiter=",",
        header="component,explained_variance_ratio,cumulative",
        comments="",
    )
    print(f"Wrote projected features {projected.shape}")


if __name__ == "__main__":
    main()
