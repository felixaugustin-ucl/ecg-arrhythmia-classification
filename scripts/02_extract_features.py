#!/usr/bin/env python3
"""Compute the 22 handcrafted features for every record.

    python scripts/02_extract_features.py

This is the slow stage: entropy and DFA are O(n^2) in beats per record.
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ecg.config import ARTIFACT_DIR, FEATURE_LEAD_INDEX, SOURCE_FS_HZ, TABLE_DIR, ensure_output_dirs
from ecg.data.wfdb import EcgDataset
from ecg.features import extract_feature_table


def load_cached(name: str) -> EcgDataset:
    path = ARTIFACT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/01_build_dataset.py first.")
    cached = np.load(path, allow_pickle=True)
    metadata = pd.DataFrame(json.loads(str(cached["metadata"])))
    return EcgDataset(signals=cached["signals"], metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="dataset.npz")
    parser.add_argument("--lead", type=int, default=FEATURE_LEAD_INDEX)
    parser.add_argument("--fs", type=int, default=SOURCE_FS_HZ)
    parser.add_argument("--out", default="features.parquet")
    args = parser.parse_args()

    ensure_output_dirs()
    dataset = load_cached(args.dataset)
    print(f"Extracting features from lead {args.lead} at {args.fs} Hz for {len(dataset)} records")

    features = extract_feature_table(dataset, lead_index=args.lead, fs_hz=args.fs)

    missing = features.isna().mean().sort_values(ascending=False)
    print("\nMissingness by feature:")
    print((missing[missing > 0] * 100).round(1).to_string() or "  none")

    out_path = ARTIFACT_DIR / args.out
    features.to_parquet(out_path, index=False)
    features.describe().T.to_csv(TABLE_DIR / "feature_summary.csv")
    print(f"\nWrote {out_path}  shape={features.shape}")


if __name__ == "__main__":
    main()
