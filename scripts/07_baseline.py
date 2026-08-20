#!/usr/bin/env python3
"""Score feature-free baselines, to make the model numbers interpretable.

    python scripts/07_baseline.py

Reads only the WFDB .hea headers, never the .mat signals, so it runs in
seconds against the full dataset without the rest of the pipeline. Uses the
same split seed and test size as the models, so the rows are directly
comparable to outputs/tables/*_summary.csv.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from ecg.config import RANDOM_SEED, TABLE_DIR, TEST_SIZE, Paths, ensure_output_dirs
from ecg.data.wfdb import extract_dx_codes
from ecg.evaluation.baseline import evaluate_baselines, label_statistics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args()

    ensure_output_dirs()
    paths = Paths().require()

    # Sorted by path, matching load_dataset's ordering over the same tree, so
    # the split indices line up with the models' splits.
    headers = sorted(paths.records.rglob("*.hea"))
    if not headers:
        raise SystemExit(f"No .hea headers under {paths.records}")
    print(f"Reading {len(headers)} headers")

    label_sets = [sorted(set(extract_dx_codes(h))) for h in headers]
    y = MultiLabelBinarizer().fit_transform(label_sets)

    stats = label_statistics(y)
    print("\nLabel matrix:")
    for key, value in stats.items():
        print(f"  {key:<26} {value:.4f}" if isinstance(value, float) else f"  {key:<26} {value}")

    y_train, y_test = train_test_split(
        y, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    print(f"\nTrain {y_train.shape}, test {y_test.shape}")

    results = evaluate_baselines(y_train, y_test)
    print("\nBaseline performance on the held-out test split:\n")
    print(results.round(4).to_string(index=False))

    out = TABLE_DIR / "baseline_summary.csv"
    results.to_csv(out, index=False)
    pd.DataFrame([stats]).to_csv(TABLE_DIR / "label_statistics.csv", index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
