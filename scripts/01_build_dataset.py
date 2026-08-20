#!/usr/bin/env python3
"""Read the WFDB tree, attach SNOMED labels, and cache the result.

    python scripts/01_build_dataset.py --limit 500     # quick smoke run
    python scripts/01_build_dataset.py                 # full ~45k records
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from ecg.config import ARTIFACT_DIR, Paths, ensure_output_dirs
from ecg.data import attach_conditions, build_condition_lookup, canonical_lead_names, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=None, help="read at most N records")
    parser.add_argument("--out", default="dataset.npz", help="cache filename under artifacts/")
    args = parser.parse_args()

    ensure_output_dirs()
    paths = Paths().require()

    print(f"Reading records from {paths.records}")
    dataset = load_dataset(paths.records, limit=args.limit)
    print(f"  signals {dataset.signals.shape}, metadata {dataset.metadata.shape}")

    lookup = build_condition_lookup(paths.condition_names, paths.remaining_codes)
    dataset.metadata = attach_conditions(dataset.metadata, lookup)
    print(f"  {len(lookup)} SNOMED codes mapped to condition labels")

    canonical, per_record = canonical_lead_names(dataset.metadata, paths.records)
    deviating = sum(1 for names in per_record if list(names) != canonical)
    print(f"  canonical leads ({len(canonical)}): {canonical}")
    if deviating:
        print(f"  warning: {deviating} record(s) deviate from the canonical lead order")

    out_path = ARTIFACT_DIR / args.out
    np.savez_compressed(
        out_path,
        signals=dataset.signals,
        metadata=dataset.metadata.to_json(orient="records"),
        lead_names=np.array(canonical, dtype=object),
    )
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
