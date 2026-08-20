"""Lead names, read from the WFDB header rather than assumed."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


def lead_names_for_record(hea_path: Path) -> list[str]:
    """Parse per-lead names from a WFDB header's signal specification lines."""
    try:
        raw = hea_path.read_text(encoding="utf-8", errors="ignore")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    except FileNotFoundError:
        return []
    if not lines:
        return []

    tokens = lines[0].split()
    if len(tokens) < 2:
        return []
    try:
        n_leads = int(tokens[1])
    except ValueError:
        return []

    names = []
    for lead_line in lines[1 : 1 + n_leads]:
        parts = lead_line.split()
        names.append(parts[-1] if parts else "Unknown")
    return names


def canonical_lead_names(
    metadata: pd.DataFrame, records_root: Path
) -> tuple[list[str], pd.Series]:
    """Find the most common lead ordering across all records.

    Returns the canonical ordering and the per-record lists, so callers can
    check how many records deviate rather than silently assuming uniformity.
    """
    per_record = metadata.apply(
        lambda row: lead_names_for_record(
            records_root / Path(row["relative_path"]).with_suffix(".hea")
        ),
        axis=1,
    )
    counts = Counter(tuple(names) for names in per_record)
    canonical = list(counts.most_common(1)[0][0]) if counts else []
    return canonical, per_record
