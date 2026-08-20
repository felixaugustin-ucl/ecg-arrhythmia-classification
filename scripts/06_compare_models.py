#!/usr/bin/env python3
"""Collect every model's summary into one comparison table and chart.

    python scripts/06_compare_models.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from ecg.config import FIGURE_DIR, TABLE_DIR, ensure_output_dirs


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    ensure_output_dirs()
    summaries = sorted(TABLE_DIR.glob("*_summary.csv"))
    if not summaries:
        raise SystemExit(f"No *_summary.csv under {TABLE_DIR} — train a model first.")

    comparison = pd.concat([pd.read_csv(p) for p in summaries], ignore_index=True)
    comparison = comparison.sort_values("f1_macro", ascending=False).reset_index(drop=True)

    out_csv = TABLE_DIR / "model_comparison.csv"
    comparison.to_csv(out_csv, index=False)
    print(comparison.to_string(index=False))
    print(f"\nWrote {out_csv}")

    import plotly.graph_objects as go

    from ecg.viz import SERIES_COLORS, apply_theme

    fig = go.Figure()
    for metric in ("f1_micro", "f1_macro"):
        if metric in comparison.columns:
            fig.add_trace(go.Bar(
                x=comparison["model"], y=comparison[metric], name=metric,
                marker_color=SERIES_COLORS.get(metric),
            ))
    apply_theme(
        fig,
        "Handcrafted Features Hold Their Own Against Raw Signal",
        "Held-out test F1 by model, at each model's selected cutoff",
        y_title="F1",
    )
    out_html = FIGURE_DIR / "model_comparison.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
