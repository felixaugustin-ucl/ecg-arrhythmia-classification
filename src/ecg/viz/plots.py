"""Result charts, each written once and reused by every model.

In the notebook these were copy-pasted per model, at 88-94% textual
similarity, with the model name baked into every variable. Here the model name
is an argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ecg.viz.theme import SERIES_COLORS, apply_theme

#: Ray records trial completion under whichever of these it has available.
_CHRONO_COLUMNS = ("timestamp", "date", "time_total_s", "training_iteration")


def _chronological(results: pd.DataFrame) -> pd.DataFrame:
    """Order tuning trials by completion time, whichever column carries it."""
    df = results.copy()
    for column in _CHRONO_COLUMNS:
        if column in df.columns:
            parser = pd.to_datetime if column == "date" else pd.to_numeric
            df["_chrono"] = parser(df[column], errors="coerce")
            df = df.sort_values("_chrono", na_position="last")
            break
    df = df.reset_index(drop=True)
    df["trial_order"] = np.arange(1, len(df) + 1)
    df["trial_id"] = df.get("trial_id", pd.Series([""] * len(df))).astype(str)
    return df


def plot_tuning_history(
    results: pd.DataFrame,
    model_name: str,
    metric: str = "f1_macro",
    headline: str | None = None,
) -> go.Figure:
    """Search metric against chronological trial order."""
    if results.empty:
        raise ValueError(f"No tuning results to plot for {model_name}.")
    if metric not in results.columns:
        fallback = "f1_micro"
        if fallback not in results.columns:
            raise KeyError(f"Neither {metric!r} nor {fallback!r} in results.")
        metric = fallback

    df = _chronological(results)
    colour = SERIES_COLORS.get(metric, SERIES_COLORS["default"])

    fig = go.Figure(
        go.Scatter(
            x=df["trial_order"],
            y=df[metric],
            mode="lines+markers",
            line=dict(width=2.2, color=colour),
            marker=dict(size=6, color=colour),
            customdata=np.column_stack([df["trial_id"]]),
            hovertemplate=(
                "Trial order: %{x}<br>Trial ID: %{customdata[0]}<br>"
                f"{metric}: %{{y:.4f}}<extra></extra>"
            ),
            name=model_name,
        )
    )
    return apply_theme(
        fig,
        headline or f"{metric.replace('_', '-').title()} Across Search Trials",
        f"{model_name} - {metric} per Ray Tune trial, in completion order",
        x_title="Chronological trial order",
        y_title=metric,
    )


def plot_cutoff_sweep(
    sweep: pd.DataFrame,
    model_name: str,
    chosen_cutoff: float | None = None,
    metrics: tuple[str, ...] = ("f1_micro", "f1_macro", "precision_micro", "recall_micro"),
    headline: str | None = None,
) -> go.Figure:
    """Every metric against probability cutoff, with the chosen point marked."""
    fig = go.Figure()
    for metric in metrics:
        if metric not in sweep.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=sweep["cutoff"],
                y=sweep[metric],
                mode="lines+markers",
                name=metric,
                line=dict(width=2.2, color=SERIES_COLORS.get(metric, SERIES_COLORS["default"])),
                marker=dict(size=5),
                hovertemplate=f"Cutoff: %{{x:.2f}}<br>{metric}: %{{y:.4f}}<extra></extra>",
            )
        )

    if chosen_cutoff is not None:
        fig.add_vline(
            x=chosen_cutoff,
            line=dict(color="#b8b8b8", width=1.4, dash="dash"),
            annotation_text=f"chosen: {chosen_cutoff:.2f}",
            annotation_position="top",
        )

    return apply_theme(
        fig,
        headline or "Operating Point Trades Precision Against Recall",
        f"{model_name} - metrics across probability cutoffs (cross-validated on training folds)",
        x_title="Probability cutoff",
        y_title="Score",
    )


def plot_per_label_performance(
    per_label: pd.DataFrame, model_name: str, top_n: int = 25
) -> go.Figure:
    """Per-condition F1 against support, for the most frequent conditions."""
    df = per_label.head(top_n)
    fig = go.Figure(
        go.Bar(
            x=df["label"],
            y=df["f1"],
            marker=dict(color=df["support"], colorscale="Blues", showscale=True,
                        colorbar=dict(title="Support")),
            hovertemplate=(
                "%{x}<br>F1: %{y:.3f}<br>Support: %{marker.color}<extra></extra>"
            ),
            name=model_name,
        )
    )
    fig.update_xaxes(tickangle=45)
    return apply_theme(
        fig,
        "Performance Concentrates in the Common Rhythms",
        f"{model_name} - per-condition F1 for the {top_n} most frequent labels",
        x_title=None,
        y_title="F1",
    )


def plot_explained_variance(
    cumulative: np.ndarray, n_components: int | None = None
) -> go.Figure:
    """Cumulative explained variance for the Incremental PCA basis."""
    components = np.arange(1, len(cumulative) + 1)
    fig = go.Figure(
        go.Scatter(
            x=components,
            y=cumulative,
            mode="lines",
            line=dict(width=2.4, color=SERIES_COLORS["default"]),
            hovertemplate="Components: %{x}<br>Cumulative variance: %{y:.4f}<extra></extra>",
            name="cumulative",
        )
    )
    if n_components:
        fig.add_vline(
            x=n_components,
            line=dict(color="#b8b8b8", width=1.4, dash="dash"),
            annotation_text=f"{n_components} retained",
        )
    return apply_theme(
        fig,
        "A Hundred Components Capture Most Waveform Variance",
        "Incremental PCA over all leads and all patients at 250 Hz",
        x_title="Number of components",
        y_title="Cumulative explained variance",
    )
