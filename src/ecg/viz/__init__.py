"""Plotly theming and the shared result charts."""

from ecg.viz.plots import (
    plot_cutoff_sweep,
    plot_explained_variance,
    plot_per_label_performance,
    plot_tuning_history,
)
from ecg.viz.theme import SERIES_COLORS, STANDARD_LEGEND, THEME, apply_theme

__all__ = [
    "SERIES_COLORS",
    "STANDARD_LEGEND",
    "THEME",
    "apply_theme",
    "plot_tuning_history",
    "plot_cutoff_sweep",
    "plot_per_label_performance",
    "plot_explained_variance",
]
