"""One chart theme, applied everywhere.

The notebook repeated ~40 lines of font/margin/colour arguments in every
figure. Here the styling is declared once and applied by :func:`apply_theme`.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

THEME: dict[str, Any] = {
    "font_family": "Helvetica Neue",
    "font_size_base": 18,
    "font_size_axis": 18,
    "font_size_subtitle": 18,
    "color_text": "#b8b8b8",
    "color_title": "white",
    "color_grid": "#1f1f1f",
    "color_background": "#000000",
    "color_line": "#636EFA",
    "margin_top": 170,
    "margin_right": 80,
    "margin_bottom": 80,
    "margin_left": 80,
    "height": 700,
}

#: Stable colour per metric, so the same series looks the same in every figure.
SERIES_COLORS = {
    "default": "#636EFA",
    "f1_micro": "#636EFA",
    "f1_macro": "rgb(20, 20, 204)",
    "precision_micro": "rgb(46, 74, 47)",
    "precision_macro": "rgb(129, 130, 130)",
    "recall_micro": "rgb(156, 126, 107)",
    "recall_macro": "rgb(230, 217, 195)",
    "val_logloss": "rgb(87, 94, 78)",
    "best_so_far_logloss": "rgb(20, 20, 204)",
}

STANDARD_LEGEND = dict(
    orientation="h", x=0, y=1.13, xanchor="left", yanchor="top", traceorder="normal"
)


def _title(headline: str, subtitle: str | None) -> dict:
    text = f"<span style='font-size:30px;font-weight:bold;'>    {headline}</span>"
    if subtitle:
        text += f"<br><span style='font-size:20px;font-weight:normal;'>      {subtitle}</span>"
    return dict(text=text, x=0, xanchor="left", pad=dict(t=50))


def apply_theme(
    fig: go.Figure,
    headline: str,
    subtitle: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    theme: dict[str, Any] | None = None,
) -> go.Figure:
    """Apply the shared styling and titles to a figure, in place."""
    style = {**THEME, **(theme or {})}
    font = dict(
        family=style["font_family"], size=style["font_size_axis"], color=style["color_text"]
    )
    axis = dict(
        title_font=font,
        tickfont=font,
        showgrid=True,
        gridcolor=style["color_grid"],
        gridwidth=0.2,
    )

    fig.update_xaxes(**({"title_text": x_title} if x_title else {}), **axis)
    fig.update_yaxes(**({"title_text": y_title} if y_title else {}), **axis)
    fig.update_layout(
        title=_title(headline, subtitle),
        paper_bgcolor=style["color_background"],
        plot_bgcolor=style["color_background"],
        font=dict(
            family=style["font_family"],
            size=style["font_size_base"],
            color=style["color_text"],
        ),
        hoverlabel=dict(font=font),
        legend=STANDARD_LEGEND,
        margin=dict(
            t=style["margin_top"],
            r=style["margin_right"],
            b=style["margin_bottom"],
            l=style["margin_left"],
        ),
        height=style["height"],
    )
    return fig
