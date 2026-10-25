"""Plotly charts for the app. One axis per chart, fixed colour per entity,
thin marks, recessive grid, direct value labels, text in neutral ink.

Colours come from the validated reference palette (categorical slots 1-5 in
fixed order) and a single-hue blue ramp for ordered severity.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Categorical colour is bound to the ENTITY (damage class), never to its rank.
CLASS_COLORS = {
    "scratch": "#2a78d6",
    "dent": "#eb6834",
    "crack": "#1baf7a",
    "broken_glass": "#eda100",
    "major_body_damage": "#e87ba4",
}
# Sequential single hue, light -> dark = minor -> severe.
SEVERITY_COLORS = {"minor": "#a9c8f0", "moderate": "#5b9be0", "severe": "#1f5fb0"}
PRIORITY_COLORS = {
    "Low": "#a9c8f0",
    "Medium": "#5b9be0",
    "High": "#1f5fb0",
    "Manual review required": "#8a8a85",
}
PRIORITY_ORDER = ["Low", "Medium", "High", "Manual review required"]

_GRID = "rgba(128,128,128,0.18)"


def _base_layout(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.18, x=0),
        bargap=0.45,
    )
    fig.update_xaxes(showgrid=False, linecolor=_GRID, ticks="")
    fig.update_yaxes(gridcolor=_GRID, zeroline=False, ticks="")
    return fig


def _pretty(s: str) -> str:
    return s.replace("_", " ").capitalize()


def probability_chart(probs: dict, predicted: str) -> go.Figure:
    """Horizontal bars of class probabilities for one prediction."""
    items = sorted(probs.items(), key=lambda kv: kv[1])
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in items],
            y=[_pretty(k) for k, _ in items],
            orientation="h",
            marker_color=[CLASS_COLORS.get(k, "#8a8a85") for k, _ in items],
            marker_line_width=0,
            text=[f"{v:.0%}" for _, v in items],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    _base_layout(fig, "Class probabilities", height=280)
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", showgrid=True, gridcolor=_GRID)
    fig.update_yaxes(showgrid=False)
    return fig


def damage_frequency_chart(freq: pd.DataFrame) -> go.Figure:
    """Claims per damage class."""
    freq = freq.sort_values("claims", ascending=False)
    fig = go.Figure(
        go.Bar(
            x=[_pretty(c) for c in freq["damage_class"]],
            y=freq["claims"],
            marker_color=[CLASS_COLORS.get(c, "#8a8a85") for c in freq["damage_class"]],
            marker_line_width=0,
            text=freq["claims"],
            textposition="outside",
            cliponaxis=False,
            customdata=freq["share_pct"],
            hovertemplate="%{x}: %{y} claims (%{customdata}%)<extra></extra>",
        )
    )
    _base_layout(fig, "Damage type frequency")
    fig.update_yaxes(title_text="Claims")
    return fig


def severity_chart(dist: pd.DataFrame) -> go.Figure:
    """Claims per severity level (ordered)."""
    fig = go.Figure(
        go.Bar(
            x=[_pretty(s) for s in dist["severity"]],
            y=dist["claims"],
            marker_color=[SEVERITY_COLORS[s] for s in dist["severity"]],
            marker_line_width=0,
            text=[f"{c} ({p}%)" for c, p in zip(dist["claims"], dist["share_pct"])],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y} claims<extra></extra>",
        )
    )
    _base_layout(fig, "Severity distribution")
    fig.update_yaxes(title_text="Claims")
    return fig


def priority_chart(pri: pd.DataFrame) -> go.Figure:
    """Claims per processing priority (label + count, not colour alone)."""
    pri = pri.set_index("priority").reindex(PRIORITY_ORDER).fillna(0).reset_index()
    fig = go.Figure(
        go.Bar(
            x=pri["priority"],
            y=pri["claims"],
            marker_color=[PRIORITY_COLORS[p] for p in pri["priority"]],
            marker_line_width=0,
            text=pri["claims"].astype(int),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y} claims<extra></extra>",
        )
    )
    _base_layout(fig, "Claim-processing priority")
    fig.update_yaxes(title_text="Claims")
    return fig


def cost_range_chart(costs: pd.DataFrame) -> go.Figure:
    """Indicative repair-cost range (low to high, INR) per damage class and severity."""
    fig = go.Figure()
    for sev in ("minor", "moderate", "severe"):
        sub = costs[costs["severity"] == sev]
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                name=_pretty(sev),
                x=[_pretty(c) for c in sub["damage_class"]],
                y=sub["avg_high"] - sub["avg_low"],
                base=sub["avg_low"],
                marker_color=SEVERITY_COLORS[sev],
                marker_line_width=0,
                customdata=sub[["avg_low", "avg_high"]],
                hovertemplate="%{x} / " + sev + "<br>INR %{customdata[0]:,.0f} - %{customdata[1]:,.0f}<extra></extra>",
            )
        )
    _base_layout(fig, "Indicative repair-cost range by damage type (INR)", height=380)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="INR", tickformat=",")
    return fig
