"""Plotly charts for the app. High-tech automotive telemetry styling,
modern interactive tooltips, rounded bars, and multi-dimensional analysis.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Categorical entity palette matching automotive intelligence theme
CLASS_COLORS = {
    "scratch": "#00d2ff",
    "dent": "#f59e0b",
    "crack": "#10b981",
    "broken_glass": "#8b5cf6",
    "major_body_damage": "#f43f5e",
}

# Sequential severity palette
SEVERITY_COLORS = {
    "minor": "#10b981",
    "moderate": "#f59e0b",
    "severe": "#f43f5e",
}

# Priority routing palette
PRIORITY_COLORS = {
    "Low": "#10b981",
    "Medium": "#00d2ff",
    "High": "#f59e0b",
    "Manual review required": "#f43f5e",
}
PRIORITY_ORDER = ["Low", "Medium", "High", "Manual review required"]

VEHICLE_COLORS = [
    "#00d2ff", "#3b82f6", "#10b981", "#f59e0b",
    "#ec4899", "#8b5cf6", "#14b8a6", "#f97316",
]

_GRID = "rgba(128, 128, 128, 0.14)"


def _base_layout(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    """Apply modern telemetry layout styling to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.01,
            y=0.96,
            font=dict(size=14, family="sans-serif"),
        ),
        height=height,
        margin=dict(l=14, r=14, t=48, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            y=-0.22,
            x=0.01,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        bargap=0.35,
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.92)",
            bordercolor="rgba(255, 255, 255, 0.2)",
            font=dict(family="sans-serif", size=12, color="#ffffff"),
        ),
        transition_duration=400,
    )
    fig.update_xaxes(showgrid=False, linecolor=_GRID, ticks="")
    fig.update_yaxes(gridcolor=_GRID, zeroline=False, ticks="")
    return fig


def _pretty(s: str) -> str:
    return s.replace("_", " ").capitalize()


def probability_chart(probs: dict, predicted: str) -> go.Figure:
    """Horizontal bars of class probabilities with rounded caps and highlighted winner."""
    items = sorted(probs.items(), key=lambda kv: kv[1])
    classes = [_pretty(k) for k, _ in items]
    vals = [v for _, v in items]
    raw_keys = [k for k, _ in items]

    colors = [CLASS_COLORS.get(k, "#0284c7") for k in raw_keys]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=classes,
            orientation="h",
            marker=dict(
                color=colors,
                cornerradius=6,
                line=dict(
                    color=["#ffffff" if k == predicted else "rgba(0,0,0,0)" for k in raw_keys],
                    width=[2 if k == predicted else 0 for k in raw_keys],
                ),
            ),
            text=[f"<b>{v:.1%}</b>" for v in vals],
            textposition="outside",
            cliponaxis=False,
            customdata=[f"Top Match: {c}" if k == predicted else f"Class: {c}" for c, k in zip(classes, raw_keys)],
            hovertemplate="<b>%{customdata}</b><br>Probability: <b>%{x:.1%}</b><extra></extra>",
        )
    )
    _base_layout(fig, "Neural Class Probability", height=290)
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%", showgrid=True, gridcolor=_GRID)
    fig.update_yaxes(showgrid=False)
    return fig


def damage_frequency_chart(freq: pd.DataFrame) -> go.Figure:
    """Modern bar chart of claims per damage class with volume and share %."""
    freq = freq.sort_values("claims", ascending=False)
    classes_pretty = [_pretty(c) for c in freq["damage_class"]]
    colors = [CLASS_COLORS.get(c, "#0284c7") for c in freq["damage_class"]]

    fig = go.Figure(
        go.Bar(
            x=classes_pretty,
            y=freq["claims"],
            marker=dict(
                color=colors,
                cornerradius=8,
            ),
            text=[f"<b>{c:,}</b> ({s:.1f}%)" for c, s in zip(freq["claims"], freq["share_pct"])],
            textposition="outside",
            cliponaxis=False,
            customdata=freq["share_pct"],
            hovertemplate="<b>%{x}</b><br>Claim Count: <b>%{y:,}</b><br>Cohort Share: <b>%{customdata:.1f}%</b><extra></extra>",
        )
    )
    _base_layout(fig, "Damage Type Distribution", height=350)
    fig.update_yaxes(title_text="Claims Count")
    return fig


def severity_chart(dist: pd.DataFrame) -> go.Figure:
    """Interactive severity breakdown chart with glow colors and percentages."""
    severities_pretty = [_pretty(s) for s in dist["severity"]]
    colors = [SEVERITY_COLORS.get(s, "#0284c7") for s in dist["severity"]]

    fig = go.Figure(
        go.Bar(
            x=severities_pretty,
            y=dist["claims"],
            marker=dict(
                color=colors,
                cornerradius=8,
            ),
            text=[f"<b>{c:,}</b> ({p:.1f}%)" for c, p in zip(dist["claims"], dist["share_pct"])],
            textposition="outside",
            cliponaxis=False,
            customdata=dist["share_pct"],
            hovertemplate="<b>%{x} Severity</b><br>Claims: <b>%{y:,}</b><br>Cohort Share: <b>%{customdata:.1f}%</b><extra></extra>",
        )
    )
    _base_layout(fig, "Severity Level Breakdown", height=350)
    fig.update_yaxes(title_text="Claims Count")
    return fig


def priority_chart(pri: pd.DataFrame) -> go.Figure:
    """Claims per processing priority with status-coded colors and percentages."""
    pri = pri.set_index("priority").reindex(PRIORITY_ORDER).fillna(0).reset_index()
    total_claims = max(int(pri["claims"].sum()), 1)
    shares = (pri["claims"] / total_claims * 100).round(1)

    colors = [PRIORITY_COLORS.get(p, "#8a8a85") for p in pri["priority"]]

    fig = go.Figure(
        go.Bar(
            x=pri["priority"],
            y=pri["claims"].astype(int),
            marker=dict(
                color=colors,
                cornerradius=8,
            ),
            text=[f"<b>{int(c):,}</b> ({s:.1f}%)" for c, s in zip(pri["claims"], shares)],
            textposition="outside",
            cliponaxis=False,
            customdata=shares,
            hovertemplate="<b>%{x}</b><br>Claims: <b>%{y:,}</b><br>Workflow Share: <b>%{customdata:.1f}%</b><extra></extra>",
        )
    )
    _base_layout(fig, "Claim Routing Priority", height=360)
    fig.update_yaxes(title_text="Claims Count")
    return fig


def cost_range_chart(costs: pd.DataFrame) -> go.Figure:
    """Grouped floating range bars of indicative repair costs (INR) per damage class and severity."""
    fig = go.Figure()
    for sev in ("minor", "moderate", "severe"):
        sub = costs[costs["severity"] == sev]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["mid"] = (sub["avg_low"] + sub["avg_high"]) / 2
        fig.add_trace(
            go.Bar(
                name=_pretty(sev),
                x=[_pretty(c) for c in sub["damage_class"]],
                y=sub["avg_high"] - sub["avg_low"],
                base=sub["avg_low"],
                marker=dict(
                    color=SEVERITY_COLORS[sev],
                    cornerradius=6,
                ),
                customdata=sub[["avg_low", "avg_high", "mid", "claims"]],
                hovertemplate=(
                    "<b>%{x} (" + _pretty(sev) + ")</b><br>"
                    "Range: <b>₹%{customdata[0]:,.0f} – ₹%{customdata[1]:,.0f}</b><br>"
                    "Midpoint: <b>₹%{customdata[2]:,.0f}</b><br>"
                    "Cohort Volume: <b>%{customdata[3]:,} claims</b><extra></extra>"
                ),
            )
        )
    _base_layout(fig, "Indicative Repair-Cost Range by Damage & Severity (INR)", height=380)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Estimated Cost (INR)", tickformat=",")
    return fig


def vehicle_mix_chart(v_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing claim volume and average exposure per vehicle segment."""
    v_df = v_df.sort_values("claims", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=v_df["vehicle_type"],
            x=v_df["claims"],
            orientation="h",
            marker=dict(
                color=VEHICLE_COLORS[: len(v_df)],
                cornerradius=6,
            ),
            text=[f"<b>{c:,}</b> ({s:.1f}%)" for c, s in zip(v_df["claims"], v_df["share_pct"])],
            textposition="outside",
            cliponaxis=False,
            customdata=v_df[["share_pct", "avg_exposure"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Claims: <b>%{x:,}</b> (%{customdata[0]:.1f}%)<br>"
                "Avg Indicative Exposure: <b>₹%{customdata[1]:,.0f}</b><extra></extra>"
            ),
        )
    )
    _base_layout(fig, "Vehicle Segment Distribution & Exposure", height=360)
    fig.update_xaxes(title_text="Claims Count", showgrid=True, gridcolor=_GRID)
    fig.update_yaxes(showgrid=False)
    return fig


def timeline_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    """Interactive time-series line & area chart showing claims intake velocity and exposure."""
    fig = go.Figure()

    # Claims count trace (left axis)
    fig.add_trace(
        go.Scatter(
            x=trend_df["period"],
            y=trend_df["claims"],
            mode="lines+markers",
            name="Weekly Claims",
            line=dict(color="#00d2ff", width=3, shape="spline"),
            marker=dict(size=6, color="#00d2ff", symbol="circle"),
            fill="tozeroy",
            fillcolor="rgba(0, 210, 255, 0.12)",
            hovertemplate="Week: <b>%{x|%b %d, %Y}</b><br>Intake: <b>%{y} claims</b><extra></extra>",
        )
    )

    # Exposure trace (secondary right axis)
    fig.add_trace(
        go.Scatter(
            x=trend_df["period"],
            y=trend_df["exposure"],
            mode="lines",
            name="Weekly Exposure (INR)",
            yaxis="y2",
            line=dict(color="#f59e0b", width=2, dash="dot", shape="spline"),
            hovertemplate="Exposure: <b>₹%{y:,.0f}</b><extra></extra>",
        )
    )

    _base_layout(fig, "Claims Intake Velocity & Exposure Timeline", height=360)
    fig.update_layout(
        yaxis=dict(title="Weekly Claims", showgrid=True, gridcolor=_GRID),
        yaxis2=dict(
            title="Exposure (INR)",
            overlaying="y",
            side="right",
            showgrid=False,
            tickformat=",",
        ),
    )
    return fig


def damage_sunburst_chart(df: pd.DataFrame) -> go.Figure:
    """Multi-tiered interactive Sunburst chart: Vehicle Type -> Damage Class -> Severity."""
    # Build hierarchical aggregation
    levels = df.groupby(["vehicle_type", "damage_class", "severity"]).size().reset_index(name="count")

    labels, parents, values = ["Total Fleet"], [""], [len(df)]

    # Tier 1: Vehicle Types
    v_counts = df["vehicle_type"].value_counts()
    for v, count in v_counts.items():
        labels.append(str(v))
        parents.append("Total Fleet")
        values.append(int(count))

    # Tier 2: Vehicle Type + Damage Class
    vd_counts = df.groupby(["vehicle_type", "damage_class"]).size().reset_index(name="count")
    for _, row in vd_counts.iterrows():
        v, d, count = row["vehicle_type"], row["damage_class"], row["count"]
        labels.append(f"{_pretty(d)} ({v})")
        parents.append(str(v))
        values.append(int(count))

    # Tier 3: Severity
    for _, row in levels.iterrows():
        v, d, s, count = row["vehicle_type"], row["damage_class"], row["severity"], row["count"]
        labels.append(f"{_pretty(s)} ({d[:4]}-{v[:3]})")
        parents.append(f"{_pretty(d)} ({v})")
        values.append(int(count))

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colorscale="Viridis",
            ),
            hovertemplate="<b>%{label}</b><br>Volume: <b>%{value:,} claims</b><br>Sub-cohort: <b>%{percentParent:.1%}</b><extra></extra>",
        )
    )
    _base_layout(fig, "Interactive Hierarchy: Vehicle Segment ➔ Damage ➔ Severity", height=420)
    fig.update_layout(margin=dict(t=48, l=10, r=10, b=10))
    return fig
