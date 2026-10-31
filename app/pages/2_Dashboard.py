"""Management dashboard: KPIs, modern interactive charts and the claims table."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import importlib
import pandas as pd
import streamlit as st

from components import charts
from components.ui import CFG, metric_card, page_setup, section_header, synthetic_notice
from src import analytics

# Reload in case Streamlit kept an older in-memory version in sys.modules
try:
    importlib.reload(analytics)
    importlib.reload(charts)
except Exception:
    pass
from src.utils import format_inr


def _get_vehicle_distribution(d: pd.DataFrame) -> pd.DataFrame:
    """Safely get vehicle distribution, with inline fallback if module was stale."""
    if hasattr(analytics, "vehicle_distribution"):
        return analytics.vehicle_distribution(d)
    if d.empty or "vehicle_type" not in d:
        return pd.DataFrame(columns=["vehicle_type", "claims", "share_pct", "avg_exposure"])
    temp = d.copy()
    temp["mid_cost"] = (temp["est_cost_low"] + temp["est_cost_high"]) / 2
    g = temp.groupby("vehicle_type").agg(
        claims=("claim_id", "count"),
        avg_exposure=("mid_cost", "mean"),
    ).reset_index()
    g["share_pct"] = (g["claims"] / g["claims"].sum() * 100).round(1)
    g["avg_exposure"] = g["avg_exposure"].round(0)
    return g.sort_values("claims", ascending=False).reset_index(drop=True)


def _get_timeline_trends(d: pd.DataFrame) -> pd.DataFrame:
    """Safely get timeline trends, with inline fallback if module was stale."""
    if hasattr(analytics, "timeline_trends"):
        return analytics.timeline_trends(d)
    if d.empty or "claim_date" not in d:
        return pd.DataFrame(columns=["period", "claims", "exposure"])
    temp = d.copy()
    temp["date"] = pd.to_datetime(temp["claim_date"], errors="coerce")
    temp = temp.dropna(subset=["date"])
    if temp.empty:
        return pd.DataFrame(columns=["period", "claims", "exposure"])
    temp["period"] = temp["date"].dt.to_period("W").dt.start_time
    temp["mid_cost"] = (temp["est_cost_low"] + temp["est_cost_high"]) / 2
    return temp.groupby("period").agg(
        claims=("claim_id", "count"),
        exposure=("mid_cost", "sum"),
    ).reset_index().sort_values("period")

page_setup("Operations Dashboard", "📊")

section_header("Management Dashboard", "Claim-level telemetry, portfolio severity, and actuarial cost analytics")

claims = analytics.load_claims(CFG)
if claims.empty:
    st.info("No claims yet. Generate sample data with `python -m src.analytics --generate`.")
    st.stop()

if (claims["data_source"] == "synthetic").any():
    synthetic_notice()

# ---- Filters ---------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem;">
          <h3 style="margin:0; font-size:1.05rem; font-weight:700;">🔍 Telemetry & Claim Filters</h3>
          <span style="font-size:0.8rem; color:var(--muted);">Filter by date range, vehicle segment, or damage profile</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    f1, f2, f3, f4, f5 = st.columns(5)
    dmin, dmax = claims["claim_date"].min().date(), claims["claim_date"].max().date()
    date_range = f1.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    vehicles = f2.multiselect("Vehicle type", sorted(claims["vehicle_type"].dropna().unique()))
    classes = f3.multiselect("Damage type", sorted(claims["damage_class"].dropna().unique()))
    severities = f4.multiselect("Severity", analytics.SEVERITY_ORDER)
    sources = f5.multiselect("Data source", sorted(claims["data_source"].dropna().unique()))

df = claims.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df = df[(df["claim_date"] >= start) & (df["claim_date"] <= end)]
if vehicles:
    df = df[df["vehicle_type"].isin(vehicles)]
if classes:
    df = df[df["damage_class"].isin(classes)]
if severities:
    df = df[df["severity"].isin(severities)]
if sources:
    df = df[df["data_source"].isin(sources)]

if df.empty:
    st.warning("No claims match the selected filters.")
    st.stop()

# Filter count telemetry banner
st.markdown(
    f"""
    <div style="background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:0.6rem 1rem; margin:1rem 0; display:flex; align-items:center; justify-content:space-between;">
      <span style="font-size:0.85rem; color:var(--muted);">
        Active cohort: <strong>{len(df):,}</strong> of <strong>{len(claims):,}</strong> total claims in portfolio
      </span>
      <span class="badge" style="background:rgba(0,210,255,0.12); color:#00d2ff; border:1px solid rgba(0,210,255,0.25); font-size:0.72rem;">
        Active Filter Subset
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- KPI cards --------------------------------------------------------------
section_header("Key performance indicators", "Aggregate metrics and risk exposure for active claim cohort")
k = analytics.compute_kpis(df)
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card("Total claims", f"{k['total_claims']:,}", sub="Active filtered volume", color="#0284c7", icon="📋")
with c2:
    metric_card("Severe claims", f"{k['pct_severe']}%", sub="Of cohort volume", color="#f43f5e", icon="🚨")
with c3:
    metric_card("Avg. confidence", f"{k['avg_confidence']:.0%}", sub="Model confidence mean", color="#10b981", icon="🎯")
with c4:
    metric_card("Manual review", f"{k['manual_review']:,}", sub="Low-confidence queue", color="#f59e0b", icon="🔍")
with c5:
    metric_card("Est. exposure", format_inr(k["est_exposure_mid"]), sub="Midpoint indicative sum", color="#00d2ff", icon="💰")

if k["above_range"]:
    st.warning(f"⚠️ {k['above_range']} claim(s) have a claimed amount exceeding the indicative repair range and are flagged for review.")

# ---- Modern Interactive Charts with Tabs ------------------------------------
section_header("Analytics charts", "Interactive damage frequencies, vehicle segments, repair cost ranges, and timeline telemetry")

tab_overview, tab_costs, tab_timeline, tab_hierarchy = st.tabs([
    "📊 Overview & Vehicle Mix",
    "💰 Cost & Exposure Ranges",
    "📈 Intake Velocity & Timeline",
    "🔬 Interactive Hierarchy (Sunburst)",
])

with tab_overview:
    r1a, r1b = st.columns(2)
    with r1a:
        with st.container(border=True):
            st.plotly_chart(
                charts.damage_frequency_chart(analytics.damage_frequency(df)),
                width="stretch",
            )
    with r1b:
        with st.container(border=True):
            st.plotly_chart(
                charts.severity_chart(analytics.severity_distribution(df)),
                width="stretch",
            )

    r2a, r2b = st.columns(2)
    with r2a:
        with st.container(border=True):
            st.plotly_chart(
                charts.vehicle_mix_chart(_get_vehicle_distribution(df)),
                width="stretch",
            )
    with r2b:
        with st.container(border=True):
            st.plotly_chart(
                charts.priority_chart(analytics.priority_breakdown(df)),
                width="stretch",
            )

with tab_costs:
    with st.container(border=True):
        st.plotly_chart(
            charts.cost_range_chart(analytics.cost_summary(df)),
            width="stretch",
        )
    st.caption("Floating bars depict the indicative repair-cost span from average low to high estimate across damage classes and severity levels.")

with tab_timeline:
    with st.container(border=True):
        st.plotly_chart(
            charts.timeline_trend_chart(_get_timeline_trends(df)),
            width="stretch",
        )
    st.caption("Blue filled curve indicates weekly claim intake volume (left axis); amber dotted line traces total weekly indicative financial exposure in INR (right axis).")

with tab_hierarchy:
    with st.container(border=True):
        st.plotly_chart(
            charts.damage_sunburst_chart(df),
            width="stretch",
        )
    st.caption("💡 Click any segment (Vehicle Type ➔ Damage Class ➔ Severity) to drill down interactively. Click the center circle to zoom back out.")

# ---- Table + export ---------------------------------------------------------
section_header("Claims register", "Filtered claim-level tabular records")
table = df.sort_values("claim_date", ascending=False).copy()
table["claim_date"] = table["claim_date"].dt.date
st.dataframe(table, width="stretch", hide_index=True)

st.download_button(
    "📥 Download Filtered Claims Register (CSV)",
    data=table.to_csv(index=False).encode("utf-8"),
    file_name="claims_filtered.csv",
    mime="text/csv",
    use_container_width=False,
)

with st.expander("📊 View Raw Numerical Data for Charts", expanded=False):
    t_c1, t_c2, t_c3 = st.columns(3)
    with t_c1:
        st.write("**Damage Frequency Distribution**")
        st.dataframe(analytics.damage_frequency(df), hide_index=True, width="stretch")
    with t_c2:
        st.write("**Severity Level Breakdown**")
        st.dataframe(analytics.severity_distribution(df), hide_index=True, width="stretch")
    with t_c3:
        st.write("**Vehicle Segment Breakdown**")
        st.dataframe(_get_vehicle_distribution(df), hide_index=True, width="stretch")
    st.write("**Indicative Cost Summary (INR)**")
    st.dataframe(analytics.cost_summary(df), hide_index=True, width="stretch")
