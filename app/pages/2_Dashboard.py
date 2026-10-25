"""Management dashboard: KPIs, charts and the claims table."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import streamlit as st

from components import charts
from components.ui import CFG, metric_card, page_setup, section_header, synthetic_notice
from src import analytics
from src.utils import format_inr

page_setup("Dashboard", "📊")

section_header("Management Dashboard", "Claim-level predictions and business metrics for insurance operations")

claims = analytics.load_claims(CFG)
if claims.empty:
    st.info("No claims yet. Generate sample data with `python -m src.analytics --generate`.")
    st.stop()

if (claims["data_source"] == "synthetic").any():
    synthetic_notice()

# ---- Filters ---------------------------------------------------------------
with st.container(border=True):
    st.subheader("Filters")
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

# ---- KPI cards --------------------------------------------------------------
section_header("Key performance indicators", "Aggregate metrics for the filtered claims")
k = analytics.compute_kpis(df)
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card("Total claims", f"{k['total_claims']:,}")
with c2:
    metric_card("Severe claims", f"{k['pct_severe']}%", "of all claims")
with c3:
    metric_card("Avg. confidence", f"{k['avg_confidence']:.0%}", "model confidence")
with c4:
    metric_card("Manual review", f"{k['manual_review']:,}", "low-confidence claims")
with c5:
    metric_card("Est. exposure", format_inr(k["est_exposure_mid"]), "midpoint of indicative range")

if k["above_range"]:
    st.warning(f"⚠️ {k['above_range']} claim(s) have a claimed amount above the indicative range and may need review.")

# ---- Charts -----------------------------------------------------------------
section_header("Analytics charts", "Damage frequency, severity, cost ranges and processing priority")
row1a, row1b = st.columns(2)
with row1a:
    st.plotly_chart(charts.damage_frequency_chart(analytics.damage_frequency(df)), width="stretch")
with row1b:
    st.plotly_chart(charts.severity_chart(analytics.severity_distribution(df)), width="stretch")

row2a, row2b = st.columns([1.4, 1])
with row2a:
    st.plotly_chart(charts.cost_range_chart(analytics.cost_summary(df)), width="stretch")
with row2b:
    st.plotly_chart(charts.priority_chart(analytics.priority_breakdown(df)), width="stretch")

# ---- Table + export ---------------------------------------------------------
section_header("Claims register", "Filtered claim-level records")
table = df.sort_values("claim_date", ascending=False).copy()
table["claim_date"] = table["claim_date"].dt.date
st.dataframe(table, width="stretch", hide_index=True)
st.download_button(
    "Download filtered claims (CSV)",
    data=table.to_csv(index=False).encode("utf-8"),
    file_name="claims_filtered.csv",
    mime="text/csv",
)

with st.expander("Table view of chart data", expanded=False):
    st.write("Damage frequency")
    st.dataframe(analytics.damage_frequency(df), hide_index=True)
    st.write("Severity distribution")
    st.dataframe(analytics.severity_distribution(df), hide_index=True)
    st.write("Indicative cost summary (INR)")
    st.dataframe(analytics.cost_summary(df), hide_index=True)
