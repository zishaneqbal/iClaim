"""Shared Streamlit helpers: path setup, cached model loading, modern automotive UI styling."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.config import load_config  # noqa: E402
from src.predict import load_trained_model, model_available  # noqa: E402

CFG = load_config()


def inject_css() -> None:
    """Inject the shared visual language for the claims workspace."""
    st.markdown(
        """
        <style>
        :root {
          --ink: var(--text-color);
          --muted: color-mix(in srgb, var(--text-color) 65%, transparent);
          --line: color-mix(in srgb, var(--text-color) 16%, transparent);
          --canvas: var(--background-color);
          --panel: var(--secondary-background-color);
          --blue: var(--primary-color);
          --cyan: color-mix(in srgb, var(--primary-color) 55%, #00d2ff);
          --mint: var(--green-color, #10b981);
          --amber: var(--yellow-color, #f59e0b);
          --rose: var(--red-color, #f43f5e);
        }
        footer {visibility: hidden;}

        /* Modern subtle automotive telematics grid background */
        [data-testid="stAppViewContainer"] {
          background-color: var(--canvas);
          background-image:
            radial-gradient(circle at 92% 2%, color-mix(in srgb, var(--cyan) 13%, transparent), transparent 26rem),
            radial-gradient(circle at 4% 18%, color-mix(in srgb, var(--blue) 12%, transparent), transparent 30rem),
            linear-gradient(var(--line) 1px, transparent 1px),
            linear-gradient(90deg, var(--line) 1px, transparent 1px);
          background-size: 100% 100%, 100% 100%, 48px 48px, 48px 48px;
          background-position: 0 0, 0 0, -1px -1px, -1px -1px;
        }

        .block-container {
          padding-top: 1.6rem;
          padding-bottom: 3.2rem;
          max-width: 1440px;
        }

        [data-testid="stHeader"] {
          background: color-mix(in srgb, var(--background-color) 85%, transparent);
          backdrop-filter: blur(16px);
          border-bottom: 1px solid var(--line);
        }

        [data-testid="stSidebar"] {
          background: var(--background-color);
          border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * {color: var(--text-color);}
        [data-testid="stSidebar"] [data-testid="stPageLink"] a {
          border-radius: 12px;
          padding: 0.6rem 0.8rem;
          transition: all 0.22s ease;
          border: 1px solid transparent;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
          background: color-mix(in srgb, var(--primary-color) 12%, transparent);
          border-color: color-mix(in srgb, var(--primary-color) 25%, transparent);
          transform: translateX(3px);
        }

        /* Section header */
        .section-header {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          margin: 2.2rem 0 0.8rem 0;
          animation: rise-in .48s cubic-bezier(.16,1,.3,1) both;
        }
        .section-header h2 {
          margin: 0;
          font-size: 1.45rem;
          color: var(--ink);
          font-weight: 800;
          letter-spacing: -0.4px;
        }
        .section-header .divider {
          flex: 1;
          height: 1px;
          background: linear-gradient(90deg, var(--line), transparent);
        }

        /* Modern Hero Command Card */
        .hero-card {
          position: relative;
          overflow: hidden;
          padding: 2.2rem 2.4rem;
          border-radius: 24px;
          background: linear-gradient(125deg, #09152a 0%, #102648 45%, #0e4c75 82%, #0d6884 100%);
          border: 1px solid rgba(0, 210, 255, 0.28);
          box-shadow: 0 24px 50px rgba(9, 21, 42, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.16);
          color: #fff;
          margin-bottom: 1.5rem;
          animation: hero-enter .65s cubic-bezier(.2,.8,.2,1) both;
        }
        .hero-card::before {
          content: "";
          position: absolute;
          width: 440px;
          height: 440px;
          right: -100px;
          top: -210px;
          background: radial-gradient(circle, rgba(0, 210, 255, 0.22) 0%, transparent 70%);
          border-radius: 50%;
          pointer-events: none;
          animation: pulse-glow 6s ease-in-out infinite alternate;
        }
        .hero-card::after {
          content: "";
          position: absolute;
          width: 320px;
          height: 320px;
          right: 220px;
          bottom: -220px;
          background: radial-gradient(circle, rgba(23, 105, 224, 0.22) 0%, transparent 70%);
          border-radius: 50%;
          pointer-events: none;
        }
        .hero-card h1 {
          margin: 0;
          color: #ffffff;
          font-size: 2.1rem;
          font-weight: 800;
          letter-spacing: -0.8px;
          text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .hero-card p {
          margin: 0.4rem 0 0;
          color: rgba(225, 242, 255, 0.9);
          font-size: 1.02rem;
          max-width: 820px;
          line-height: 1.5;
        }
        .hero-icon {
          display: grid;
          place-items: center;
          width: 66px;
          height: 66px;
          border-radius: 18px;
          background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.06));
          border: 1px solid rgba(255, 255, 255, 0.3);
          font-size: 2.1rem;
          box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
          animation: float-slow 4s ease-in-out infinite;
        }
        .hero-chips-row {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 0.55rem;
          margin-top: 1.1rem;
        }
        .hero-chip {
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
          padding: 0.32rem 0.8rem;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.18);
          font-size: 0.78rem;
          font-weight: 600;
          color: rgba(240, 248, 255, 0.95);
          letter-spacing: 0.3px;
        }

        /* Pulsing indicator dots */
        .pulse-indicator {
          display: inline-block;
          width: 8px;
          height: 8px;
          border-radius: 50%;
          margin-right: 6px;
          position: relative;
        }
        .pulse-dot-mint {
          background: #10b981;
          animation: pulse-dot-green 1.8s infinite;
        }
        .pulse-dot-amber {
          background: #f59e0b;
          animation: pulse-dot-amber 1.8s infinite;
        }
        .pulse-dot-rose {
          background: #f43f5e;
          animation: pulse-dot-rose 1.8s infinite;
        }
        .pulse-dot-cyan {
          background: #00d2ff;
          animation: pulse-dot-cyan 1.8s infinite;
        }

        @keyframes pulse-dot-green {
          0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
          70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
          100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        @keyframes pulse-dot-amber {
          0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
          70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(245, 158, 11, 0); }
          100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
        }
        @keyframes pulse-dot-rose {
          0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.7); }
          70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(244, 63, 94, 0); }
          100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(244, 63, 94, 0); }
        }
        @keyframes pulse-dot-cyan {
          0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 210, 255, 0.7); }
          70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(0, 210, 255, 0); }
          100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 210, 255, 0); }
        }

        /* Modern Badges */
        .badge {
          display: inline-flex;
          align-items: center;
          gap: 0.45rem;
          padding: 0.4rem 0.9rem;
          border-radius: 999px;
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.4px;
          white-space: nowrap;
          transition: all 0.2s ease;
        }
        .badge-model-ready {
          background: rgba(16, 185, 129, 0.16);
          color: #10b981;
          border: 1px solid rgba(16, 185, 129, 0.35);
          box-shadow: 0 0 14px rgba(16, 185, 129, 0.15);
        }
        .badge-model-notready {
          background: rgba(100, 116, 139, 0.18);
          color: #94a3b8;
          border: 1px solid rgba(100, 116, 139, 0.3);
        }

        /* Modern Pro Metric Cards */
        .metric-card-pro {
          position: relative;
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 1.1rem 1.25rem;
          box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
          transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1);
          overflow: hidden;
          height: 100%;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          animation: rise-in .45s ease both;
        }
        .metric-card-pro::before {
          content: "";
          position: absolute;
          top: 0; left: 0; right: 0;
          height: 3px;
          background: linear-gradient(90deg, var(--card-accent, var(--blue)), transparent);
          border-radius: 16px 16px 0 0;
          opacity: 0.9;
        }
        .metric-card-pro:hover {
          transform: translateY(-4px);
          border-color: color-mix(in srgb, var(--card-accent, var(--blue)) 45%, var(--line));
          box-shadow: 0 12px 28px rgba(0, 0, 0, 0.12), 0 0 18px color-mix(in srgb, var(--card-accent, var(--blue)) 18%, transparent);
        }
        .metric-card-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 0.35rem;
        }
        .metric-label {
          font-size: 0.76rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.75px;
          color: var(--muted);
        }
        .metric-icon-badge {
          display: inline-grid;
          place-items: center;
          width: 32px;
          height: 32px;
          border-radius: 10px;
          background: color-mix(in srgb, var(--card-accent, var(--blue)) 14%, var(--panel));
          border: 1px solid color-mix(in srgb, var(--card-accent, var(--blue)) 30%, transparent);
          font-size: 1.05rem;
        }
        .metric-val {
          font-size: 1.7rem;
          font-weight: 800;
          letter-spacing: -0.5px;
          line-height: 1.2;
          margin: 0.2rem 0 0.35rem 0;
        }
        .metric-sub {
          font-size: 0.8rem;
          color: var(--muted);
          line-height: 1.35;
        }

        /* Interactive Pipeline Cards */
        .pipeline-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 1.3rem 1.2rem;
          position: relative;
          overflow: hidden;
          transition: all 0.28s cubic-bezier(.16,1,.3,1);
          height: 100%;
          display: flex;
          flex-direction: column;
        }
        .pipeline-card:hover {
          transform: translateY(-5px);
          border-color: color-mix(in srgb, var(--blue) 50%, var(--line));
          box-shadow: 0 14px 30px rgba(0, 0, 0, 0.1), 0 0 20px color-mix(in srgb, var(--cyan) 18%, transparent);
        }
        .pipeline-card-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 0.9rem;
        }
        .pipeline-step-badge {
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 1px;
          padding: 0.25rem 0.65rem;
          border-radius: 999px;
          background: color-mix(in srgb, var(--blue) 15%, var(--panel));
          color: var(--blue);
          border: 1px solid color-mix(in srgb, var(--blue) 30%, transparent);
        }
        .pipeline-icon {
          font-size: 1.7rem;
        }
        .pipeline-card h3 {
          margin: 0 0 0.4rem 0;
          font-size: 1.05rem;
          font-weight: 700;
          color: var(--ink);
        }
        .pipeline-card p {
          margin: 0;
          font-size: 0.85rem;
          color: var(--muted);
          line-height: 1.45;
        }

        /* Damage Class Explorer Cards */
        .class-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 1.15rem 1.2rem;
          transition: all 0.25s ease;
          position: relative;
        }
        .class-card:hover {
          transform: translateY(-3px);
          border-color: color-mix(in srgb, var(--cyan) 45%, var(--line));
          box-shadow: 0 10px 24px rgba(0,0,0,0.08);
        }
        .class-card-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 0.5rem;
        }
        .class-card-title {
          font-weight: 700;
          font-size: 1rem;
          color: var(--ink);
        }
        .class-card-desc {
          font-size: 0.83rem;
          color: var(--muted);
          line-height: 1.4;
          margin: 0;
        }

        /* HUD Vision Inspection Framing */
        .hud-card {
          position: relative;
          background: color-mix(in srgb, var(--panel) 92%, var(--blue));
          border: 1px solid color-mix(in srgb, var(--cyan) 35%, var(--line));
          border-radius: 16px;
          padding: 0.9rem;
          margin-bottom: 0.8rem;
          overflow: hidden;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        }
        .hud-top-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding-bottom: 0.45rem;
          margin-bottom: 0.6rem;
          border-bottom: 1px dashed color-mix(in srgb, var(--cyan) 30%, var(--line));
          font-family: monospace;
        }
        .hud-tag {
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.8px;
          color: var(--cyan);
          display: inline-flex;
          align-items: center;
        }
        .hud-title {
          font-size: 0.82rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          color: var(--ink);
        }
        .hud-viewport {
          position: relative;
          border-radius: 12px;
          overflow: hidden;
          border: 1px solid color-mix(in srgb, var(--cyan) 25%, transparent);
        }
        .hud-viewport::before, .hud-viewport::after {
          content: "";
          position: absolute;
          width: 18px;
          height: 18px;
          pointer-events: none;
          z-index: 2;
        }
        .hud-viewport::before {
          top: 6px; left: 6px;
          border-top: 2px solid var(--cyan);
          border-left: 2px solid var(--cyan);
        }
        .hud-viewport::after {
          bottom: 6px; right: 6px;
          border-bottom: 2px solid var(--cyan);
          border-right: 2px solid var(--cyan);
        }
        .hud-scanline {
          position: absolute;
          left: 0; right: 0;
          height: 2px;
          background: linear-gradient(90deg, transparent, var(--cyan), transparent);
          box-shadow: 0 0 10px var(--cyan);
          pointer-events: none;
          z-index: 3;
          animation: scanline-sweep 3s ease-in-out infinite alternate;
        }

        /* Modern styled uploader */
        [data-testid="stFileUploaderDropzone"] {
          border: 2px dashed color-mix(in srgb, var(--cyan) 45%, var(--line)) !important;
          border-radius: 18px !important;
          background: color-mix(in srgb, var(--panel) 92%, var(--blue)) !important;
          padding: 2.2rem 1.2rem !important;
          transition: all 0.25s ease !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
          border-color: var(--cyan) !important;
          background: color-mix(in srgb, var(--panel) 82%, var(--blue)) !important;
          transform: translateY(-2px);
          box-shadow: 0 12px 28px rgba(0, 210, 255, 0.12);
        }

        /* Primary action button glow */
        .stButton > button[kind="primary"] {
          background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
          border: 1px solid rgba(255,255,255,0.2) !important;
          box-shadow: 0 6px 20px rgba(2, 132, 199, 0.35) !important;
          border-radius: 12px !important;
          font-weight: 700 !important;
          letter-spacing: 0.3px !important;
          transition: all 0.22s ease !important;
        }
        .stButton > button[kind="primary"]:hover {
          transform: translateY(-2px) !important;
          box-shadow: 0 10px 26px rgba(2, 132, 199, 0.5) !important;
          filter: brightness(1.1);
        }

        .stButton > button:not([kind="primary"]),
        [data-testid="stDownloadButton"] > button {
          border: 1px solid var(--line) !important;
          color: var(--ink) !important;
          background: var(--panel) !important;
          border-radius: 12px !important;
          transition: all 0.2s ease !important;
        }
        .stButton > button:not([kind="primary"]):hover,
        [data-testid="stDownloadButton"] > button:hover {
          border-color: var(--blue) !important;
          background: color-mix(in srgb, var(--panel) 82%, var(--blue)) !important;
          transform: translateY(-1px);
        }

        .stDataFrame {
          border: 1px solid var(--line) !important;
          border-radius: 16px !important;
          overflow: hidden;
        }

        /* Animations */
        @keyframes rise-in {
          from {opacity: 0; transform: translateY(12px);}
          to {opacity: 1; transform: translateY(0);}
        }
        @keyframes hero-enter {
          from {opacity: 0; transform: translateY(18px) scale(0.985);}
          to {opacity: 1; transform: translateY(0) scale(1);}
        }
        @keyframes float-slow {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-7px); }
        }
        @keyframes pulse-glow {
          0% { opacity: 0.5; transform: scale(0.96); }
          100% { opacity: 0.85; transform: scale(1.04); }
        }
        @keyframes scanline-sweep {
          0% { top: 6%; opacity: 0.2; }
          50% { opacity: 0.9; }
          100% { top: 92%; opacity: 0.2; }
        }

        @media (prefers-reduced-motion: reduce) {
          *, *:before, *:after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
          }
        }
        @media (max-width: 768px) {
          .block-container {padding: 1rem 0.9rem 2rem;}
          .hero-card {padding: 1.4rem;}
          .hero-card h1 {font-size: 1.6rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_setup(title: str, icon: str = "🚗") -> None:
    """Set page config (must be the first Streamlit call on each page)."""
    st.set_page_config(page_title=title, page_icon=icon, layout="wide", initial_sidebar_state="expanded")
    inject_css()


def section_header(title: str, subtitle: str | None = None) -> None:
    """Render a modern section header with icon and divider."""
    st.markdown(
        f"""
        <div class="section-header">
          <span style="font-size:1.55rem">{icon_for(title)}</span>
          <h2>{title}</h2>
          <div class="divider"></div>
        </div>
        {f'<p style="margin:0 0 0.8rem 2.3rem; color:var(--muted); font-size:0.92rem;">{subtitle}</p>' if subtitle else ''}
        """,
        unsafe_allow_html=True,
    )


def icon_for(section: str) -> str:
    """Return an emoji icon for a section title."""
    mapping = {
        "Claim": "📸",
        "Dashboard": "📊",
        "Management Dashboard": "📊",
        "System overview": "⚡",
        "Result": "🔍",
        "Claim analytics": "💰",
        "Claims": "📋",
        "Claims register": "📋",
        "Key performance indicators": "📈",
        "Analytics charts": "📉",
        "Inspection pipeline": "🔄",
        "Damage taxonomy": "🏷️",
    }
    return mapping.get(section, "▸")


def sidebar_nav(current_page: str = "Home") -> None:
    """Render a modern sidebar with app navigation and model status."""
    with st.sidebar:
        st.markdown(
            '<div style="padding: 0.6rem 0.4rem 1rem 0.4rem; display: flex; align-items: center; gap: 0.75rem;">'
            '<span style="font-size: 2rem;">🚗</span> '
            '<div>'
            '<div style="font-size: 1.15rem; font-weight: 800; color: var(--text-color); line-height: 1.1;">iClaim</div>'
            '<div style="font-size: 0.75rem; color: var(--muted);">Vehicle Damage AI</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 0.6rem;">Navigation</div>',
            unsafe_allow_html=True,
        )
        st.page_link("app.py", label="🏠 Home", disabled=(current_page == "Home"))
        st.page_link("pages/1_Claim.py", label="📸 Claim Assessment", disabled=(current_page == "Claim"))
        st.page_link("pages/2_Dashboard.py", label="📊 Operations Dashboard", disabled=(current_page == "Dashboard"))
        st.divider()
        st.markdown(
            '<div style="font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 0.5rem;">AI Telemetry</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="padding: 0.6rem 0.8rem; border-radius: 12px; background: var(--secondary-background-color); border: 1px solid var(--border-color);">{model_status_badge()}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Decision-support only. Final claim approval requires human inspector review.")


def metric_card(
    label: str,
    value: str,
    sub: str | None = None,
    color: str | None = None,
    icon: str | None = None,
) -> None:
    """Render a modern automotive AI KPI card with animated hover effect."""
    l_lower = label.lower()
    if icon is None:
        if "damage class" in l_lower or "damage type" in l_lower:
            icon = "🚗"
        elif "severity" in l_lower:
            icon = "⚖️"
        elif "confidence" in l_lower:
            icon = "🎯"
        elif "priority" in l_lower:
            icon = "⚡"
        elif "total claims" in l_lower:
            icon = "📋"
        elif "severe" in l_lower:
            icon = "🚨"
        elif "manual review" in l_lower:
            icon = "🔍"
        elif "exposure" in l_lower or "cost" in l_lower:
            icon = "💰"
        elif "model" in l_lower:
            icon = "🧠"
        elif "backbone" in l_lower:
            icon = "⚙️"
        else:
            icon = "📊"

    if color is None:
        if "severe" in l_lower or "manual review" in l_lower or "not trained" in str(value).lower():
            color = "#f43f5e"
        elif "confidence" in l_lower or "ready" in str(value).lower():
            color = "#10b981"
        elif "severity" in l_lower or "priority" in l_lower:
            color = "#00d2ff"
        else:
            color = "#0284c7"

    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""

    st.markdown(
        f"""
        <div class="metric-card-pro" style="--card-accent: {color};">
          <div class="metric-card-top">
            <span class="metric-label">{label}</span>
            <span class="metric-icon-badge">{icon}</span>
          </div>
          <div class="metric-val" style="color: {color};">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_badge(level: str) -> str:
    """Return HTML for a styled severity badge with animated glowing indicator."""
    lvl = level.lower()
    config_map = {
        "minor": ("#10b981", "pulse-dot-mint", "Minor"),
        "moderate": ("#f59e0b", "pulse-dot-amber", "Moderate"),
        "severe": ("#f43f5e", "pulse-dot-rose", "Severe"),
    }
    color, dot, display_text = config_map.get(lvl, ("#64748b", "pulse-dot-amber", level.capitalize()))
    return (
        f'<span class="badge" style="background:color-mix(in srgb, {color} 16%, var(--panel)); '
        f'color:{color}; border:1px solid color-mix(in srgb, {color} 40%, transparent);">'
        f'<span class="pulse-indicator {dot}"></span>{display_text}'
        f'</span>'
    )


def priority_badge(priority: str) -> str:
    """Return HTML for a styled claim priority badge with glowing indicator."""
    p_lower = priority.lower()
    if "low" in p_lower:
        color, dot, icon = "#10b981", "pulse-dot-mint", "🟢"
    elif "medium" in p_lower:
        color, dot, icon = "#00d2ff", "pulse-dot-cyan", "🔵"
    elif "high" in p_lower:
        color, dot, icon = "#f59e0b", "pulse-dot-amber", "🟠"
    else:  # manual review
        color, dot, icon = "#f43f5e", "pulse-dot-rose", "⚠️"

    return (
        f'<span class="badge" style="background:color-mix(in srgb, {color} 16%, var(--panel)); '
        f'color:{color}; border:1px solid color-mix(in srgb, {color} 40%, transparent);">'
        f'<span class="pulse-indicator {dot}"></span>{icon} {priority}'
        f'</span>'
    )


def model_status_badge() -> str:
    """Return HTML for the model-availability badge with animated pulsing indicator."""
    ready = model_available(CFG)
    dot_cls = "pulse-dot-mint" if ready else "pulse-dot-rose"
    cls = "badge-model-ready" if ready else "badge-model-notready"
    text = "Vision Model Active" if ready else "Model Not Trained"
    return (
        f'<span class="badge {cls}">'
        f'<span class="pulse-indicator {dot_cls}"></span>'
        f'{text}'
        f'</span>'
    )


def synthetic_notice() -> None:
    """Explain that sample claims are synthetic."""
    st.info(
        "🔖 Rows marked `synthetic` are generated demo records for operational simulation. "
        "Repair-cost ranges are indicative actuarial assumptions from config.yaml.",
        icon="ℹ️",
    )


def hud_image_frame(title: str, telemetry: str = "CAM_INSPECTION // ACTIVE") -> str:
    """Return HTML header banner for automotive vision inspection HUD viewport."""
    return f"""
    <div class="hud-top-bar">
      <span class="hud-tag"><span class="pulse-indicator pulse-dot-cyan"></span>{telemetry}</span>
      <span class="hud-title">{title}</span>
    </div>
    """


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    """Load the trained model once per server process."""
    return load_trained_model(cfg=CFG)


def require_model() -> bool:
    """Show setup help and return False if no trained model exists."""
    if model_available(CFG):
        return True
    st.error("No trained model found at `models/best_model.keras`.")
    st.markdown(
        "Train one first:\n\n"
        "```bash\n"
        "python -m src.data_loader --prepare\n"
        "python -m src.train --config config.yaml\n"
        "```\n"
        "See `docs/DATASET_GUIDE.md` for the dataset download steps."
    )
    return False


def label(text: str) -> str:
    """'major_body_damage' -> 'Major body damage'."""
    return text.replace("_", " ").capitalize()
