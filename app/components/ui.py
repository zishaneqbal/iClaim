"""Shared Streamlit helpers: path setup, cached model loading."""

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
          --line: color-mix(in srgb, var(--text-color) 18%, transparent);
          --canvas: var(--background-color);
          --panel: var(--secondary-background-color);
          --blue: var(--primary-color);
          --cyan: color-mix(in srgb, var(--primary-color) 60%, #35c4d8);
          --mint: var(--green-color);
          --amber: var(--yellow-color);
          --rose: var(--red-color);
        }
        footer {visibility: hidden;}
        [data-testid="stAppViewContainer"] {
          background: radial-gradient(circle at 92% 2%, color-mix(in srgb, var(--cyan) 14%, transparent), transparent 23rem),
                      radial-gradient(circle at 4% 18%, color-mix(in srgb, var(--blue) 11%, transparent), transparent 28rem), var(--canvas);
        }
        /* Keep header visible so the menu, theme picker, and sidebar toggle work. */
        .block-container {padding-top: 1.75rem; padding-bottom: 3rem; max-width: 1440px;}
        [data-testid="stHeader"] {background: color-mix(in srgb, var(--background-color) 88%, transparent); backdrop-filter: blur(14px);}
        [data-testid="stSidebar"] {background: var(--background-color);}
        [data-testid="stSidebar"] * {color: var(--text-color);}
        [data-testid="stSidebar"] [data-testid="stPageLink"] a {border-radius: 10px; padding: .55rem .65rem;}
        [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {background: color-mix(in srgb, var(--primary-color) 12%, transparent);}
        [data-testid="stSidebar"] hr {border-color: var(--border-color);}
        .section-header {
          display: flex; align-items: center; gap: 0.75rem;
          margin: 2.1rem 0 0.9rem 0;
          animation: rise-in .48s ease both;
        }
        .section-header h2 {
          margin: 0; font-size: 1.5rem; color: var(--ink);
          font-weight: 800; letter-spacing: -0.5px;
        }
        .section-header .divider {flex: 1; height: 1px; background: linear-gradient(90deg, var(--line), transparent);}
        .hero-card {
          position: relative; overflow: hidden; padding: 1.8rem 2rem; border-radius: 22px;
          background: linear-gradient(115deg, #102143 0%, #174b96 56%, #137caa 100%);
          box-shadow: 0 20px 48px rgba(16,33,67,.25); color: #fff; margin-bottom: 1.25rem;
          animation: hero-enter .65s cubic-bezier(.2,.8,.2,1) both;
        }
        .hero-card:before, .hero-card:after {content:""; position:absolute; border-radius:999px; pointer-events:none;}
        .hero-card:before {width: 360px; height: 360px; right: -105px; top: -230px; background: rgba(64,224,208,.14);}
        .hero-card:after {width: 290px; height: 290px; right: 100px; bottom: -230px; border: 1px solid rgba(255,255,255,.15);}
        .hero-card h1 {margin:0; color:#fff; font-size:2rem; font-weight:800; letter-spacing:-.8px;}
        .hero-card p {margin:.35rem 0 0; color:rgba(235,246,255,.84); font-size:1rem;}
        .hero-icon {display:grid; place-items:center; width:58px; height:58px; border-radius:16px; background:rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.2); font-size:1.8rem;}
        .badge {
          display: inline-flex; align-items: center; gap: 0.4rem;
          padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.8rem;
          font-weight: 700; letter-spacing: 0.4px; text-transform: uppercase;
          white-space: nowrap;
        }
        .badge-severity {color: #fff;}
        .badge-priority {color: #fff;}
        .badge-model-ready {background: #17a778; color: #fff; box-shadow: 0 0 0 5px rgba(23,167,120,.13);}
        .badge-model-notready {background: #64748b; color: #fff;}

          padding: .9rem 1.1rem; margin: 1rem 0; font-size: .85rem;
          color: var(--ink); display: flex; align-items: flex-start; gap: .6rem;
        }
        [data-testid="stFileUploaderDropzone"] {
          border: 2px dashed color-mix(in srgb, var(--blue) 45%, var(--line)) !important; border-radius: 16px !important;
          background: color-mix(in srgb, var(--panel) 88%, var(--blue)) !important; padding: 2rem 1rem !important;
          transition: all .2s ease !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
          border-color: var(--blue) !important; background: color-mix(in srgb, var(--panel) 78%, var(--blue)) !important; transform: translateY(-2px);
        }
        .stButton > button:not([kind="primary"]),
        [data-testid="stDownloadButton"] > button {
          border: 1px solid var(--line) !important; color: var(--ink) !important;
          background: var(--panel) !important; border-radius: 10px !important;
        }
        .stButton > button:not([kind="primary"]):hover,
        [data-testid="stDownloadButton"] > button:hover {border-color: var(--blue) !important; background: color-mix(in srgb, var(--panel) 82%, var(--blue)) !important;}
        .stDataFrame {border: 1px solid var(--line) !important; border-radius: 14px !important; overflow: hidden;}
        [data-testid="stMetric"] {background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: .75rem 1rem;}
        [data-testid="stTabs"] [data-baseweb="tab-list"] {gap: .35rem;}
        [data-testid="stTabs"] button {border-radius: 9px 9px 0 0;}
        .stButton > button:focus-visible,
        [data-testid="stDownloadButton"] > button:focus-visible,
        [data-testid="stPageLink"] a:focus-visible,
        [data-testid="stFileUploaderDropzone"]:focus-within {
          outline: 3px solid color-mix(in srgb, var(--primary-color) 55%, transparent) !important;
          outline-offset: 2px !important;
        }
        @keyframes rise-in {from {opacity: 0; transform: translateY(12px);} to {opacity: 1; transform: translateY(0);}}
        @keyframes hero-enter {from {opacity:0; transform:translateY(18px) scale(.985);} to {opacity:1; transform:translateY(0) scale(1);}}
        @media (prefers-reduced-motion: reduce) {
          *, *:before, *:after {animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important;}
        }
        @media (max-width: 720px) {
          .block-container {padding: 1rem .9rem 2rem;}
          .hero-card {padding: 1.3rem;}
          .hero-card h1 {font-size: 1.55rem;}
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
          <span style="font-size:1.6rem">{icon_for(title)}</span>
          <h2>{title}</h2>
          <div class="divider"></div>
        </div>
        {f'<p style="margin:0 0 0.6rem 2.4rem; color:var(--muted); font-size:0.95rem;">{subtitle}</p>' if subtitle else ''}
        """,
        unsafe_allow_html=True,
    )


def icon_for(section: str) -> str:
    """Return an emoji icon for a section title."""
    mapping = {
        "Claim": "📷",
        "Dashboard": "📊",
        "Vehicle Damage Detection and Claim Analytics": "🚗",
        "Result": "🔍",
        "Claim analytics": "💰",
        "Claims": "📋",
    }
    return mapping.get(section, "▸")


def sidebar_nav(current_page: str = "Home") -> None:
    """Render a modern sidebar with app navigation and model status."""
    with st.sidebar:
        st.markdown(
            '<div style="padding: 0.75rem 0.5rem 1.25rem 0.5rem; margin-bottom: 1rem;">'
            '<span style="font-size: 2.2rem;">🚗</span> '
            '<span style="font-size: 1.15rem; font-weight: 800; color: var(--text-color); vertical-align: middle; margin-left: 0.5rem;">Vehicle Damage</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 0.75rem;">Navigation</div>',
            unsafe_allow_html=True,
        )
        # Native Streamlit page links (paths relative to the entrypoint dir 'app/')
        st.page_link("app.py", label="🏠 Home", icon="🏠", disabled=(current_page == "Home"))
        st.page_link("pages/1_Claim.py", label="📷 Claim", icon="📷", disabled=(current_page == "Claim"))
        st.page_link("pages/2_Dashboard.py", label="📊 Dashboard", icon="📊", disabled=(current_page == "Dashboard"))
        st.divider()
        st.markdown(
            '<div style="font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 0.5rem; margin-top: 1rem;">Model</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="padding: 0.6rem 0.85rem; border-radius: 10px; background: var(--secondary-background-color); border: 1px solid var(--border-color);">{model_status_badge()}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Decision support only. Final claims require human review.")


def metric_card(label: str, value: str, sub: str | None = None, color: str = "#1f5fb0") -> None:
    """Render a theme-aware KPI card using native Streamlit elements."""
    del color  # Retained for backwards-compatible callers; native theme controls color.
    with st.container(border=True):
        st.caption(label)
        st.subheader(value)
        if sub:
            st.caption(sub)


def severity_badge(level: str) -> str:
    """Return HTML for a colored severity badge."""
    color = {"minor": "#1baf7a", "moderate": "#eda100", "severe": "#e87ba4"}.get(level, "#64748b")
    return f'<span class="badge badge-severity" style="background:{color};">{level.capitalize()}</span>'


def priority_badge(priority: str) -> str:
    """Return HTML for a colored priority badge."""
    color = {
        "Low": "#1baf7a", "Medium": "#2a78d6", "High": "#eda100",
        "Manual review required": "#e87ba4",
    }.get(priority, "#64748b")
    return f'<span class="badge badge-priority" style="background:{color};">{priority}</span>'


def model_status_badge() -> str:
    """Return HTML for the model-availability badge."""
    ready = model_available(CFG)
    cls = "badge-model-ready" if ready else "badge-model-notready"
    text = "Model ready" if ready else "Not trained"
    return f'<span class="badge {cls}">{text}</span>'


def synthetic_notice() -> None:
    """Explain that sample claims are synthetic."""
    st.info(
        "🔖 Rows marked `synthetic` are generated demo data, not real insurer claims. "
        "Repair-cost ranges are indicative assumptions from config.yaml.",
        icon="ℹ️",
    )


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
