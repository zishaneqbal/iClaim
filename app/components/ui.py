"""Shared Streamlit helpers: path setup, cached model loading, modern automotive UI styling with SVG vector icons and advanced gradients."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.config import load_config
from src.predict import load_trained_model, model_available

CFG = load_config()

SVG_ICONS = {
    "car": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"/><circle cx="7" cy="17" r="2"/><path d="M9 17h6"/><circle cx="17" cy="17" r="2"/></svg>',
    "camera": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>',
    "dashboard": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 16v-4"/><path d="M12 16v-8"/><path d="M17 16v-2"/></svg>',
    "chart": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/></svg>',
    "brain": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04"/></svg>',
    "microscope": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="M6 18h8"/><path d="M3 22h18"/><path d="m14 22 3-3-3-3"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/><path d="m12 6 4-4 3 3-4 4"/></svg>',
    "shield_check": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>',
    "scales": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>',
    "tag": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z"/><circle cx="7" cy="7" r="1.5" fill="{c}"/></svg>',
    "cpu": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>',
    "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    "alert": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>',
    "currency": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="M6 3h12"/><path d="M6 8h12"/><path d="m6 13 8.5 8"/><path d="M6 13h3a4 4 0 0 0 0-8"/></svg>',
    "search": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
    "target": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "zap": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg>',
    "hammer": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="m15 12-8.5 8.5c-.83.83-2.17.83-3 0a2.12 2.12 0 0 1 0-3L12 9"/><path d="M17.64 15 22 10.64"/><path d="m20.91 3.26-6.36 6.36"/><path d="m9.64 6 4.25-4.24c.78-.78 2.05-.78 2.83 0l3.53 3.53c.78.78.78 2.05 0 2.83L16 12.36"/></svg>',
    "spark": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>',
    "glass": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><polygon points="6 2 18 2 18 6 6 6 6 2"/><path d="M14 18v3a1 1 0 0 1-1 1h-2a1 1 0 0 1-1-1v-3"/><path d="M6 6c0 4 3 7 6 7s6-3 6-7"/><path d="m10 6 2 4-2 2"/></svg>',
    "flame": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>',
    "clipboard": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/></svg>',
    "download": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>',
    "filter": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>',
    "arrow_right": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>',
    "home": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
    "check": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><polyline points="20 6 9 17 4 12"/></svg>',
    "check_circle": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "layers": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
}


def svg_icon(name: str, size: int = 18, color: str = "currentColor", class_name: str = "") -> str:
    """Return crisp inline SVG markup for a given icon name."""
    template = SVG_ICONS.get(name, SVG_ICONS["activity"])
    return template.format(s=size, c=color, cls=class_name)


def inject_css() -> None:
    """Inject custom CSS for modern automotive UI styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

        html, body, [class*="css"], .stApp, button, input, select, textarea {
          font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Plus Jakarta Sans", "Inter", -system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
          letter-spacing: -0.012em;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

        [data-testid="stSidebarNav"] {
          display: none !important;
        }

        :root {
          --ios-blur: blur(22px) saturate(180%);
          --ios-blur-sm: blur(14px) saturate(160%);

          --app-bg: #070d18;
          --app-text: #f8fafc;
          --app-muted: #94a3b8;
          --muted: #94a3b8;
          --panel: rgba(13, 23, 40, 0.75);
          --app-line: rgba(255, 255, 255, 0.1);
          --line: rgba(255, 255, 255, 0.1);
          --grid-line: rgba(255, 255, 255, 0.025);
          --grid-major: rgba(0, 210, 255, 0.07);

          --glass-bg: rgba(14, 24, 42, 0.72);
          --glass-card: rgba(13, 23, 40, 0.75);
          --glass-card-hover: rgba(18, 32, 56, 0.85);
          --glass-border: rgba(0, 210, 255, 0.22);
          --glass-border-hover: rgba(0, 210, 255, 0.75);
          --glass-specular: inset 0 1px 0 rgba(255, 255, 255, 0.16);
          --glass-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);

          --glow-hover: 0 0 28px rgba(0, 210, 255, 0.4), 0 16px 40px rgba(0, 0, 0, 0.4);
          --glow-active: 0 0 42px rgba(0, 210, 255, 0.85), inset 0 0 14px rgba(0, 210, 255, 0.3);

          --accent-cyan: #00d2ff;
          --accent-blue: #0284c7;
          --cyan: #00d2ff;
          --blue: #0284c7;
          --text-color: #f8fafc;
          --bg: #070d18;
          --btn-primary-bg: linear-gradient(135deg, rgba(0, 180, 219, 0.9) 0%, rgba(2, 132, 199, 0.95) 100%);
          --btn-primary-hover: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
          --btn-sec-bg: rgba(12, 24, 44, 0.72);
          --btn-sec-border: rgba(0, 210, 255, 0.25);
          --btn-sec-text: #f8fafc;
        }

        html[data-theme-mode="light"],
        html:has([data-base-theme="light"]),
        [data-base-theme="light"],
        [data-theme="light"],
        body[data-theme-mode="light"],
        .stApp[data-base-theme="light"],
        [data-testid="stAppViewContainer"][data-base-theme="light"] {
          --app-bg: #f6f8fc;
          --app-text: #0f172a;
          --app-muted: #475569;
          --muted: #475569;
          --panel: rgba(255, 255, 255, 0.86);
          --app-line: rgba(0, 0, 0, 0.08);
          --line: rgba(0, 0, 0, 0.08);
          --grid-line: transparent;
          --grid-major: transparent;

          --glass-bg: rgba(255, 255, 255, 0.78);
          --glass-card: rgba(255, 255, 255, 0.84);
          --glass-card-hover: rgba(255, 255, 255, 0.96);
          --glass-border: rgba(0, 122, 255, 0.18);
          --glass-border-hover: rgba(0, 122, 255, 0.55);
          --glass-specular: inset 0 1px 2px #ffffff, inset 0 -1px 0 rgba(0, 0, 0, 0.03);
          --glass-shadow: 0 10px 32px rgba(15, 23, 42, 0.06), 0 1px 3px rgba(15, 23, 42, 0.03);

          --glow-hover: 0 14px 38px rgba(0, 122, 255, 0.18), 0 2px 8px rgba(0, 122, 255, 0.1), inset 0 1px 2px #ffffff;
          --glow-active: 0 18px 46px rgba(0, 122, 255, 0.35), inset 0 0 12px rgba(0, 122, 255, 0.15);

          --accent-cyan: #0284c7;
          --accent-blue: #007aff;
          --cyan: #0284c7;
          --blue: #007aff;
          --text-color: #0f172a;
          --bg: #f6f8fc;
          --btn-primary-bg: linear-gradient(135deg, #007aff 0%, #0056d6 100%);
          --btn-primary-hover: linear-gradient(135deg, #1a8cff 0%, #0066ff 100%);
          --btn-sec-bg: rgba(255, 255, 255, 0.85);
          --btn-sec-border: rgba(0, 0, 0, 0.12);
          --btn-sec-text: #0f172a;
        }

        footer {visibility: hidden;}
        [data-testid="stAppViewContainer"] {
          scroll-timeline-name: --page-scroll;
          scroll-timeline-axis: block;
          background-color: var(--app-bg) !important;
          background-image:
            radial-gradient(ellipse 90% 45% at 50% -10%, rgba(2, 132, 199, 0.10), transparent 70%),
            radial-gradient(ellipse 60% 40% at 90% 40%, rgba(0, 210, 255, 0.05), transparent 60%),
            linear-gradient(to right, var(--grid-line) 1px, transparent 1px),
            linear-gradient(to bottom, var(--grid-line) 1px, transparent 1px),
            linear-gradient(to right, var(--grid-major) 1px, transparent 1px),
            linear-gradient(to bottom, var(--grid-major) 1px, transparent 1px);
          background-size:
            100% 100%, 100% 100%,
            28px 28px, 28px 28px,
            140px 140px, 140px 140px;
          background-position: 0 0, 0 0, 0 0, 0 0, 0 0, 0 0;
          background-attachment: fixed;
          transition: background-color 0.3s ease;
        }

        html[data-theme-mode="light"] [data-testid="stAppViewContainer"],
        html:has([data-base-theme="light"]) [data-testid="stAppViewContainer"],
        [data-base-theme="light"] [data-testid="stAppViewContainer"],
        .stApp[data-base-theme="light"] [data-testid="stAppViewContainer"] {
          background-color: #f6f8fc !important;
          background-image:
            radial-gradient(circle at 12% 15%, rgba(186, 230, 253, 0.55) 0%, transparent 45%),
            radial-gradient(circle at 88% 12%, rgba(224, 231, 255, 0.65) 0%, transparent 50%),
            radial-gradient(circle at 50% 88%, rgba(241, 245, 249, 0.85) 0%, transparent 55%) !important;
          background-size: 100% 100% !important;
        }

        .scroll-parallax-bg {
          position: fixed;
          top: -40px;
          left: 0;
          width: 100vw;
          height: 120vh;
          pointer-events: none;
          z-index: 0;
          opacity: 0.35;
          background-image:
            radial-gradient(circle 1.5px at 140px 140px, var(--accent-cyan) 100%, transparent 100%),
            linear-gradient(to right, var(--grid-major) 1px, transparent 1px),
            linear-gradient(to bottom, var(--grid-major) 1px, transparent 1px);
          background-size: 140px 140px, 140px 140px, 140px 140px;
          animation: scroll-parallax-shift auto linear both;
          animation-timeline: --page-scroll;
        }

        html[data-theme-mode="light"] .scroll-parallax-bg,
        html:has([data-base-theme="light"]) .scroll-parallax-bg,
        [data-base-theme="light"] .scroll-parallax-bg,
        .stApp[data-base-theme="light"] .scroll-parallax-bg {
          opacity: 0.3 !important;
          background-image:
            radial-gradient(circle 380px at 15% 25%, rgba(56, 189, 248, 0.28) 0%, transparent 70%),
            radial-gradient(circle 440px at 85% 70%, rgba(129, 140, 248, 0.22) 0%, transparent 70%) !important;
          background-size: 100% 100% !important;
        }

        @keyframes scroll-parallax-shift {
          0% { transform: translateY(0px); }
          100% { transform: translateY(-80px); }
        }

        .scroll-progress-bar {
          position: fixed;
          top: 0;
          left: 0;
          height: 3.5px;
          width: 0%;
          background: linear-gradient(90deg, #0284c7 0%, #00d2ff 60%, #38bdf8 100%);
          z-index: 999999;
          transition: width 0.06s linear;
        }

        .block-container {
          position: relative;
          z-index: 1;
          padding-top: 1.6rem;
          padding-bottom: 3.2rem;
          max-width: 1440px;
        }

        [data-testid="stHeader"], header {
          background: var(--glass-bg) !important;
          backdrop-filter: var(--ios-blur) !important;
          -webkit-backdrop-filter: var(--ios-blur) !important;
          border-bottom: 1px solid var(--glass-border) !important;
        }

        [data-testid="stSidebar"],
        section[data-testid="stSidebar"] {
          background: var(--glass-bg) !important;
          backdrop-filter: var(--ios-blur) !important;
          -webkit-backdrop-filter: var(--ios-blur) !important;
          border-right: 1px solid var(--glass-border) !important;
          overflow-x: hidden !important;
          transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1), margin-left 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }

        section[data-testid="stSidebar"][aria-expanded="true"],
        [data-testid="stSidebar"][aria-expanded="true"] {
          width: 285px !important;
          min-width: 285px !important;
          max-width: 285px !important;
          flex-basis: 285px !important;
          resize: none !important;
        }

        section[data-testid="stSidebar"][aria-expanded="false"],
        [data-testid="stSidebar"][aria-expanded="false"] {
          margin-left: -285px !important;
          min-width: 0 !important;
          max-width: 0 !important;
          width: 0 !important;
          border-right: none !important;
        }

        [data-testid="stSidebarResizeHandle"],
        div[data-testid="stSidebarResizeHandle"],
        div[class*="stSidebarResizeHandle"],
        [data-testid="stSidebarResizerHandle"],
        div[data-testid="stSidebarResizerRoot"],
        section[data-testid="stSidebar"] [data-testid="stSidebarResizeHandle"],
        section[data-testid="stSidebar"] ~ [data-testid="stSidebarResizeHandle"],
        section[data-testid="stSidebar"] ~ div[role="separator"],
        div[class*="resizerHandle"],
        div[class*="Resizer"],
        *[style*="cursor: col-resize"],
        *[style*="cursor: ew-resize"] {
          display: none !important;
          pointer-events: none !important;
          cursor: default !important;
          width: 0 !important;
          opacity: 0 !important;
        }

        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="collapsedControl"],
        [data-testid="collapsedControl"] button,
        [data-testid="stSidebarHeader"] button {
          display: flex !important;
          visibility: visible !important;
          pointer-events: auto !important;
          cursor: pointer !important;
          opacity: 1 !important;
          z-index: 1000 !important;
        }
        [data-testid="stSidebar"] * { color: var(--app-text) !important; }
        [data-testid="stSidebar"] [data-testid="stPageLink"] a {
          border-radius: 12px;
          padding: 0.6rem 0.8rem;
          transition: all 0.24s cubic-bezier(0.16, 1, 0.3, 1);
          border: 1px solid var(--app-line);
          background: var(--glass-bg);
          backdrop-filter: var(--ios-blur-sm);
        }
        [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
          background: var(--glass-card-hover) !important;
          border-color: var(--glass-border-hover) !important;
          box-shadow: var(--glow-hover) !important;
          transform: translateX(4px);
        }
        [data-testid="stSidebar"] [data-testid="stPageLink"] a:active {
          box-shadow: var(--glow-active) !important;
          transform: scale(0.98);
        }

        .section-header {
          display: flex;
          align-items: center;
          gap: 0.85rem;
          margin: 2.2rem 0 1rem 0;
          animation: rise-in .48s cubic-bezier(.16,1,.3,1) both;
        }
        .section-header-text {
          display: flex;
          flex-direction: column;
          gap: 0.15rem;
        }
        .section-header-icon {
          display: inline-grid;
          place-items: center;
          width: 36px;
          height: 36px;
          border-radius: 10px;
          background: rgba(0, 122, 255, 0.08);
          border: 1px solid rgba(0, 122, 255, 0.18);
          color: var(--accent-cyan);
          backdrop-filter: var(--ios-blur-sm);
          transition: all 0.2s ease;
          flex-shrink: 0;
        }
        .section-header:hover .section-header-icon {
          box-shadow: var(--glow-hover);
          border-color: var(--glass-border-hover);
        }
        .section-header h2 {
          margin: 0;
          font-size: 1.35rem;
          color: var(--app-text) !important;
          font-weight: 800;
          letter-spacing: -0.025em;
          line-height: 1.2;
        }
        .section-header-sub {
          margin: 0 !important;
          font-size: 0.84rem !important;
          color: var(--app-muted) !important;
          font-weight: 500 !important;
          letter-spacing: -0.01em !important;
          line-height: 1.35 !important;
        }
        .section-header .divider {
          flex: 1;
          height: 1px;
          background: linear-gradient(90deg, var(--glass-border), transparent);
          margin-left: 0.5rem;
        }

        .hero-card {
          position: relative;
          overflow: hidden;
          padding: 2.2rem 2.6rem;
          border-radius: 20px;
          background: linear-gradient(135deg,
            #061324 0%,
            #092444 38%,
            #0b3a62 72%,
            #0284c7 100%
          );
          backdrop-filter: var(--ios-blur);
          -webkit-backdrop-filter: var(--ios-blur);
          border: 1px solid rgba(0, 210, 255, 0.35);
          box-shadow:
            0 16px 42px rgba(0, 0, 0, 0.35),
            inset 0 1px 1px rgba(255, 255, 255, 0.25);
          color: #ffffff !important;
          margin-bottom: 1.6rem;
          animation: hero-enter .5s cubic-bezier(.2,.8,.2,1) both;
          transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .hero-card:hover {
          border-color: rgba(0, 210, 255, 0.75);
          box-shadow:
            0 0 35px rgba(0, 210, 255, 0.4),
            0 22px 52px rgba(0, 0, 0, 0.45),
            inset 0 1px 1px rgba(255, 255, 255, 0.4);
          transform: translateY(-2px);
        }
        .hero-card:active {
          box-shadow: 0 0 45px rgba(0, 210, 255, 0.7), inset 0 0 16px rgba(0, 210, 255, 0.3);
        }
        .hero-card h1 {
          margin: 0;
          font-size: 2.25rem;
          font-weight: 800;
          letter-spacing: -0.8px;
          background: linear-gradient(135deg, #ffffff 20%, #d8edfc 60%, #38bdf8 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
        .hero-card p {
          margin: 0.5rem 0 0;
          color: rgba(225, 240, 255, 0.95) !important;
          font-size: 1.02rem;
          max-width: 820px;
          line-height: 1.55;
          letter-spacing: -0.01em;
        }
        .hero-icon {
          display: grid;
          place-items: center;
          width: 58px;
          height: 58px;
          border-radius: 16px;
          background: rgba(0, 210, 255, 0.16);
          backdrop-filter: var(--ios-blur-sm);
          border: 1px solid rgba(0, 210, 255, 0.45);
          color: #00d2ff;
          transition: all 0.22s ease;
        }
        .hero-card:hover .hero-icon {
          box-shadow: 0 0 24px rgba(0, 210, 255, 0.55);
          border-color: #00d2ff;
        }
        .hero-chips-row {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 0.6rem;
          margin-top: 1.2rem;
        }
        .hero-chip {
          display: inline-flex;
          align-items: center;
          gap: 0.45rem;
          padding: 0.35rem 0.85rem;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.12);
          backdrop-filter: var(--ios-blur-sm);
          -webkit-backdrop-filter: var(--ios-blur-sm);
          border: 1px solid rgba(255, 255, 255, 0.24);
          font-size: 0.78rem;
          font-weight: 600;
          color: #ffffff !important;
          letter-spacing: 0.2px;
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.15);
          transition: all 0.22s ease;
          cursor: pointer;
        }
        .hero-chip:hover {
          background: rgba(0, 210, 255, 0.28);
          border-color: #00d2ff;
          box-shadow: 0 0 22px rgba(0, 210, 255, 0.55);
          transform: translateY(-2px);
        }
        .hero-chip:active {
          box-shadow: 0 0 32px rgba(0, 210, 255, 0.85);
          transform: scale(0.96);
        }

        [data-testid="stContainer"] {
          background: var(--glass-card) !important;
          backdrop-filter: var(--ios-blur) !important;
          -webkit-backdrop-filter: var(--ios-blur) !important;
          border: 1px solid var(--glass-border) !important;
          border-radius: 16px !important;
          box-shadow: var(--glass-shadow), var(--glass-specular) !important;
          transition: all 0.24s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        [data-testid="stContainer"]:hover {
          background: var(--glass-card-hover) !important;
          border-color: var(--glass-border-hover) !important;
          box-shadow: var(--glow-hover) !important;
          transform: translateY(-2.5px);
        }
        [data-testid="stContainer"]:active {
          box-shadow: var(--glow-active) !important;
          transform: scale(0.99);
        }

        .metric-card-pro {
          position: relative;
          background: var(--glass-card) !important;
          backdrop-filter: var(--ios-blur) !important;
          -webkit-backdrop-filter: var(--ios-blur) !important;
          border: 1px solid var(--glass-border) !important;
          border-radius: 16px;
          padding: 1.15rem 1.3rem;
          box-shadow: var(--glass-shadow), var(--glass-specular) !important;
          transition: all 0.24s cubic-bezier(0.16, 1, 0.3, 1);
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
          top: 0; left: 16px; right: 16px;
          height: 2px;
          background: linear-gradient(90deg, transparent, var(--card-accent, var(--accent-cyan)), transparent);
          border-radius: 999px;
          opacity: 0.8;
        }
        .metric-card-pro:hover {
          transform: translateY(-3px);
          background: var(--glass-card-hover) !important;
          border-color: var(--glass-border-hover) !important;
          box-shadow: var(--glow-hover) !important;
        }
        .metric-card-pro:active {
          transform: scale(0.98);
          box-shadow: var(--glow-active) !important;
        }
        .metric-card-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 0.4rem;
        }
        .metric-label {
          font-size: 0.72rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--app-muted) !important;
        }
        .metric-icon-badge {
          display: inline-grid;
          place-items: center;
          width: 32px;
          height: 32px;
          border-radius: 9px;
          background: rgba(0, 122, 255, 0.08);
          border: 1px solid rgba(0, 122, 255, 0.18);
          color: var(--card-accent, var(--accent-cyan));
          backdrop-filter: var(--ios-blur-sm);
          transition: all 0.2s ease;
        }
        .metric-card-pro:hover .metric-icon-badge {
          box-shadow: 0 0 14px var(--card-accent, var(--accent-cyan));
          border-color: var(--card-accent, var(--accent-cyan));
        }
        .metric-val {
          font-size: 1.85rem;
          font-weight: 800;
          letter-spacing: -0.03em;
          line-height: 1.15;
          margin: 0.25rem 0 0.35rem 0;
          font-feature-settings: "tnum" 1;
          font-variant-numeric: tabular-nums;
          color: var(--app-text) !important;
        }
        .metric-sub {
          font-size: 0.78rem;
          color: var(--app-muted) !important;
          line-height: 1.4;
          font-weight: 500;
          letter-spacing: -0.01em;
        }

        .pipeline-card {
          background: var(--glass-card) !important;
          backdrop-filter: var(--ios-blur) !important;
          -webkit-backdrop-filter: var(--ios-blur) !important;
          border: 1px solid var(--glass-border) !important;
          border-radius: 16px;
          padding: 1.35rem !important;
          min-height: 180px;
          box-shadow: var(--glass-shadow), var(--glass-specular) !important;
          transition: all 0.24s cubic-bezier(0.16, 1, 0.3, 1);
          display: flex;
          flex-direction: column;
          justify-content: flex-start;
        }
        .pipeline-card:hover {
          transform: translateY(-3px);
          background: var(--glass-card-hover) !important;
          border-color: var(--glass-border-hover) !important;
          box-shadow: var(--glow-hover) !important;
        }
        .pipeline-card:active {
          transform: scale(0.98);
          box-shadow: var(--glow-active) !important;
        }
        .pipeline-card-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 0.75rem;
        }
        .pipeline-card h3 {
          color: var(--app-text) !important;
          font-size: 1.05rem;
          font-weight: 700;
          letter-spacing: -0.02em;
          line-height: 1.25;
          margin: 0 0 0.45rem 0;
        }
        .pipeline-card p {
          color: var(--app-muted) !important;
          font-size: 0.82rem;
          line-height: 1.5;
          letter-spacing: -0.01em;
          margin: 0;
        }
        .pipeline-step-badge {
          font-size: 0.68rem;
          font-weight: 800;
          letter-spacing: 0.08em;
          padding: 0.22rem 0.55rem;
          border-radius: 6px;
          background: rgba(0, 122, 255, 0.08);
          color: var(--accent-blue);
          border: 1px solid rgba(0, 122, 255, 0.2);
          backdrop-filter: var(--ios-blur-sm);
        }
        .pipeline-icon-box {
          display: inline-grid;
          place-items: center;
          width: 36px;
          height: 36px;
          border-radius: 10px;
          background: var(--glass-bg);
          border: 1px solid var(--glass-border);
          color: var(--accent-cyan);
          backdrop-filter: var(--ios-blur-sm);
          transition: all 0.2s ease;
        }
        .pipeline-card:hover .pipeline-icon-box {
          box-shadow: var(--glow-hover);
          border-color: var(--glass-border-hover);
        }

        .class-card {
          background: var(--glass-card) !important;
          backdrop-filter: var(--ios-blur) !important;
          -webkit-backdrop-filter: var(--ios-blur) !important;
          border: 1px solid var(--glass-border) !important;
          border-radius: 14px;
          padding: 1.15rem;
          min-height: 130px;
          box-shadow: var(--glass-shadow), var(--glass-specular) !important;
          transition: all 0.24s cubic-bezier(0.16, 1, 0.3, 1);
          display: flex;
          flex-direction: column;
          justify-content: space-between;
        }
        .class-card:hover {
          transform: translateY(-3px);
          background: var(--glass-card-hover) !important;
          border-color: var(--glass-border-hover) !important;
          box-shadow: var(--glow-hover) !important;
        }
        .class-card:active {
          transform: scale(0.98);
          box-shadow: var(--glow-active) !important;
        }
        .class-card-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 0.5rem;
        }
        .class-card-title-row {
          display: flex;
          align-items: center;
          gap: 0.45rem;
        }
        .class-card-title {
          color: var(--app-text) !important;
          font-size: 0.95rem;
          font-weight: 700;
          letter-spacing: -0.015em;
        }
        .class-card-desc {
          color: var(--app-muted) !important;
          font-size: 0.78rem;
          line-height: 1.45;
          letter-spacing: -0.01em;
          margin: 0;
        }

        .hud-card {
          position: relative;
          background: var(--glass-card) !important;
          backdrop-filter: var(--ios-blur) !important;
          -webkit-backdrop-filter: var(--ios-blur) !important;
          border: 1px solid var(--glass-border) !important;
          border-radius: 18px;
          padding: 0.95rem;
          margin-bottom: 0.8rem;
          overflow: hidden;
          box-shadow: var(--glass-shadow), var(--glass-specular) !important;
          transition: all 0.26s ease;
        }
        .hud-card:hover {
          border-color: var(--glass-border-hover) !important;
          box-shadow: var(--glow-hover) !important;
        }
        .hud-top-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding-bottom: 0.45rem;
          margin-bottom: 0.6rem;
          border-bottom: 1px dashed var(--glass-border);
          font-family: monospace;
        }
        .hud-tag {
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.8px;
          color: var(--accent-cyan) !important;
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
        }
        .hud-title {
          font-size: 0.82rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          color: var(--app-text) !important;
        }
        .hud-viewport {
          position: relative;
          border-radius: 14px;
          overflow: hidden;
          border: 1px solid var(--glass-border);
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
          border-top: 2px solid var(--accent-cyan);
          border-left: 2px solid var(--accent-cyan);
        }
        .hud-viewport::after {
          bottom: 6px; right: 6px;
          border-bottom: 2px solid var(--accent-cyan);
          border-right: 2px solid var(--accent-cyan);
        }
        .hud-scanline {
          position: absolute;
          left: 0; right: 0;
          height: 2px;
          background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
          box-shadow: 0 0 10px var(--accent-cyan);
          pointer-events: none;
          z-index: 3;
          animation: scanline-sweep 3s ease-in-out infinite alternate;
        }
        
        .stButton > button,
        .stButton > button[kind="primary"],
        [data-testid="stDownloadButton"] > button {
          border-radius: 14px !important;
          font-weight: 700 !important;
          letter-spacing: 0.3px !important;
          backdrop-filter: var(--ios-blur-sm) !important;
          -webkit-backdrop-filter: var(--ios-blur-sm) !important;
          transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
          cursor: pointer !important;
          position: relative !important;
        }

        .stButton > button[kind="primary"] {
          background: var(--btn-primary-bg) !important;
          color: #ffffff !important;
          border: 1px solid rgba(255, 255, 255, 0.35) !important;
          box-shadow: 0 4px 16px rgba(0, 122, 255, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
        }
        .stButton > button[kind="primary"]:hover {
          background: var(--btn-primary-hover) !important;
          border-color: #ffffff !important;
          box-shadow: var(--glow-hover) !important;
          transform: translateY(-2px) !important;
        }
        .stButton > button[kind="primary"]:active {
          box-shadow: var(--glow-active) !important;
          transform: scale(0.97) !important;
        }

        .stButton > button:not([kind="primary"]),
        [data-testid="stDownloadButton"] > button {
          border: 1px solid var(--btn-sec-border) !important;
          color: var(--btn-sec-text) !important;
          background: var(--btn-sec-bg) !important;
          box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08), var(--glass-specular) !important;
        }
        .stButton > button:not([kind="primary"]):hover,
        [data-testid="stDownloadButton"] > button:hover {
          border-color: var(--glass-border-hover) !important;
          background: var(--glass-card-hover) !important;
          box-shadow: var(--glow-hover) !important;
          transform: translateY(-2px) !important;
        }
        .stButton > button:not([kind="primary"]):active,
        [data-testid="stDownloadButton"] > button:active {
          box-shadow: var(--glow-active) !important;
          transform: scale(0.97) !important;
        }

        div[data-baseweb="select"] > div,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input {
          background: var(--glass-card) !important;
          backdrop-filter: var(--ios-blur-sm) !important;
          -webkit-backdrop-filter: var(--ios-blur-sm) !important;
          border: 1px solid var(--glass-border) !important;
          border-radius: 14px !important;
          color: var(--app-text) !important;
          box-shadow: var(--glass-specular) !important;
          transition: all 0.22s ease !important;
        }
        div[data-baseweb="select"] > div:hover,
        [data-testid="stTextInput"] input:hover,
        [data-testid="stNumberInput"] input:hover,
        [data-testid="stDateInput"] input:hover {
          border-color: var(--glass-border-hover) !important;
          box-shadow: var(--glow-hover) !important;
        }
        div[data-baseweb="select"]:focus-within > div,
        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stDateInput"] input:focus {
          border-color: var(--accent-cyan) !important;
          box-shadow: var(--glow-active) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
          border: 2px dashed var(--glass-border-hover) !important;
          border-radius: 20px !important;
          background: var(--glass-card) !important;
          backdrop-filter: var(--ios-blur) !important;
          -webkit-backdrop-filter: var(--ios-blur) !important;
          padding: 2.2rem 1.2rem !important;
          box-shadow: var(--glass-shadow), var(--glass-specular) !important;
          transition: all 0.25s ease !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
          border-color: var(--accent-cyan) !important;
          background: var(--glass-card-hover) !important;
          transform: translateY(-2px);
          box-shadow: var(--glow-hover) !important;
        }
        [data-testid="stFileUploaderDropzone"]:active {
          box-shadow: var(--glow-active) !important;
        }

        [data-baseweb="tab-list"] {
          background: var(--glass-card) !important;
          backdrop-filter: var(--ios-blur-sm) !important;
          -webkit-backdrop-filter: var(--ios-blur-sm) !important;
          border-radius: 14px !important;
          padding: 4px !important;
          border: 1px solid var(--glass-border) !important;
        }
        [data-baseweb="tab-list"] button {
          border-radius: 10px !important;
          transition: all 0.2s ease !important;
          color: var(--app-muted) !important;
        }
        [data-baseweb="tab-list"] button:hover {
          background: rgba(0, 122, 255, 0.12) !important;
          color: var(--accent-blue) !important;
          box-shadow: 0 0 16px rgba(0, 122, 255, 0.25) !important;
        }
        [data-baseweb="tab-list"] button[aria-selected="true"] {
          background: var(--glass-card-hover) !important;
          color: var(--accent-blue) !important;
          box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1) !important;
        }

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
          backdrop-filter: var(--ios-blur-sm);
          -webkit-backdrop-filter: var(--ios-blur-sm);
          transition: all 0.22s ease;
          cursor: default;
        }
        .badge:hover {
          box-shadow: 0 0 18px currentColor;
          transform: translateY(-1px);
        }
        .badge:active {
          box-shadow: 0 0 28px currentColor;
          transform: scale(0.96);
        }
        .badge-model-ready {
          background: rgba(16, 185, 129, 0.16);
          color: #10b981;
          border: 1px solid rgba(16, 185, 129, 0.4);
        }
        .badge-model-notready {
          background: rgba(100, 116, 139, 0.16);
          color: #94a3b8;
          border: 1px solid rgba(100, 116, 139, 0.35);
        }

        .pulse-indicator {
          display: inline-block;
          width: 8px;
          height: 8px;
          border-radius: 50%;
          margin-right: 6px;
          vertical-align: middle;
          transition: all 0.2s ease;
        }
        .pulse-dot-mint { background: #10b981; }
        .pulse-dot-amber { background: #f59e0b; }
        .pulse-dot-rose { background: #f43f5e; }
        .pulse-dot-cyan { background: #00d2ff; }

        .stDataFrame {
          border: 1px solid var(--glass-border) !important;
          border-radius: 18px !important;
          overflow: hidden;
          backdrop-filter: var(--ios-blur-sm) !important;
          box-shadow: var(--glass-shadow) !important;
        }

        @keyframes rise-in {
          from {opacity: 0; transform: translateY(10px);}
          to {opacity: 1; transform: translateY(0);}
        }
        @keyframes hero-enter {
          from {opacity: 0; transform: translateY(12px) scale(0.99);}
          to {opacity: 1; transform: translateY(0) scale(1);}
        }
        @keyframes scanline-sweep {
          0% { top: 6%; opacity: 0.2; }
          50% { opacity: 0.9; }
          100% { top: 92%; opacity: 0.2; }
        }

        html[data-theme-mode="light"] .hero-card,
        html:has([data-base-theme="light"]) .hero-card,
        [data-base-theme="light"] .hero-card,
        .stApp[data-base-theme="light"] .hero-card {
          background: linear-gradient(135deg,
            rgba(255, 255, 255, 0.94) 0%,
            rgba(240, 247, 255, 0.92) 55%,
            rgba(224, 242, 254, 0.85) 100%
          ) !important;
          border: 1px solid rgba(0, 122, 255, 0.22) !important;
          box-shadow:
            0 14px 38px rgba(15, 23, 42, 0.06),
            0 2px 6px rgba(15, 23, 42, 0.03),
            inset 0 1px 2px #ffffff !important;
          color: #0f172a !important;
        }
        html[data-theme-mode="light"] .hero-card:hover,
        html:has([data-base-theme="light"]) .hero-card:hover,
        [data-base-theme="light"] .hero-card:hover,
        .stApp[data-base-theme="light"] .hero-card:hover {
          border-color: rgba(0, 122, 255, 0.5) !important;
          box-shadow:
            0 18px 48px rgba(0, 122, 255, 0.16),
            0 4px 12px rgba(15, 23, 42, 0.04),
            inset 0 1px 2px #ffffff !important;
        }
        html[data-theme-mode="light"] .hero-card h1,
        html:has([data-base-theme="light"]) .hero-card h1,
        [data-base-theme="light"] .hero-card h1,
        .stApp[data-base-theme="light"] .hero-card h1 {
          background: linear-gradient(135deg, #0f172a 15%, #0369a1 85%) !important;
          -webkit-background-clip: text !important;
          -webkit-text-fill-color: transparent !important;
        }
        html[data-theme-mode="light"] .hero-card p,
        html:has([data-base-theme="light"]) .hero-card p,
        [data-base-theme="light"] .hero-card p,
        .stApp[data-base-theme="light"] .hero-card p {
          color: #475569 !important;
        }
        html[data-theme-mode="light"] .hero-card .hero-icon,
        html:has([data-base-theme="light"]) .hero-card .hero-icon,
        [data-base-theme="light"] .hero-card .hero-icon,
        .stApp[data-base-theme="light"] .hero-card .hero-icon {
          background: rgba(0, 122, 255, 0.1) !important;
          border-color: rgba(0, 122, 255, 0.25) !important;
          color: #007aff !important;
        }
        html[data-theme-mode="light"] .hero-card .hero-chip,
        html:has([data-base-theme="light"]) .hero-card .hero-chip,
        [data-base-theme="light"] .hero-card .hero-chip,
        .stApp[data-base-theme="light"] .hero-card .hero-chip {
          background: rgba(255, 255, 255, 0.85) !important;
          border: 1px solid rgba(0, 122, 255, 0.2) !important;
          color: #1e293b !important;
          box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04), inset 0 1px 1px #ffffff !important;
        }
        html[data-theme-mode="light"] .hero-card .hero-chip:hover,
        html:has([data-base-theme="light"]) .hero-card .hero-chip:hover,
        [data-base-theme="light"] .hero-card .hero-chip:hover,
        .stApp[data-base-theme="light"] .hero-card .hero-chip:hover {
          background: rgba(0, 122, 255, 0.12) !important;
          border-color: #007aff !important;
          color: #007aff !important;
          box-shadow: 0 4px 14px rgba(0, 122, 255, 0.22) !important;
        }

        html[data-theme-mode="light"] [data-testid="stSidebar"],
        html:has([data-base-theme="light"]) [data-testid="stSidebar"],
        [data-base-theme="light"] [data-testid="stSidebar"],
        .stApp[data-base-theme="light"] [data-testid="stSidebar"] {
          background: rgba(248, 250, 252, 0.88) !important;
          border-right: 1px solid rgba(0, 0, 0, 0.08) !important;
        }
        html[data-theme-mode="light"] [data-testid="stSidebar"] *,
        html:has([data-base-theme="light"]) [data-testid="stSidebar"] *,
        [data-base-theme="light"] [data-testid="stSidebar"] *,
        .stApp[data-base-theme="light"] [data-testid="stSidebar"] * {
          color: #0f172a !important;
        }
        html[data-theme-mode="light"] [data-testid="stSidebar"] [data-testid="stPageLink"] a,
        html:has([data-base-theme="light"]) [data-testid="stSidebar"] [data-testid="stPageLink"] a,
        [data-base-theme="light"] [data-testid="stSidebar"] [data-testid="stPageLink"] a,
        .stApp[data-base-theme="light"] [data-testid="stSidebar"] [data-testid="stPageLink"] a {
          background: rgba(255, 255, 255, 0.8) !important;
          border: 1px solid rgba(0, 0, 0, 0.06) !important;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
          color: #0f172a !important;
        }
        html[data-theme-mode="light"] [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover,
        html:has([data-base-theme="light"]) [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover,
        [data-base-theme="light"] [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover,
        .stApp[data-base-theme="light"] [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
          background: #ffffff !important;
          border-color: rgba(0, 122, 255, 0.35) !important;
          box-shadow: 0 4px 14px rgba(0, 122, 255, 0.14) !important;
          color: #007aff !important;
        }

        html[data-theme-mode="light"] .section-header h2,
        html:has([data-base-theme="light"]) .section-header h2,
        [data-base-theme="light"] .section-header h2,
        .stApp[data-base-theme="light"] .section-header h2 {
          color: #0f172a !important;
        }
        html[data-theme-mode="light"] .section-header-icon,
        html:has([data-base-theme="light"]) .section-header-icon,
        [data-base-theme="light"] .section-header-icon,
        .stApp[data-base-theme="light"] .section-header-icon {
          background: rgba(0, 122, 255, 0.08) !important;
          border-color: rgba(0, 122, 255, 0.22) !important;
          color: #007aff !important;
        }

        html[data-theme-mode="light"] .metric-val,
        html:has([data-base-theme="light"]) .metric-val,
        [data-base-theme="light"] .metric-val,
        .stApp[data-base-theme="light"] .metric-val,
        html[data-theme-mode="light"] .pipeline-card h3,
        html:has([data-base-theme="light"]) .pipeline-card h3,
        [data-base-theme="light"] .pipeline-card h3,
        .stApp[data-base-theme="light"] .pipeline-card h3,
        html[data-theme-mode="light"] .class-card-title,
        html:has([data-base-theme="light"]) .class-card-title,
        [data-base-theme="light"] .class-card-title,
        .stApp[data-base-theme="light"] .class-card-title {
          color: #0f172a !important;
        }

        html[data-theme-mode="light"] .metric-sub,
        html:has([data-base-theme="light"]) .metric-sub,
        [data-base-theme="light"] .metric-sub,
        .stApp[data-base-theme="light"] .metric-sub,
        html[data-theme-mode="light"] .pipeline-card p,
        html:has([data-base-theme="light"]) .pipeline-card p,
        [data-base-theme="light"] .pipeline-card p,
        .stApp[data-base-theme="light"] .pipeline-card p,
        html[data-theme-mode="light"] .class-card-desc,
        html:has([data-base-theme="light"]) .class-card-desc,
        [data-base-theme="light"] .class-card-desc,
        .stApp[data-base-theme="light"] .class-card-desc {
          color: #475569 !important;
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

    st.html(
        """
        <div class="scroll-progress-bar" id="scroll-progress-bar"></div>
        <div class="scroll-parallax-bg" id="scroll-parallax-bg"></div>
        <script>
        (function() {
          function detectTheme() {
            try {
              let pDoc = document;
              let app = document.querySelector('.stApp') || document.querySelector('[data-testid="stAppViewContainer"]');
              try {
                if (!app && window.parent && window.parent.document) {
                  pDoc = window.parent.document;
                  app = pDoc.querySelector('.stApp') || pDoc.querySelector('[data-testid="stAppViewContainer"]');
                }
              } catch (e) {}

              const baseTheme = app ? app.getAttribute('data-base-theme') : null;
              let isLight = false;
              if (baseTheme === 'light') {
                isLight = true;
              } else if (baseTheme === 'dark') {
                isLight = false;
              } else if (app) {
                const bg = (window.getComputedStyle ? window.getComputedStyle(app).backgroundColor : '') || '';
                const rgb = bg.match(/\\d+/g);
                if (rgb && rgb.length >= 3) {
                  const brightness = (parseInt(rgb[0])*299 + parseInt(rgb[1])*587 + parseInt(rgb[2])*114) / 1000;
                  isLight = brightness > 135;
                } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
                  isLight = true;
                }
              } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
                isLight = true;
              }
              const mode = isLight ? 'light' : 'dark';
              document.documentElement.setAttribute('data-theme-mode', mode);
              if (document.body) {
                document.body.setAttribute('data-theme-mode', mode);
              }
              try {
                if (pDoc && pDoc !== document) {
                  pDoc.documentElement.setAttribute('data-theme-mode', mode);
                  if (pDoc.body) {
                    pDoc.body.setAttribute('data-theme-mode', mode);
                  }
                }
              } catch (e) {}
            } catch(e) {}
          }

          function getScrollMetrics() {
            const candidates = [
              document.querySelector('[data-testid="stAppViewContainer"]'),
              document.querySelector('section[data-testid="stMain"]'),
              document.querySelector('.main'),
              document.documentElement,
              document.body
            ];
            for (const el of candidates) {
              if (el) {
                const top = el.scrollTop || 0;
                const scrollRange = (el.scrollHeight || 0) - (el.clientHeight || 0);
                if (scrollRange > 10) {
                  return { top: top, scrollRange: scrollRange };
                }
              }
            }
            const winTop = window.scrollY || window.pageYOffset || document.documentElement.scrollTop || 0;
            const winRange = Math.max(1, (document.documentElement.scrollHeight || 1) - window.innerHeight);
            return { top: winTop, scrollRange: winRange };
          }

          function updateScroll() {
            const metrics = getScrollMetrics();
            const ratio = Math.max(0, Math.min(1, metrics.top / Math.max(1, metrics.scrollRange)));
            const pct = Math.round(ratio * 100);

            const bar = document.getElementById('scroll-progress-bar');
            if (bar) {
              bar.style.width = pct + '%';
            }

            const bg = document.getElementById('scroll-parallax-bg');
            if (bg) {
              bg.style.transform = 'translateY(' + Math.round(-ratio * 70) + 'px)';
            }
          }

          window.addEventListener('scroll', updateScroll, { capture: true, passive: true });
          document.addEventListener('scroll', updateScroll, { capture: true, passive: true });

          function disableSidebarResizing() {
            try {
              const docs = [document];
              try { if (window.parent && window.parent.document) docs.push(window.parent.document); } catch(e) {}
              docs.forEach(function(d) {
                const resizers = d.querySelectorAll('[data-testid="stSidebarResizeHandle"], div[data-testid="stSidebarResizeHandle"], [data-testid="stSidebarResizerHandle"], div[data-testid="stSidebarResizerRoot"], div[class*="resizerHandle"], div[class*="Resizer"]');
                resizers.forEach(function(el) {
                  el.style.display = 'none';
                  el.style.pointerEvents = 'none';
                  el.style.width = '0px';
                });
              });
            } catch(e) {}
          }

          detectTheme();
          updateScroll();
          disableSidebarResizing();

          if (window.MutationObserver) {
            const observer = new MutationObserver(function() {
              detectTheme();
              disableSidebarResizing();
            });
            const target = document.querySelector('.stApp') || document.body;
            if (target) {
              observer.observe(target, { attributes: true, attributeFilter: ['data-base-theme', 'class', 'style'] });
            }
          }

          if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', detectTheme);
          }

          setTimeout(function() { detectTheme(); updateScroll(); disableSidebarResizing(); }, 80);
          setTimeout(function() { detectTheme(); updateScroll(); disableSidebarResizing(); }, 350);
          setTimeout(function() { detectTheme(); updateScroll(); disableSidebarResizing(); }, 1000);
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def page_setup(title: str, icon: str = ":material/directions_car:", current_page: str = "Home") -> None:
    """Set page config (must be the first Streamlit call on each page)."""
    st.set_page_config(page_title=title, page_icon=icon, layout="wide", initial_sidebar_state="expanded")
    inject_css()
    sidebar_nav(current_page)


def section_header(title: str, subtitle: str | None = None, icon: str | None = None) -> None:
    """Render a modern section header with clean SVG vector icon and divider."""
    icon_name = icon or icon_name_for(title)
    icon_svg = svg_icon(icon_name, size=18, color="var(--accent-blue)")
    sub_markup = f'<p class="section-header-sub">{subtitle}</p>' if subtitle else ''
    st.markdown(
        f"""
        <div class="section-header">
          <span class="section-header-icon">{icon_svg}</span>
          <div class="section-header-text">
            <h2>{title}</h2>
            {sub_markup}
          </div>
          <div class="divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def icon_name_for(section: str) -> str:
    """Return an SVG icon name for a section title."""
    mapping = {
        "Claim": "camera",
        "Dashboard": "dashboard",
        "Management Dashboard": "dashboard",
        "System overview": "cpu",
        "Result": "search",
        "Claim analytics": "currency",
        "Claims": "clipboard",
        "Claims register": "clipboard",
        "Key performance indicators": "activity",
        "Analytics charts": "chart",
        "Inspection pipeline": "layers",
        "Damage taxonomy": "tag",
    }
    return mapping.get(section, "activity")


def sidebar_nav(current_page: str = "Home") -> None:
    """Render a modern sidebar with app navigation and model status."""
    with st.sidebar:
        car_icon = svg_icon("car", size=22, color="var(--accent-blue)")
        st.markdown(
            f'''
            <div style="padding: 0.2rem 0.2rem 1.1rem 0.2rem; display: flex; align-items: center; gap: 0.75rem;">
              <span style="display:inline-grid; place-items:center; width:38px; height:38px; border-radius:11px; background:rgba(0,122,255,0.1); border:1px solid rgba(0,122,255,0.22); color:var(--accent-blue);">{car_icon}</span>
              <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: var(--app-text); line-height: 1.15; letter-spacing: -0.025em;">iClaim</div>
                <div style="font-size: 0.74rem; color: var(--app-muted); font-weight: 500;">Vehicle Damage AI</div>
              </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size: 0.68rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; color: var(--app-muted); margin-bottom: 0.6rem;">Navigation</div>',
            unsafe_allow_html=True,
        )
        st.page_link("app.py", label="Home", icon=":material/home:", disabled=(current_page == "Home"))
        st.page_link("pages/1_Claim.py", label="Claim Assessment", icon=":material/camera_alt:", disabled=(current_page == "Claim"))
        st.page_link("pages/2_Dashboard.py", label="Operations Dashboard", icon=":material/dashboard:", disabled=(current_page == "Dashboard"))
        st.divider()
        st.markdown(
            '<div style="font-size: 0.68rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; color: var(--app-muted); margin-bottom: 0.5rem;">AI Telemetry</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="padding: 0.65rem 0.85rem; border-radius: 12px; background: var(--glass-card); border: 1px solid var(--glass-border);">{model_status_badge()}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Automated decision-support. Final claim approval requires licensed surveyor inspection.")


def metric_card(
    label: str,
    value: str,
    sub: str | None = None,
    color: str | None = None,
    icon: str | None = None,
) -> None:
    """Render a modern automotive AI KPI card with crisp SVG vector icon and animated hover effect."""
    l_lower = label.lower()
    icon_name = icon or "activity"
    if icon is None:
        if "damage class" in l_lower or "damage type" in l_lower:
            icon_name = "tag"
        elif "severity" in l_lower:
            icon_name = "scales"
        elif "confidence" in l_lower:
            icon_name = "target"
        elif "priority" in l_lower:
            icon_name = "zap"
        elif "total claims" in l_lower:
            icon_name = "clipboard"
        elif "severe" in l_lower:
            icon_name = "alert"
        elif "manual review" in l_lower:
            icon_name = "search"
        elif "exposure" in l_lower or "cost" in l_lower:
            icon_name = "currency"
        elif "model" in l_lower:
            icon_name = "brain"
        elif "backbone" in l_lower:
            icon_name = "cpu"
        else:
            icon_name = "chart"

    if color is None:
        if "severe" in l_lower or "manual review" in l_lower or "not trained" in str(value).lower():
            color = "#f43f5e"
        elif "confidence" in l_lower or "ready" in str(value).lower() or "active" in str(value).lower():
            color = "#10b981"
        elif "severity" in l_lower or "priority" in l_lower:
            color = "#00d2ff"
        else:
            color = "#0284c7"

    icon_html = svg_icon(icon_name, size=16, color=color)
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""

    st.markdown(
        f"""
        <div class="metric-card-pro" style="--card-accent: {color};">
          <div class="metric-card-top">
            <span class="metric-label">{label}</span>
            <span class="metric-icon-badge" style="color:{color}; background:color-mix(in srgb, {color} 12%, var(--glass-bg)); border:1px solid color-mix(in srgb, {color} 24%, transparent);">{icon_html}</span>
          </div>
          <div class="metric-val">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_badge(level: str) -> str:
    """Return HTML for a styled severity badge with status indicator."""
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
    """Return HTML for a styled claim priority badge with status indicator and SVG status icon."""
    p_lower = priority.lower()
    if "low" in p_lower:
        color, dot, icon_name = "#10b981", "pulse-dot-mint", "check"
    elif "medium" in p_lower:
        color, dot, icon_name = "#00d2ff", "pulse-dot-cyan", "activity"
    elif "high" in p_lower:
        color, dot, icon_name = "#f59e0b", "pulse-dot-amber", "zap"
    else:
        color, dot, icon_name = "#f43f5e", "pulse-dot-rose", "alert"

    status_icon_svg = svg_icon(icon_name, size=13, color=color)

    return (
        f'<span class="badge" style="background:color-mix(in srgb, {color} 16%, var(--panel)); '
        f'color:{color}; border:1px solid color-mix(in srgb, {color} 40%, transparent);">'
        f'<span class="pulse-indicator {dot}"></span>{status_icon_svg} {priority}'
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
    """Explain that sample claims are synthetic using vector icon styling."""
    info_icon = svg_icon("alert", size=18, color="#00d2ff")
    st.markdown(
        f"""
        <div style="background:color-mix(in srgb, var(--cyan) 8%, var(--panel)); border:1px solid color-mix(in srgb, var(--cyan) 30%, transparent); border-radius:12px; padding:0.75rem 1.1rem; margin-bottom:1rem; display:flex; align-items:center; gap:0.75rem;">
          <span style="display:inline-grid; place-items:center; color:var(--cyan);">{info_icon}</span>
          <span style="font-size:0.85rem; color:var(--ink);">
            Rows marked <code>synthetic</code> are generated simulation records. Repair-cost ranges are indicative actuarial assumptions from config.yaml.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hud_image_frame(title: str, telemetry: str = "CAM_INSPECTION // ACTIVE", icon: str = "camera") -> str:
    """Return HTML header banner for automotive vision inspection HUD viewport with SVG icon."""
    icon_svg = svg_icon(icon, size=14, color="var(--cyan)")
    return f"""
    <div class="hud-top-bar">
      <span class="hud-tag"><span class="pulse-indicator pulse-dot-cyan"></span>{icon_svg} {telemetry}</span>
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
