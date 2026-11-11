"""Streamlit entry point (Home).

Run:  streamlit run app/app.py
"""

from __future__ import annotations

import importlib
import streamlit as st

from components import ui

try:
    importlib.reload(ui)
except Exception:
    pass

from components.ui import (
    CFG,
    model_available,
    model_status_badge,
    metric_card,
    page_setup,
    section_header,
    svg_icon,
)

page_setup("iClaim", ":material/directions_car:", current_page="Home")

car_hero_icon = svg_icon("car", size=32, color="#00d2ff")
brain_icon = svg_icon("brain", size=14, color="#00d2ff")
microscope_icon = svg_icon("microscope", size=14, color="#00d2ff")
zap_icon = svg_icon("zap", size=14, color="#f59e0b")
currency_icon = svg_icon("currency", size=14, color="#10b981")

st.markdown(
    f"""
    <div class="hero-card">
      <div style="display:flex; align-items:flex-start; gap:1.35rem;">
        <span class="hero-icon">{car_hero_icon}</span>
        <div style="flex:1;">
          <h1>Vehicle Damage Intelligence</h1>
          <p>
            Autonomous CNN-powered preliminary damage assessment with insurance claim triage,
            Grad-CAM explainability heatmaps, and actuarial cost analytics.
          </p>
          <div class="hero-chips-row">
            <span class="hero-chip">{brain_icon} MobileNetV2 Transfer Head</span>
            <span class="hero-chip">{microscope_icon} Grad-CAM Visual Saliency</span>
            <span class="hero-chip">{zap_icon} Automated Severity Triage</span>
            <span class="hero-chip">{currency_icon} INR Repair Estimation</span>
          </div>
        </div>
        <div style="position:relative; z-index:1;">
          {model_status_badge()}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c_act1, c_act2 = st.columns(2)
with c_act1:
    camera_act_icon = svg_icon("camera", size=22, color="#00d2ff")
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.4rem;">
              <span style="display:inline-grid; place-items:center; width:38px; height:38px; border-radius:10px; background:rgba(0,210,255,0.12); border:1px solid rgba(0,210,255,0.25); color:#00d2ff;">
                {camera_act_icon}
              </span>
              <h3 style="margin:0; font-size:1.15rem; font-weight:700;">Claim Assessment</h3>
            </div>
            <p style="color:var(--muted); font-size:0.88rem; line-height:1.45; margin-bottom:0.75rem;">
              Upload a vehicle photograph and policy details. Generates damage class prediction,
              Grad-CAM attention heatmap, blur quality check, and processing priority routing.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/1_Claim.py", label="Launch Claim Assessment", icon=":material/camera_alt:")

with c_act2:
    dash_act_icon = svg_icon("dashboard", size=22, color="#3b82f6")
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.4rem;">
              <span style="display:inline-grid; place-items:center; width:38px; height:38px; border-radius:10px; background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.25); color:#3b82f6;">
                {dash_act_icon}
              </span>
              <h3 style="margin:0; font-size:1.15rem; font-weight:700;">Operations Dashboard</h3>
            </div>
            <p style="color:var(--muted); font-size:0.88rem; line-height:1.45; margin-bottom:0.75rem;">
              Management overview for claims officers: damage type distribution, severity frequency,
              indicative repair-cost exposure ranges, and audit anomaly flags.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/2_Dashboard.py", label="Open Operations Dashboard", icon=":material/dashboard:")

section_header("System overview", "Core neural network and triage parameters at a glance", icon="cpu")
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card(
        "Damage classes",
        str(len(CFG["data"]["class_names"])),
        "scratch • dent • crack • glass • major",
        color="#00d2ff",
        icon="tag",
    )
with c2:
    metric_card(
        "Severity levels",
        "3",
        "minor • moderate • severe",
        color="#f59e0b",
        icon="scales",
    )
with c3:
    is_ready = model_available(CFG)
    metric_card(
        "Model state",
        "Active" if is_ready else "Not trained",
        "MobileNetV2 head loaded" if is_ready else "Run training script",
        color="#10b981" if is_ready else "#f43f5e",
        icon="brain",
    )
with c4:
    metric_card(
        "Backbone",
        "MobileNetV2",
        "ImageNet transfer weights",
        color="#0284c7",
        icon="cpu",
    )

section_header("Inspection pipeline", "End-to-end automated assessment architecture", icon="layers")

p1, p2, p3, p4 = st.columns(4)
with p1:
    icon_step1 = svg_icon("camera", size=20, color="#00d2ff")
    st.markdown(
        f"""
        <div class="pipeline-card">
          <div class="pipeline-card-top">
            <span class="pipeline-step-badge">STEP 01</span>
            <span class="pipeline-icon-box">{icon_step1}</span>
          </div>
          <h3>Image Intake & QA</h3>
          <p>
            Validates image MIME type, payload size (&le;10 MB), and estimates Laplacian blur variance to flag unsharp photos.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p2:
    icon_step2 = svg_icon("brain", size=20, color="#00d2ff")
    st.markdown(
        f"""
        <div class="pipeline-card">
          <div class="pipeline-card-top">
            <span class="pipeline-step-badge">STEP 02</span>
            <span class="pipeline-icon-box">{icon_step2}</span>
          </div>
          <h3>CNN Classification</h3>
          <p>
            MobileNetV2 backbone extracts deep visual features. Dense classification head computes class probability distribution.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p3:
    icon_step3 = svg_icon("microscope", size=20, color="#00d2ff")
    st.markdown(
        f"""
        <div class="pipeline-card">
          <div class="pipeline-card-top">
            <span class="pipeline-step-badge">STEP 03</span>
            <span class="pipeline-icon-box">{icon_step3}</span>
          </div>
          <h3>Grad-CAM Explainability</h3>
          <p>
            Computes gradients of predicted class against the final convolutional layer, projecting spatial activation heatmaps.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p4:
    icon_step4 = svg_icon("scales", size=20, color="#00d2ff")
    st.markdown(
        f"""
        <div class="pipeline-card">
          <div class="pipeline-card-top">
            <span class="pipeline-step-badge">STEP 04</span>
            <span class="pipeline-icon-box">{icon_step4}</span>
          </div>
          <h3>Triage & Cost Routing</h3>
          <p>
            Combines damage class, confidence, and activated area ratio to evaluate severity, indicative INR cost range, and priority.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

section_header("Damage taxonomy", "Recognized damage categories and baseline operational handling", icon="tag")

t1, t2, t3, t4, t5 = st.columns(5)
with t1:
    spark_icon = svg_icon("spark", size=16, color="#00d2ff")
    st.markdown(
        f"""
        <div class="class-card">
          <div class="class-card-top">
            <div class="class-card-title-row">
              {spark_icon}
              <span class="class-card-title">Scratch</span>
            </div>
            <span class="badge" style="background:rgba(16,185,129,0.15); color:#10b981; padding:0.2rem 0.5rem; font-size:0.7rem;">Minor</span>
          </div>
          <p class="class-card-desc">Clear-coat and surface paint abrasions. Fast-track approval candidate.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with t2:
    hammer_icon = svg_icon("hammer", size=16, color="#f59e0b")
    st.markdown(
        f"""
        <div class="class-card">
          <div class="class-card-top">
            <div class="class-card-title-row">
              {hammer_icon}
              <span class="class-card-title">Dent</span>
            </div>
            <span class="badge" style="background:rgba(245,158,11,0.15); color:#f59e0b; padding:0.2rem 0.5rem; font-size:0.7rem;">Moderate</span>
          </div>
          <p class="class-card-desc">Sheet metal deformations requiring paintless dent repair (PDR) or panel work.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with t3:
    crack_icon = svg_icon("zap", size=16, color="#f59e0b")
    st.markdown(
        f"""
        <div class="class-card">
          <div class="class-card-top">
            <div class="class-card-title-row">
              {crack_icon}
              <span class="class-card-title">Crack</span>
            </div>
            <span class="badge" style="background:rgba(245,158,11,0.15); color:#f59e0b; padding:0.2rem 0.5rem; font-size:0.7rem;">Moderate</span>
          </div>
          <p class="class-card-desc">Fractures in plastic bumpers, fiberglass, or body trim assemblies.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with t4:
    glass_icon = svg_icon("glass", size=16, color="#8b5cf6")
    st.markdown(
        f"""
        <div class="class-card">
          <div class="class-card-top">
            <div class="class-card-title-row">
              {glass_icon}
              <span class="class-card-title">Broken Glass</span>
            </div>
            <span class="badge" style="background:rgba(139,92,246,0.15); color:#8b5cf6; padding:0.2rem 0.5rem; font-size:0.7rem;">Moderate</span>
          </div>
          <p class="class-card-desc">Windshield, window, or light cluster compromise requiring replacement.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with t5:
    flame_icon = svg_icon("flame", size=16, color="#f43f5e")
    st.markdown(
        f"""
        <div class="class-card">
          <div class="class-card-top">
            <div class="class-card-title-row">
              {flame_icon}
              <span class="class-card-title">Major Body</span>
            </div>
            <span class="badge" style="background:rgba(244,63,94,0.15); color:#f43f5e; padding:0.2rem 0.5rem; font-size:0.7rem;">Severe</span>
          </div>
          <p class="class-card-desc">High-impact chassis/crumple-zone damage requiring full physical adjuster dispatch.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not model_available(CFG):
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Train a model first")
        st.write(
            "No trained model found at `models/best_model.keras`. The Dashboard still works "
            "using synthetic sample claims, but live photo analysis requires trained weights."
        )
        st.code("python -m src.data_loader --prepare\npython -m src.train --config config.yaml", language="bash")
        st.caption("See `docs/DATASET_GUIDE.md` for dataset download steps.")
