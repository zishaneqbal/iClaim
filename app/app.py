"""Streamlit entry point (Home).

Run:  streamlit run app/app.py
"""

from __future__ import annotations

import streamlit as st

from components.ui import (
    CFG,
    model_available,
    model_status_badge,
    metric_card,
    page_setup,
    section_header,
)

page_setup("iClaim", "🚗")

# ---- Hero header -----------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-card">
      <div style="display:flex; align-items:center; gap:1rem;">
        <span class="hero-icon">🚗</span>
        <div>
          <h1>Vehicle Damage Intelligence</h1>
          <p>
            CNN-based preliminary damage assessment with insurance claim analytics
          </p>
        </div>
        <div style="margin-left:auto; position:relative; z-index:1;">
          {model_status_badge()}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
# ---- Intro copy ------------------------------------------------------------
st.markdown(
    """
This decision-support tool helps insurance staff get a **fast, consistent first look** at vehicle
damage photographs. It supports, and does not replace, professional inspection and final claim
approval.

**What it does**

- **Claim**: upload a vehicle photo and claim details. A CNN classifies the damage
  type, the system assesses severity, shows a Grad-CAM heatmap, and suggests a processing priority.
- **Dashboard**: management view of damage frequency, severity distribution, indicative
  repair-cost ranges and claim priorities.
"""
)

# ---- Stat cards ------------------------------------------------------------
section_header("System overview", "Key facts at a glance")
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Damage classes", str(len(CFG["data"]["class_names"])), "scratch • dent • crack • broken glass • major body damage")
with c2:
    metric_card("Severity levels", "3", "minor • moderate • severe")
with c3:
    metric_card("Model", "Ready" if model_available(CFG) else "Not trained", "MobileNetV2 transfer head" if model_available(CFG) else "Train with `python -m src.train`", color="#1baf7a" if model_available(CFG) else "#e87ba4")
with c4:
    metric_card("Backbone", "MobileNetV2", "ImageNet-pretrained transfer learning")

# ---- Not trained nudge -----------------------------------------------------
if not model_available(CFG):
    with st.container(border=True):
        st.subheader("Train a model first")
        st.write(
            "No trained model found at `models/best_model.keras`. The Dashboard still works "
            "using synthetic sample claims, but image prediction needs a trained model."
        )
        st.code("python -m src.data_loader --prepare\npython -m src.train --config config.yaml", language="bash")
        st.caption("See `docs/DATASET_GUIDE.md` for dataset download steps.")

# ---- Supported classes -----------------------------------------------------
with st.expander("Supported damage classes", expanded=False):
    st.write(
        ", ".join(c.replace("_", " ") for c in CFG["data"]["class_names"])
    )
