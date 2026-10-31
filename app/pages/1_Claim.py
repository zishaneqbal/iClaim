"""Upload a vehicle image + claim details, run the CNN, and review the result."""

from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import importlib
import pandas as pd
import streamlit as st
from PIL import Image

from components import charts

try:
    importlib.reload(charts)
except Exception:
    pass
from components.charts import probability_chart
from components.ui import (
    CFG,
    get_model,
    hud_image_frame,
    label,
    metric_card,
    page_setup,
    priority_badge,
    require_model,
    section_header,
    severity_badge,
)
from src.analytics import append_prediction_log, build_claim_record, validate_claim_id
from src.config import get_logger
from src.predict import predict
from src.utils import ImageValidationError, estimate_blur, format_range_inr, validate_image

BLUR_THRESHOLD = 60.0  # variance of Laplacian below this is considered blurry
logger = get_logger(__name__)

page_setup("Claim Assessment", "📸")

section_header("Claim", "Upload a vehicle damage photograph for neural classification, Grad-CAM saliency, and actuarial triage")

if not require_model():
    st.stop()

# ---- Input cards ------------------------------------------------------------
left, right = st.columns([1.1, 1])

with left:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.6rem;">
              <h3 style="margin:0; font-size:1.1rem; font-weight:700;">📸 Vehicle Inspection Photo</h3>
              <span class="badge" style="background:rgba(0,210,255,0.12); color:#00d2ff; font-size:0.7rem; border:1px solid rgba(0,210,255,0.25);">Optical Input</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        upload = st.file_uploader(
            "Upload a photo of the damaged vehicle",
            label_visibility="collapsed",
            type=[e.lstrip(".") for e in CFG["data"]["allowed_extensions"]],
            help=f"Max {CFG['data']['max_upload_mb']} MB. Clear, well-lit, close-up photos of vehicle panels work best.",
        )
        st.caption(f"Accepted: {', '.join(CFG['data']['allowed_extensions'])} • Max size: {CFG['data']['max_upload_mb']} MB")

with right:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.6rem;">
              <h3 style="margin:0; font-size:1.1rem; font-weight:700;">📋 Policy & Claim Metadata</h3>
              <span class="badge" style="background:rgba(23,105,224,0.12); color:#0284c7; font-size:0.7rem; border:1px solid rgba(23,105,224,0.25);">Policy Record</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        claim_id = st.text_input("Claim ID (optional)", placeholder="e.g. CLM-2026-0142", label_visibility="collapsed")
        c1, c2 = st.columns(2)
        claim_date = c1.date_input("Claim date", value=date.today())
        vehicle_type = c2.selectbox("Vehicle type", CFG["synthetic_claims"]["vehicle_types"])
        policy_type = c1.selectbox("Policy type", CFG["synthetic_claims"]["policy_types"])
        claimed_amount = c2.number_input(
            f"Claimed amount ({CFG['analytics']['currency']}, optional)", min_value=0.0, step=1000.0, value=0.0
        )

st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
analyse = st.button("🚀 Run AI Damage Assessment", type="primary", disabled=upload is None, use_container_width=True)

if analyse and upload is not None:
    data = upload.getvalue()
    check = validate_image(data, filename=upload.name, size_bytes=len(data), cfg=CFG)
    if not check.ok:
        st.error(check.message)
        st.session_state.pop("result", None)
    else:
        try:
            safe_claim_id = validate_claim_id(claim_id)
            with st.spinner("Executing MobileNetV2 neural forward pass and Grad-CAM generation..."):
                pred = predict(data, model=get_model(), cfg=CFG, return_overlay=True)
            claim_info = {
                "claim_id": safe_claim_id,
                "claim_date": claim_date,
                "vehicle_type": vehicle_type,
                "policy_type": policy_type,
                "claimed_amount": claimed_amount or None,
            }
            record = build_claim_record(claim_info, pred, CFG, data_source="user")
            st.session_state["result"] = {
                "pred": pred,
                "record": record,
                "image_bytes": data,
                "blur": estimate_blur(data),
                "saved": False,
            }
        except ImageValidationError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(str(exc))
        except Exception:  # keep the UI friendly
            logger.exception("Image analysis failed")
            st.error("Image analysis failed. Please try a different image or verify the model setup.")

# ---- Result ----------------------------------------------------------------
res = st.session_state.get("result")
if res:
    pred, record = res["pred"], res["record"]

    section_header("Result", "Deep learning inference telemetry and preliminary claim assessment")

    # Assessment summary status strip
    display_date = (
        record["claim_date"].strftime("%Y-%m-%d")
        if hasattr(record["claim_date"], "strftime")
        else str(record["claim_date"])
    )
    st.markdown(
        f"""
        <div style="background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:0.85rem 1.25rem; margin-bottom:1.25rem; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.8rem;">
          <div style="display:flex; align-items:center; gap:0.6rem;">
            <span class="pulse-indicator pulse-dot-mint"></span>
            <strong style="font-size:0.95rem;">Assessment Completed</strong>
            <span style="color:var(--muted); font-size:0.85rem;">• Claim ID: <code>{record['claim_id']}</code> • Date: {display_date}</span>
          </div>
          <div>
            {priority_badge(record['priority'])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quality warnings
    if res["blur"] < BLUR_THRESHOLD:
        st.warning(f"📷 Low sharpness detected (variance of Laplacian: {res['blur']:.1f} < {BLUR_THRESHOLD:.1f}). Prediction confidence may be affected. A sharper photo is recommended.")
    if pred["low_confidence"]:
        st.warning(
            f"⚠️ Low neural confidence ({pred['class_confidence']:.0%}). "
            "This claim has been automatically routed to **Manual Review**. The image may not show an identifiable vehicle panel "
            "or the damage is outside standard distribution."
        )

    # Top metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Damage class", label(pred["damage_class"]), sub="MobileNetV2 classification", icon="🏷️")
    with m2:
        metric_card("Severity", pred["severity"].capitalize(), sub=f"Severity score {pred['severity_score']:.2f}", icon="⚖️")
    with m3:
        metric_card("Confidence", f"{pred['class_confidence']:.0%}", sub="Top-1 probability score", icon="🎯")
    with m4:
        metric_card("Priority triage", record["priority"], sub="Operational workflow routing", icon="⚡")

    # Image / Grad-CAM / Probabilities
    img_col, cam_col, prob_col = st.columns([1, 1, 1.1])
    with img_col:
        with st.container(border=True):
            st.markdown(hud_image_frame("Uploaded Image", "SENSOR: OPTICAL RGB INPUT"), unsafe_allow_html=True)
            st.image(Image.open(io.BytesIO(res["image_bytes"])), width="stretch")
    with cam_col:
        with st.container(border=True):
            st.markdown(hud_image_frame("Grad-CAM Heatmap", "EXPLAINABILITY"), unsafe_allow_html=True)
            if pred.get("gradcam_overlay") is not None:
                st.image(pred["gradcam_overlay"], caption="Where the CNN focused its attention", width="stretch")
            else:
                st.caption("Grad-CAM overlay unavailable for this model.")
    with prob_col:
        with st.container(border=True):
            st.subheader("Class probabilities")
            st.plotly_chart(probability_chart(pred["class_probabilities"], pred["damage_class"]), width="stretch")

    # Claim analytics card
    with st.container(border=True):
        st.subheader("Claim analytics & actuarial triage")
        cost = format_range_inr(record["est_cost_low"], record["est_cost_high"])
        flag_text = {
            "within_range": "Claimed amount is within the indicative range.",
            "above_range": "Claimed amount is **above** the indicative range — physical adjuster review recommended.",
            "below_range": "Claimed amount is below the indicative range.",
            "no_amount": "No claimed amount entered.",
        }[record["amount_flag"]]

        ca1, ca2 = st.columns(2)
        with ca1:
            st.markdown(f"**Indicative repair cost:** <span style='font-size:1.15rem; font-weight:800; color:var(--cyan);'>{cost}</span> *(assumption-based estimate)*", unsafe_allow_html=True)
            st.markdown(f"**Severity level:** {severity_badge(pred['severity'])}", unsafe_allow_html=True)
        with ca2:
            st.markdown(f"**Routing priority:** {priority_badge(record['priority'])}", unsafe_allow_html=True)
            st.markdown(f"**Amount audit flag:** {flag_text}")

        area_txt = ""
        if pred.get("damage_area_ratio") is not None:
            area_txt = f"Grad-CAM activated damage area: {pred['damage_area_ratio']:.0%}"
        st.caption(f"Severity score: {pred['severity_score']:.2f}" + (f"  •  {area_txt}" if area_txt else "") + "  •  Subject to manual adjuster verification.")

    # Raw record table (collapsible)
    with st.expander("View full claim JSON / tabular record", expanded=False):
        st.dataframe(pd.DataFrame([record]), width="stretch", hide_index=True)

    # Save button
    if st.button("💾 Save Record to Operations Dashboard", type="primary", disabled=res["saved"], use_container_width=True):
        append_prediction_log(record, CFG)
        res["saved"] = True
        st.success("Claim record saved successfully to the dashboard register.")
    elif res["saved"]:
        st.success("Claim record saved to the dashboard register.")
