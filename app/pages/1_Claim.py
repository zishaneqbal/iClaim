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

import pandas as pd
import streamlit as st
from PIL import Image

from components.charts import probability_chart
from components.ui import (
    CFG,
    get_model,
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

page_setup("Claim", "📷")

section_header("Claim", "Upload a vehicle damage photo and get a preliminary claim assessment")

if not require_model():
    st.stop()

# ---- Input cards ------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    with st.container(border=True):
        st.subheader("Vehicle image")
        upload = st.file_uploader(
            "Upload a photo of the damaged vehicle",
            label_visibility="collapsed",
            type=[e.lstrip(".") for e in CFG["data"]["allowed_extensions"]],
            help=f"Max {CFG['data']['max_upload_mb']} MB. Clear, well-lit, close-up photos work best.",
        )

with right:
    with st.container(border=True):
        st.subheader("Claim details")
        claim_id = st.text_input("Claim ID (optional)", placeholder="e.g. CLM-2026-0142", label_visibility="collapsed")
        c1, c2 = st.columns(2)
        claim_date = c1.date_input("Claim date", value=date.today())
        vehicle_type = c2.selectbox("Vehicle type", CFG["synthetic_claims"]["vehicle_types"])
        policy_type = c1.selectbox("Policy type", CFG["synthetic_claims"]["policy_types"])
        claimed_amount = c2.number_input(
            f"Claimed amount ({CFG['analytics']['currency']}, optional)", min_value=0.0, step=1000.0, value=0.0
        )

analyse = st.button("Analyse damage", type="primary", disabled=upload is None)

if analyse and upload is not None:
    data = upload.getvalue()
    check = validate_image(data, filename=upload.name, size_bytes=len(data), cfg=CFG)
    if not check.ok:
        st.error(check.message)
        st.session_state.pop("result", None)
    else:
        try:
            safe_claim_id = validate_claim_id(claim_id)
            with st.spinner("Analysing image..."):
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

    section_header("Result", "Preliminary damage assessment")

    # quality warnings
    if res["blur"] < BLUR_THRESHOLD:
        st.warning("📷 The image looks blurry, so the prediction may be unreliable. Consider a sharper photo.")
    if pred["low_confidence"]:
        st.warning(
            f"⚠️ Low model confidence ({pred['class_confidence']:.0%}). "
            "This claim is routed to **manual review**. The image may not show a vehicle "
            "or the damage may be unclear."
        )

    # top metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Damage type", label(pred["damage_class"]))
    with m2:
        metric_card("Severity", pred["severity"].capitalize(), sub=f"score {pred['severity_score']:.2f}")
    with m3:
        metric_card("Confidence", f"{pred['class_confidence']:.0%}")
    with m4:
        metric_card("Priority", record["priority"])

    # image / grad-cam / probabilities
    img_col, cam_col, prob_col = st.columns([1, 1, 1.1])
    with img_col:
        with st.container(border=True):
            st.subheader("Uploaded image")
            st.image(Image.open(io.BytesIO(res["image_bytes"])), width="stretch")
    with cam_col:
        with st.container(border=True):
            st.subheader("Grad-CAM heatmap")
            if pred.get("gradcam_overlay") is not None:
                st.image(pred["gradcam_overlay"], caption="Where the model looked", width="stretch")
            else:
                st.caption("Grad-CAM overlay unavailable for this model.")
    with prob_col:
        with st.container(border=True):
            st.subheader("Class probabilities")
            st.plotly_chart(probability_chart(pred["class_probabilities"], pred["damage_class"]), width="stretch")

    # claim analytics card
    with st.container(border=True):
        st.subheader("Claim analytics")
        cost = format_range_inr(record["est_cost_low"], record["est_cost_high"])
        flag_text = {
            "within_range": "Claimed amount is within the indicative range.",
            "above_range": "Claimed amount is **above** the indicative range - review recommended.",
            "below_range": "Claimed amount is below the indicative range.",
            "no_amount": "No claimed amount entered.",
        }[record["amount_flag"]]
        ca1, ca2 = st.columns(2)
        ca1.write(f"**Indicative repair cost:** {cost} (assumption-based, not a quote)")
        ca1.write(f"**Severity:** {severity_badge(pred['severity'])}", unsafe_allow_html=True)
        ca2.write(f"**Priority:** {priority_badge(record['priority'])}", unsafe_allow_html=True)
        ca2.write(f"**Amount flag:** {flag_text}")
        area_txt = ""
        if pred.get("damage_area_ratio") is not None:
            area_txt = f"Grad-CAM damage area {pred['damage_area_ratio']:.0%}"
        st.caption(f"Severity score: {pred['severity_score']:.2f}" + (f"  •  {area_txt}" if area_txt else ""))

    # raw record table (collapsible)
    with st.expander("View full claim record", expanded=False):
        st.dataframe(pd.DataFrame([record]), width="stretch", hide_index=True)

    # save button
    if st.button("Save to dashboard", type="primary", disabled=res["saved"]):
        append_prediction_log(record, CFG)
        res["saved"] = True
        st.success("Saved to dashboard. Open the Dashboard page to see it.")
    elif res["saved"]:
        st.success("Saved to dashboard.")
