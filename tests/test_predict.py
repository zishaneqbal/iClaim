"""predict() output contract (the Streamlit app and analytics depend on it)."""

import io

import numpy as np
import pytest
from PIL import Image

from src.predict import predict
from src.severity import SEVERITY_LEVELS
from src.utils import ImageValidationError

CONTRACT_KEYS = {"damage_class", "class_confidence", "severity", "class_probabilities"}


def test_contract_keys_and_types(cfg, baseline_model, rgb_image):
    out = predict(rgb_image, model=baseline_model, cfg=cfg)
    assert CONTRACT_KEYS <= set(out)
    assert out["damage_class"] in cfg["data"]["class_names"]
    assert out["severity"] in SEVERITY_LEVELS
    assert 0.0 <= out["class_confidence"] <= 1.0
    assert set(out["class_probabilities"]) == set(cfg["data"]["class_names"])
    assert abs(sum(out["class_probabilities"].values()) - 1.0) < 0.01


def test_confidence_matches_top_probability(cfg, baseline_model, rgb_image):
    out = predict(rgb_image, model=baseline_model, cfg=cfg)
    top = max(out["class_probabilities"], key=out["class_probabilities"].get)
    assert top == out["damage_class"]
    assert abs(out["class_probabilities"][top] - out["class_confidence"]) < 1e-3


def test_accepts_bytes_and_numpy(cfg, baseline_model, rgb_image):
    buf = io.BytesIO()
    rgb_image.save(buf, format="PNG")
    assert predict(buf.getvalue(), model=baseline_model, cfg=cfg, use_gradcam=False)["damage_class"]
    arr = np.asarray(rgb_image)
    assert predict(arr, model=baseline_model, cfg=cfg, use_gradcam=False)["damage_class"]


def test_invalid_image_raises(cfg, baseline_model):
    with pytest.raises(ImageValidationError):
        predict(b"not an image", model=baseline_model, cfg=cfg)


def test_overlay_returned_when_requested(cfg, baseline_model, rgb_image):
    out = predict(rgb_image, model=baseline_model, cfg=cfg, return_overlay=True)
    overlay = out["gradcam_overlay"]
    assert overlay.shape == (rgb_image.height, rgb_image.width, 3)
    assert overlay.dtype == np.uint8


def test_no_gradcam_means_no_area(cfg, baseline_model, rgb_image):
    out = predict(rgb_image, model=baseline_model, cfg=cfg, use_gradcam=False)
    assert out["damage_area_ratio"] is None


def test_low_confidence_flag_consistent(cfg, baseline_model, rgb_image):
    out = predict(rgb_image, model=baseline_model, cfg=cfg)
    assert out["low_confidence"] == (out["class_confidence"] < cfg["analytics"]["min_confidence"])


def test_class_count_mismatch_raises(cfg, baseline_model, rgb_image):
    with pytest.raises(ValueError):
        predict(rgb_image, model=baseline_model, cfg=cfg, class_names=["a", "b"])
