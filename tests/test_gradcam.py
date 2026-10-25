"""Grad-CAM heatmaps."""

import numpy as np
import pytest

from src.gradcam import damage_area_ratio, make_gradcam_heatmap, overlay_heatmap
from src.model import build_model
from src.utils import preprocess_image


@pytest.mark.parametrize("backbone", ["baseline", "mobilenetv2"])
def test_heatmap_range_and_shape(cfg, rgb_image, backbone):
    model = build_model(backbone, cfg)
    heat = make_gradcam_heatmap(model, preprocess_image(rgb_image, cfg))
    assert heat.ndim == 2
    assert 0.0 <= heat.min() and heat.max() <= 1.0 + 1e-6


def test_explain_specific_class(cfg, baseline_model, rgb_image):
    batch = preprocess_image(rgb_image, cfg)
    heat = make_gradcam_heatmap(baseline_model, batch, class_index=2)
    assert heat.shape[0] > 1


def test_area_ratio_values():
    heat = np.zeros((10, 10), dtype="float32")
    heat[:5, :] = 1.0
    assert damage_area_ratio(heat, 0.5) == pytest.approx(0.5)
    assert damage_area_ratio(np.zeros((4, 4)), 0.5) == 0.0
    assert damage_area_ratio(np.array([]), 0.5) == 0.0


def test_overlay_matches_image_size(rgb_image):
    overlay = overlay_heatmap(rgb_image, np.random.rand(7, 7).astype("float32"))
    assert overlay.shape == (rgb_image.height, rgb_image.width, 3)
    assert overlay.dtype == np.uint8
