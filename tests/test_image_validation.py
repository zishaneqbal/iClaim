"""Image validation and preprocessing (README: robustness tests)."""

import io

import numpy as np
import pytest
from PIL import Image, ImageFilter

from src.utils import (
    ImageValidationError,
    estimate_blur,
    load_image,
    preprocess_image,
    validate_image,
)


def _png_bytes(size=(200, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 40, 40)).save(buf, format="PNG")
    return buf.getvalue()


def test_valid_image_passes(cfg, rgb_image):
    assert validate_image(rgb_image, filename="car.jpg", size_bytes=1000, cfg=cfg).ok


def test_none_is_rejected(cfg):
    res = validate_image(None, cfg=cfg)
    assert not res.ok and "No image" in res.message


def test_wrong_extension_is_rejected(cfg):
    res = validate_image(_png_bytes(), filename="notes.pdf", cfg=cfg)
    assert not res.ok and "Unsupported file type" in res.message


def test_empty_file_is_rejected(cfg):
    res = validate_image(b"", filename="car.jpg", size_bytes=0, cfg=cfg)
    assert not res.ok and "empty" in res.message.lower()


def test_oversized_file_is_rejected(cfg):
    too_big = (cfg["data"]["max_upload_mb"] + 1) * 1024 * 1024
    res = validate_image(_png_bytes(), filename="car.png", size_bytes=too_big, cfg=cfg)
    assert not res.ok and "too large" in res.message.lower()


def test_corrupted_bytes_are_rejected(cfg):
    res = validate_image(b"this is definitely not an image", filename="car.jpg", cfg=cfg)
    assert not res.ok and "corrupted" in res.message.lower()


def test_tiny_image_is_rejected(cfg):
    res = validate_image(_png_bytes((20, 20)), filename="car.png", cfg=cfg)
    assert not res.ok and "too small" in res.message.lower()


def test_excessive_pixel_count_is_rejected(cfg):
    cfg["data"]["max_image_pixels"] = 1_000
    res = validate_image(_png_bytes((40, 40)), filename="car.png", cfg=cfg)
    assert not res.ok and "dimensions are too large" in res.message.lower()


def test_load_image_raises_on_invalid(cfg):
    with pytest.raises(ImageValidationError):
        load_image(b"garbage", cfg=cfg)


def test_preprocess_shape_and_range(cfg, rgb_image):
    batch = preprocess_image(rgb_image, cfg)
    h, w = cfg["data"]["image_size"]
    assert batch.shape == (1, h, w, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= batch.min() and batch.max() <= 255.0


def test_grayscale_and_rgba_are_converted(cfg):
    gray = Image.new("L", (128, 128), 100)
    rgba = Image.new("RGBA", (128, 128), (10, 20, 30, 255))
    assert preprocess_image(gray, cfg).shape[-1] == 3
    assert preprocess_image(rgba, cfg).shape[-1] == 3


def test_blur_score_is_lower_for_blurred_image(rgb_image):
    sharp = estimate_blur(rgb_image)
    blurred = estimate_blur(rgb_image.filter(ImageFilter.GaussianBlur(8)))
    assert blurred < sharp
