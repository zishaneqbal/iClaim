"""Shared helpers: image validation, loading and preprocessing."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import get_logger, load_config

logger = get_logger(__name__)

ImageInput = Union[str, Path, bytes, BinaryIO, Image.Image, np.ndarray]


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation."""


@dataclass
class ValidationResult:
    ok: bool
    message: str = ""


def _to_pil(image: ImageInput, max_pixels: int | None = None) -> Image.Image:
    """Convert supported input to RGB, rejecting unsafe image dimensions before loading."""
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype("uint8")
        img = Image.fromarray(arr)
    elif isinstance(image, (str, Path)):
        img = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(image)))
    elif hasattr(image, "read"):
        data = image.read()
        if hasattr(image, "seek"):
            image.seek(0)
        img = Image.open(io.BytesIO(data))
    else:
        raise ImageValidationError(f"Unsupported image type: {type(image).__name__}")
    if max_pixels is not None and img.width * img.height > max_pixels:
        raise ImageValidationError(
            f"Image dimensions are too large ({img.width}x{img.height}). "
            f"Maximum is {max_pixels:,} pixels."
        )
    img.load()
    return img.convert("RGB")


def validate_image(
    image: ImageInput,
    filename: str | None = None,
    size_bytes: int | None = None,
    cfg: dict | None = None,
) -> ValidationResult:
    """Validate an uploaded image (type, size, readability, minimum resolution).

    Returns a ValidationResult with a friendly message when invalid.
    """
    cfg = cfg or load_config()
    dcfg = cfg["data"]

    if image is None:
        return ValidationResult(False, "No image was provided.")

    if filename:
        ext = Path(filename).suffix.lower()
        if ext not in dcfg["allowed_extensions"]:
            return ValidationResult(
                False,
                f"Unsupported file type '{ext}'. Allowed: {', '.join(dcfg['allowed_extensions'])}.",
            )

    if size_bytes is not None:
        if size_bytes == 0:
            return ValidationResult(False, "The uploaded file is empty.")
        if size_bytes > dcfg["max_upload_mb"] * 1024 * 1024:
            return ValidationResult(
                False, f"File is too large. Maximum size is {dcfg['max_upload_mb']} MB."
            )

    try:
        img = _to_pil(image, max_pixels=int(dcfg["max_image_pixels"]))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError):
        return ValidationResult(False, "The file is corrupted or is not a valid image.")
    except ImageValidationError as exc:
        return ValidationResult(False, str(exc))

    w, h = img.size
    min_side = dcfg["min_image_side_px"]
    if min(w, h) < min_side:
        return ValidationResult(
            False, f"Image is too small ({w}x{h}). Minimum side is {min_side}px."
        )

    return ValidationResult(True, "OK")


def load_image(image: ImageInput, cfg: dict | None = None) -> Image.Image:
    """Load an image as RGB PIL image, raising ImageValidationError if invalid."""
    result = validate_image(image, cfg=cfg)
    if not result.ok:
        raise ImageValidationError(result.message)
    return _to_pil(image, max_pixels=int(cfg["data"]["max_image_pixels"]))


def preprocess_image(image: ImageInput, cfg: dict | None = None) -> np.ndarray:
    """Resize to model input size and return a float32 batch of shape (1, H, W, 3).

    Pixel values stay in the 0-255 range. Backbone-specific scaling
    (e.g. MobileNetV2 preprocess_input) is applied inside the model itself.
    """
    cfg = cfg or load_config()
    h, w = cfg["data"]["image_size"]
    img = load_image(image, cfg).resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype="float32")
    return np.expand_dims(arr, axis=0)


def estimate_blur(image: ImageInput) -> float:
    """Return a sharpness score (variance of Laplacian). Lower means blurrier."""
    import cv2

    img = _to_pil(image)
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def format_inr(value: float) -> str:
    """Format a number as Indian-rupee text, e.g. 'INR 12,500'."""
    return f"INR {int(round(value)):,}"


def format_range_inr(low: float, high: float) -> str:
    """Format a cost range as text."""
    return f"{format_inr(low)} - {format_inr(high)}"


def split_counts(n: int, train: float, val: float) -> Tuple[int, int, int]:
    """Return (n_train, n_val, n_test) that sum to n."""
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    n_test = n - n_train - n_val
    return n_train, n_val, max(n_test, 0)


def to_jsonable(obj: Any) -> Any:
    """Convert numpy types to plain Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
