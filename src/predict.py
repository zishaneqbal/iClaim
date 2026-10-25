"""Inference entry point used by the Streamlit app, notebooks and tests.

STABLE CONTRACT - predict() always returns these four keys:

    {
      "damage_class": "dent",
      "class_confidence": 0.91,
      "severity": "moderate",
      "class_probabilities": {"scratch": 0.03, "dent": 0.91, ...}
    }

Additional keys (severity_score, damage_area_ratio, low_confidence, gradcam_overlay)
are additive and may be ignored by callers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src import model as _model_module  # noqa: F401  (registers custom Keras layers)
from src.config import get_logger, load_config, resolve_path
from src.severity import assess_severity
from src.utils import ImageInput, load_image, preprocess_image

logger = get_logger(__name__)

_MODEL_CACHE: Dict[str, Any] = {}


def class_names_for(cfg: dict | None = None) -> List[str]:
    """Class names, preferring models/class_names.json written during training."""
    cfg = cfg or load_config()
    path = resolve_path(cfg["paths"]["class_names_json"])
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read %s, falling back to config", path)
    return list(cfg["data"]["class_names"])


def model_available(cfg: dict | None = None) -> bool:
    """True if a trained model file exists."""
    cfg = cfg or load_config()
    return resolve_path(cfg["paths"]["best_model"]).exists()


def load_trained_model(path: str | Path | None = None, cfg: dict | None = None):
    """Load (and cache) the trained Keras model."""
    from tensorflow import keras

    cfg = cfg or load_config()
    model_path = Path(path) if path else resolve_path(cfg["paths"]["best_model"])
    if not model_path.is_absolute():
        model_path = resolve_path(str(model_path))
    key = str(model_path)
    if key not in _MODEL_CACHE:
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Train one with: python -m src.train"
            )
        logger.info("Loading model from %s", model_path)
        _MODEL_CACHE[key] = keras.models.load_model(model_path)
    return _MODEL_CACHE[key]


def predict(
    image: ImageInput,
    model=None,
    cfg: dict | None = None,
    class_names: Optional[List[str]] = None,
    use_gradcam: bool = True,
    return_overlay: bool = False,
) -> Dict[str, Any]:
    """Classify damage type and assess severity for one image.

    Args:
        image: Path, bytes, file-like object, PIL image or numpy array.
        model: Optional preloaded Keras model (defaults to the trained model).
        cfg: Optional config dict.
        class_names: Optional class list (defaults to training-time names).
        use_gradcam: Use Grad-CAM damage area as a severity signal.
        return_overlay: Also return a Grad-CAM overlay image (RGB uint8 array).

    Raises:
        ImageValidationError: if the image is invalid.
        FileNotFoundError: if no trained model exists.
    """
    cfg = cfg or load_config()
    names = class_names or class_names_for(cfg)
    model = model if model is not None else load_trained_model(cfg=cfg)

    pil_image = load_image(image, cfg)  # validates
    batch = preprocess_image(pil_image, cfg)
    probs = np.asarray(model.predict(batch, verbose=0)[0], dtype="float64")
    if len(probs) != len(names):
        raise ValueError(
            f"Model outputs {len(probs)} classes but {len(names)} class names are configured."
        )

    idx = int(np.argmax(probs))
    damage_class = names[idx]
    confidence = float(probs[idx])

    area_ratio: Optional[float] = None
    overlay = None
    if use_gradcam or return_overlay:
        try:
            from src.gradcam import damage_area_ratio, make_gradcam_heatmap, overlay_heatmap

            heat = make_gradcam_heatmap(model, batch, idx)
            area_ratio = damage_area_ratio(heat, cfg["severity"]["gradcam_activation_threshold"])
            if return_overlay:
                overlay = overlay_heatmap(pil_image, heat)
        except Exception as exc:  # Grad-CAM must never break a prediction
            logger.warning("Grad-CAM unavailable (%s); using class prior for severity", exc)

    sev = assess_severity(damage_class, confidence, area_ratio, cfg)
    result: Dict[str, Any] = {
        "damage_class": damage_class,
        "class_confidence": round(confidence, 4),
        "severity": sev["level"],
        "class_probabilities": {n: round(float(p), 4) for n, p in zip(names, probs)},
        "severity_score": sev["score"],
        "damage_area_ratio": None if area_ratio is None else round(area_ratio, 4),
        "low_confidence": confidence < cfg["analytics"]["min_confidence"],
    }
    if return_overlay:
        result["gradcam_overlay"] = overlay
    return result


def predict_batch(images: List[ImageInput], **kwargs) -> List[Dict[str, Any]]:
    """Run predict() over several images (same keyword arguments)."""
    return [predict(img, **kwargs) for img in images]
