"""Severity assessment (README Section 11, Option C: rule-based).

Most public vehicle-damage datasets have damage-type labels but no severity labels,
so severity is derived from interpretable signals:

    score = w_class * class_prior + w_area * area_component + w_conf * confidence

* class_prior   - typical seriousness of the damage type (config severity.class_prior)
* area_component- Grad-CAM damage extent, scaled to 0-1 (config severity.area_ratio_saturation)
* confidence    - model confidence in the predicted class (weak signal, small weight)

The score is then bucketed into minor / moderate / severe with config thresholds.

ASSUMPTION (state this in the report): severity is a heuristic, not a learned label.
Swap this module for a trained severity head if severity labels become available;
nothing else in the pipeline needs to change.
"""

from __future__ import annotations

from typing import Dict, Optional

from src.config import load_config

SEVERITY_LEVELS = ("minor", "moderate", "severe")


def compute_severity_score(
    damage_class: str,
    confidence: float,
    area_ratio: Optional[float] = None,
    cfg: dict | None = None,
) -> float:
    """Return a severity score in [0, 1].

    If `area_ratio` is None (e.g. Grad-CAM unavailable), its weight is folded into
    the class prior so the score stays on the same scale.
    """
    cfg = cfg or load_config()
    s = cfg["severity"]
    if damage_class not in s["class_prior"]:
        raise ValueError(f"Unknown damage class '{damage_class}'")

    prior = float(s["class_prior"][damage_class])
    w = s["weights"]
    conf = min(max(float(confidence), 0.0), 1.0)

    if area_ratio is None:
        score = (w["class_prior"] + w["area_ratio"]) * prior + w["confidence"] * conf
    else:
        saturation = float(s.get("area_ratio_saturation", 0.5)) or 0.5
        area_component = min(max(float(area_ratio), 0.0) / saturation, 1.0)
        score = w["class_prior"] * prior + w["area_ratio"] * area_component + w["confidence"] * conf

    total_weight = w["class_prior"] + w["area_ratio"] + w["confidence"]
    return float(min(max(score / total_weight, 0.0), 1.0))


def score_to_level(score: float, cfg: dict | None = None) -> str:
    """Bucket a severity score into minor / moderate / severe."""
    cfg = cfg or load_config()
    t = cfg["severity"]["thresholds"]
    if score <= t["minor_max"]:
        return "minor"
    if score <= t["moderate_max"]:
        return "moderate"
    return "severe"


def assess_severity(
    damage_class: str,
    confidence: float,
    area_ratio: Optional[float] = None,
    cfg: dict | None = None,
) -> Dict[str, object]:
    """Assess severity and return {'level', 'score', 'method', 'used_area'}."""
    cfg = cfg or load_config()
    score = compute_severity_score(damage_class, confidence, area_ratio, cfg)
    return {
        "level": score_to_level(score, cfg),
        "score": round(score, 4),
        "method": cfg["severity"]["method"],
        "used_area": area_ratio is not None,
    }
