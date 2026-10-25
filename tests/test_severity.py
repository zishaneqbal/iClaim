"""Rule-based severity logic."""

import pytest

from src.severity import SEVERITY_LEVELS, assess_severity, compute_severity_score, score_to_level


def test_levels_are_valid(cfg):
    for cls in cfg["data"]["class_names"]:
        out = assess_severity(cls, 0.9, 0.3, cfg)
        assert out["level"] in SEVERITY_LEVELS
        assert 0.0 <= out["score"] <= 1.0


def test_scratch_is_minor_and_major_body_is_severe(cfg):
    assert assess_severity("scratch", 0.9, 0.05, cfg)["level"] == "minor"
    assert assess_severity("major_body_damage", 0.9, 0.5, cfg)["level"] == "severe"


def test_score_increases_with_damage_area(cfg):
    low = compute_severity_score("dent", 0.9, 0.05, cfg)
    high = compute_severity_score("dent", 0.9, 0.60, cfg)
    assert high > low


def test_missing_area_still_works(cfg):
    out = assess_severity("crack", 0.8, None, cfg)
    assert out["used_area"] is False
    assert 0.0 <= out["score"] <= 1.0


def test_unknown_class_raises(cfg):
    with pytest.raises(ValueError):
        compute_severity_score("rust", 0.9, 0.1, cfg)


def test_threshold_boundaries(cfg):
    t = cfg["severity"]["thresholds"]
    assert score_to_level(t["minor_max"], cfg) == "minor"
    assert score_to_level(t["minor_max"] + 0.001, cfg) == "moderate"
    assert score_to_level(t["moderate_max"], cfg) == "moderate"
    assert score_to_level(t["moderate_max"] + 0.001, cfg) == "severe"


def test_confidence_is_clamped(cfg):
    assert compute_severity_score("dent", 5.0, 0.2, cfg) <= 1.0
    assert compute_severity_score("dent", -1.0, 0.2, cfg) >= 0.0
