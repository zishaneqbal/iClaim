import numpy as np
import pytest
from PIL import Image

from src.vehicle_gate import (
    check_distribution_ood,
    compute_distribution_metrics,
    evaluate_claim_validity,
    verify_vehicle,
)


def test_compute_distribution_metrics_sharp():
    probs = {"scratch": 0.95, "dent": 0.02, "crack": 0.01, "broken_glass": 0.01, "major_body_damage": 0.01}
    m = compute_distribution_metrics(probs)
    assert m["max_confidence"] == 0.95
    assert m["norm_entropy"] < 0.35
    assert m["margin"] > 0.80


def test_compute_distribution_metrics_diffuse():
    probs = {"scratch": 0.20, "dent": 0.20, "crack": 0.20, "broken_glass": 0.20, "major_body_damage": 0.20}
    m = compute_distribution_metrics(probs)
    assert m["max_confidence"] == 0.20
    assert m["norm_entropy"] > 0.95
    assert m["margin"] == 0.0


def test_check_distribution_ood_pass_and_fail(cfg):
    sharp = {"scratch": 0.85, "dent": 0.05, "crack": 0.04, "broken_glass": 0.03, "major_body_damage": 0.03}
    res_sharp = check_distribution_ood(sharp, cfg=cfg)
    assert res_sharp["passed"] is True
    assert len(res_sharp["reasons"]) == 0

    diffuse = {"scratch": 0.22, "dent": 0.21, "crack": 0.20, "broken_glass": 0.19, "major_body_damage": 0.18}
    res_diffuse = check_distribution_ood(diffuse, cfg=cfg)
    assert res_diffuse["passed"] is False
    assert len(res_diffuse["reasons"]) > 0


def test_verify_vehicle_disabled(cfg):
    custom_cfg = dict(cfg)
    custom_cfg["vehicle_verification"] = {"enabled": False}
    img = Image.new("RGB", (100, 100), color=(100, 100, 100))
    res = verify_vehicle(img, cfg=custom_cfg)
    assert res["passed"] is True
    assert res["is_vehicle"] is True


def test_verify_vehicle_on_non_vehicle_image(cfg):
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    res = verify_vehicle(img, cfg=cfg)
    assert res["passed"] is False
    assert res["is_vehicle"] is False
    assert "detected_object" in res


def test_evaluate_claim_validity_rejection(cfg):
    img = Image.new("RGB", (64, 64), color=(0, 255, 0))
    diffuse = [0.2, 0.2, 0.2, 0.2, 0.2]
    res = evaluate_claim_validity(img, diffuse, cfg=cfg)
    assert res["is_valid_claim"] is False
    assert res["status"] in ("REJECTED_NON_VEHICLE", "REJECTED_OOD")
    assert "gatekeeper" in res
    assert "ood_metrics" in res


def test_spider_web_in_vehicle_classes():
    from src.vehicle_gate import VEHICLE_IMAGENET_CLASSES, AUTOMOTIVE_FRIENDLY_NAMES
    assert "spider_web" in VEHICLE_IMAGENET_CLASSES
    assert "spider_web" in AUTOMOTIVE_FRIENDLY_NAMES
    assert "seat_belt" in VEHICLE_IMAGENET_CLASSES
    assert "radiator" in VEHICLE_IMAGENET_CLASSES


def test_closeup_vehicle_panel_fallback(cfg):
    img = Image.new("RGB", (64, 64), color=(80, 80, 80))
    sharp_damage = {"scratch": 0.05, "dent": 0.85, "crack": 0.04, "broken_glass": 0.03, "major_body_damage": 0.03}
    res = evaluate_claim_validity(img, sharp_damage, cfg=cfg)
    assert res["is_valid_claim"] is True
    assert res["status"] == "VERIFIED_VEHICLE"

