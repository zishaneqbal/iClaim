"""Priority rules, cost ranges, aggregations and synthetic data."""

from datetime import date

import pandas as pd
import pytest

from src import analytics


def test_low_confidence_always_manual_review(cfg):
    low = cfg["analytics"]["min_confidence"] - 0.01
    for sev in analytics.SEVERITY_ORDER:
        assert analytics.priority_for(sev, low, cfg) == "Manual review required"


def test_priority_by_severity(cfg):
    assert analytics.priority_for("severe", 0.95, cfg) == "High"
    assert analytics.priority_for("moderate", 0.95, cfg) == "Medium"
    assert analytics.priority_for("minor", 0.95, cfg) == "Low"


def test_priority_boundary_confidence(cfg):
    exactly = cfg["analytics"]["min_confidence"]
    assert analytics.priority_for("minor", exactly, cfg) == "Low"


def test_unknown_severity_raises(cfg):
    with pytest.raises(ValueError):
        analytics.priority_for("catastrophic", 0.9, cfg)


def test_cost_range_is_ordered(cfg):
    for cls in cfg["data"]["class_names"]:
        for sev in analytics.SEVERITY_ORDER:
            low, high = analytics.estimate_cost_range(cls, sev, cfg)
            assert 0 < low < high


def test_cost_range_increases_with_severity(cfg):
    for cls in cfg["data"]["class_names"]:
        highs = [analytics.estimate_cost_range(cls, s, cfg)[1] for s in analytics.SEVERITY_ORDER]
        assert highs == sorted(highs)


def test_cost_range_unknown_raises(cfg):
    with pytest.raises(ValueError):
        analytics.estimate_cost_range("rust", "minor", cfg)


@pytest.mark.parametrize(
    "claimed,expected",
    [(None, "no_amount"), (0, "no_amount"), (5000, "within_range"), (100000, "above_range"), (100, "below_range")],
)
def test_amount_flag(claimed, expected):
    assert analytics.amount_flag(claimed, 4000, 8000) == expected


@pytest.mark.parametrize("claim_id", ["CLM-2026_0142", "ABC123", None, "  "])
def test_safe_claim_id_is_accepted(claim_id):
    expected = claim_id.strip() if isinstance(claim_id, str) and claim_id.strip() else None
    assert analytics.validate_claim_id(claim_id) == expected


@pytest.mark.parametrize("claim_id", ["=SUM(A1:A2)", "+cmd", "CLM 1", "x/y", "a" * 65])
def test_unsafe_claim_id_is_rejected(claim_id):
    with pytest.raises(ValueError, match="Claim ID"):
        analytics.validate_claim_id(claim_id)


def test_build_claim_record(cfg):
    pred = {"damage_class": "dent", "severity": "moderate", "class_confidence": 0.9}
    rec = analytics.build_claim_record(
        {"claim_id": "X1", "claim_date": date(2026, 1, 5), "vehicle_type": "SUV", "claimed_amount": 12000},
        pred,
        cfg,
    )
    assert set(analytics.CLAIM_COLUMNS) <= set(rec)
    assert rec["priority"] == "Medium"
    assert rec["data_source"] == "user"
    assert rec["amount_flag"] == "within_range"


def test_synthetic_claims_shape_and_label(cfg):
    df = analytics.generate_synthetic_claims(50, cfg, seed=1, end_date=date(2026, 6, 1))
    assert len(df) == 50
    assert list(df.columns) == analytics.CLAIM_COLUMNS
    assert (df["data_source"] == "synthetic").all()
    assert df["claim_id"].is_unique
    assert df["damage_class"].isin(cfg["data"]["class_names"]).all()


def test_synthetic_claims_are_reproducible(cfg):
    a = analytics.generate_synthetic_claims(40, cfg, seed=7, end_date=date(2026, 6, 1))
    b = analytics.generate_synthetic_claims(40, cfg, seed=7, end_date=date(2026, 6, 1))
    pd.testing.assert_frame_equal(a, b)


@pytest.fixture()
def claims(cfg) -> pd.DataFrame:
    df = analytics.generate_synthetic_claims(120, cfg, seed=3, end_date=date(2026, 6, 1))
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    return df


def test_frequency_sums_to_total(claims):
    freq = analytics.damage_frequency(claims)
    assert freq["claims"].sum() == len(claims)
    assert abs(freq["share_pct"].sum() - 100) < 0.5


def test_severity_distribution_has_all_levels(claims):
    dist = analytics.severity_distribution(claims)
    assert list(dist["severity"]) == analytics.SEVERITY_ORDER
    assert dist["claims"].sum() == len(claims)


def test_priority_breakdown_total(claims):
    assert analytics.priority_breakdown(claims)["claims"].sum() == len(claims)


def test_cost_summary_ranges_are_ordered(claims):
    cs = analytics.cost_summary(claims)
    assert (cs["avg_low"] < cs["avg_high"]).all()


def test_kpis(claims):
    k = analytics.compute_kpis(claims)
    assert k["total_claims"] == len(claims)
    assert 0 <= k["pct_severe"] <= 100
    assert 0 < k["avg_confidence"] <= 1
    assert k["est_exposure_mid"] > 0


def test_empty_inputs_are_safe():
    empty = pd.DataFrame(columns=analytics.CLAIM_COLUMNS)
    assert analytics.damage_frequency(empty).empty
    assert analytics.severity_distribution(empty)["claims"].sum() == 0
    assert analytics.compute_kpis(empty)["total_claims"] == 0


def test_log_round_trip(cfg, tmp_path):
    cfg["paths"]["predictions_log"] = str(tmp_path / "log.csv")
    cfg["paths"]["claims_csv"] = str(tmp_path / "missing.csv")
    pred = {"damage_class": "scratch", "severity": "minor", "class_confidence": 0.88}
    rec = analytics.build_claim_record({"claim_id": "LOG-1"}, pred, cfg)
    analytics.append_prediction_log(rec, cfg)
    analytics.append_prediction_log({**rec, "claim_id": "LOG-2"}, cfg)
    loaded = analytics.load_claims(cfg)
    assert set(loaded["claim_id"]) == {"LOG-1", "LOG-2"}
