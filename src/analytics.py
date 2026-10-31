"""Insurance claim analytics: priority rules, indicative cost ranges and aggregations.

CLI (create the synthetic sample data that ships with the project):
    python -m src.analytics --generate

IMPORTANT: repair-cost ranges are indicative assumptions from config.yaml, not real
insurer data, and the sample claims are SYNTHETIC. Every record carries a
`data_source` column ("synthetic" or "user") so this is never hidden.
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import get_logger, load_config, resolve_path

logger = get_logger(__name__)

CLAIM_COLUMNS = [
    "claim_id", "claim_date", "vehicle_type", "policy_type",
    "damage_class", "severity", "confidence", "priority",
    "est_cost_low", "est_cost_high", "claimed_amount", "amount_flag", "data_source",
]

SEVERITY_ORDER = ["minor", "moderate", "severe"]
_CLAIM_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")


# --------------------------------------------------------------------------- rules
def priority_for(severity: str, confidence: float, cfg: dict | None = None) -> str:
    """Claim-processing priority.

    Low-confidence predictions are ALWAYS routed to manual review, regardless of severity.
    """
    cfg = cfg or load_config()
    a = cfg["analytics"]
    if confidence < a["min_confidence"]:
        return a["priority_rules"]["low_confidence"]
    if severity not in a["priority_rules"]:
        raise ValueError(f"Unknown severity '{severity}'")
    return a["priority_rules"][severity]


def estimate_cost_range(
    damage_class: str,
    severity: str,
    cfg: dict | None = None,
    vehicle_type: str | None = None,
) -> Tuple[float, float]:
    """Indicative (low, high) repair cost for a damage class and severity, optionally scaled by vehicle segment."""
    cfg = cfg or load_config()
    try:
        low, high = cfg["analytics"]["cost_ranges_inr"][damage_class][severity]
    except KeyError as exc:
        raise ValueError(f"No cost range for {damage_class}/{severity}") from exc
    low, high = float(low), float(high)
    if vehicle_type:
        multipliers = cfg.get("analytics", {}).get("vehicle_multipliers", {})
        mult = float(multipliers.get(vehicle_type, 1.0))
        low = round(low * mult, -2)
        high = round(high * mult, -2)
    return low, high


def amount_flag(claimed: Optional[float], low: float, high: float, tolerance: float = 1.25) -> str:
    """Compare a claimed amount with the indicative range.

    Returns 'no_amount', 'within_range', 'below_range' or 'above_range'.
    'above_range' means the claim exceeds the indicative high estimate (+25%): review.
    """
    if claimed is None or (isinstance(claimed, float) and np.isnan(claimed)) or claimed <= 0:
        return "no_amount"
    if claimed > high * tolerance:
        return "above_range"
    if claimed < low / tolerance:
        return "below_range"
    return "within_range"


def validate_claim_id(claim_id: Optional[str]) -> Optional[str]:
    """Return a safe claim identifier, rejecting CSV-formula and control characters."""
    if claim_id is None or not str(claim_id).strip():
        return None
    value = str(claim_id).strip()
    if not _CLAIM_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "Claim ID must use 1-64 letters, numbers, hyphens, or underscores."
        )
    return value


def build_claim_record(
    claim_info: dict,
    prediction: dict,
    cfg: dict | None = None,
    data_source: str = "user",
) -> dict:
    """Merge claim details with a predict() result into one analytics record."""
    cfg = cfg or load_config()
    damage = prediction["damage_class"]
    severity = prediction["severity"]
    conf = float(prediction["class_confidence"])
    v_type = claim_info.get("vehicle_type", "Unknown")
    low, high = estimate_cost_range(damage, severity, cfg, vehicle_type=v_type)
    claimed = claim_info.get("claimed_amount")
    claimed = float(claimed) if claimed not in (None, "") else None
    return {
        "claim_id": validate_claim_id(claim_info.get("claim_id")) or f"CLM-{pd.Timestamp.now():%Y%m%d%H%M%S}",
        "claim_date": str(claim_info.get("claim_date") or date.today()),
        "vehicle_type": claim_info.get("vehicle_type", "Unknown"),
        "policy_type": claim_info.get("policy_type", "Unknown"),
        "damage_class": damage,
        "severity": severity,
        "confidence": round(conf, 4),
        "priority": priority_for(severity, conf, cfg),
        "est_cost_low": low,
        "est_cost_high": high,
        "claimed_amount": claimed,
        "amount_flag": amount_flag(claimed, low, high),
        "data_source": data_source,
    }


# ----------------------------------------------------------------------- persistence
def append_prediction_log(record: dict, cfg: dict | None = None) -> Path:
    """Append one record to the predictions log CSV (created on first use)."""
    cfg = cfg or load_config()
    path = resolve_path(cfg["paths"]["predictions_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{c: record.get(c) for c in CLAIM_COLUMNS}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def load_claims(cfg: dict | None = None, include_log: bool = True) -> pd.DataFrame:
    """Load the sample claims plus any claims analysed through the app."""
    cfg = cfg or load_config()
    frames = []
    for key in ("claims_csv", "predictions_log") if include_log else ("claims_csv",):
        path = resolve_path(cfg["paths"][key])
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(columns=CLAIM_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")
    return df.drop_duplicates(subset="claim_id", keep="last").reset_index(drop=True)


# ----------------------------------------------------------------------- aggregations
def damage_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Count and share of claims per damage class."""
    if df.empty:
        return pd.DataFrame(columns=["damage_class", "claims", "share_pct"])
    out = df["damage_class"].value_counts().rename_axis("damage_class").reset_index(name="claims")
    out["share_pct"] = (out["claims"] / out["claims"].sum() * 100).round(1)
    return out


def severity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count and share of claims per severity level (ordered minor -> severe)."""
    counts = df["severity"].value_counts() if not df.empty else pd.Series(dtype=int)
    out = pd.DataFrame({"severity": SEVERITY_ORDER})
    out["claims"] = out["severity"].map(counts).fillna(0).astype(int)
    total = max(out["claims"].sum(), 1)
    out["share_pct"] = (out["claims"] / total * 100).round(1)
    return out


def cost_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Average indicative repair-cost range per damage class and severity."""
    if df.empty:
        return pd.DataFrame(columns=["damage_class", "severity", "claims", "avg_low", "avg_high"])
    g = df.groupby(["damage_class", "severity"]).agg(
        claims=("claim_id", "count"),
        avg_low=("est_cost_low", "mean"),
        avg_high=("est_cost_high", "mean"),
    )
    return g.round(0).reset_index()


def priority_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Count of claims per processing priority."""
    if df.empty:
        return pd.DataFrame(columns=["priority", "claims"])
    return df["priority"].value_counts().rename_axis("priority").reset_index(name="claims")


def vehicle_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count, share, and average estimated exposure per vehicle type."""
    if df.empty or "vehicle_type" not in df:
        return pd.DataFrame(columns=["vehicle_type", "claims", "share_pct", "avg_exposure"])
    temp = df.copy()
    temp["mid_cost"] = (temp["est_cost_low"] + temp["est_cost_high"]) / 2
    g = temp.groupby("vehicle_type").agg(
        claims=("claim_id", "count"),
        avg_exposure=("mid_cost", "mean"),
    ).reset_index()
    g["share_pct"] = (g["claims"] / g["claims"].sum() * 100).round(1)
    g["avg_exposure"] = g["avg_exposure"].round(0)
    return g.sort_values("claims", ascending=False).reset_index(drop=True)


def timeline_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly aggregated count and total exposure."""
    if df.empty or "claim_date" not in df:
        return pd.DataFrame(columns=["period", "claims", "exposure"])
    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["claim_date"], errors="coerce")
    temp = temp.dropna(subset=["date"])
    if temp.empty:
        return pd.DataFrame(columns=["period", "claims", "exposure"])
    temp["period"] = temp["date"].dt.to_period("W").dt.start_time
    temp["mid_cost"] = (temp["est_cost_low"] + temp["est_cost_high"]) / 2
    out = temp.groupby("period").agg(
        claims=("claim_id", "count"),
        exposure=("mid_cost", "sum"),
    ).reset_index().sort_values("period")
    return out


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Headline numbers for the dashboard."""
    if df.empty:
        return {
            "total_claims": 0, "pct_severe": 0.0, "avg_confidence": 0.0,
            "manual_review": 0, "est_exposure_mid": 0.0, "above_range": 0,
        }
    mid = (df["est_cost_low"] + df["est_cost_high"]) / 2
    return {
        "total_claims": int(len(df)),
        "pct_severe": round(float((df["severity"] == "severe").mean() * 100), 1),
        "avg_confidence": round(float(df["confidence"].mean()), 3),
        "manual_review": int((df["priority"] == "Manual review required").sum()),
        "est_exposure_mid": float(mid.sum()),
        "above_range": int((df.get("amount_flag") == "above_range").sum()) if "amount_flag" in df else 0,
    }


# --------------------------------------------------------------------- synthetic data
def generate_synthetic_claims(
    n: int | None = None,
    cfg: dict | None = None,
    seed: int | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Create SYNTHETIC claims for demos. Not real insurer data.

    Damage mix, severity and confidence are drawn from simple distributions;
    costs are sampled inside the indicative range with some noise.
    """
    cfg = cfg or load_config()
    sc = cfg["synthetic_claims"]
    n = n or sc["n_claims"]
    rng = np.random.default_rng(seed if seed is not None else sc["seed"])
    classes = cfg["data"]["class_names"]
    end = pd.Timestamp(end_date or date.today())

    class_p = np.array([0.30, 0.28, 0.14, 0.14, 0.14])
    if len(classes) != len(class_p):
        class_p = np.ones(len(classes)) / len(classes)
    sev_p = {
        "scratch": [0.70, 0.25, 0.05],
        "dent": [0.35, 0.45, 0.20],
        "crack": [0.25, 0.50, 0.25],
        "broken_glass": [0.20, 0.45, 0.35],
        "major_body_damage": [0.05, 0.25, 0.70],
    }

    rows = []
    for i in range(n):
        dmg = str(rng.choice(classes, p=class_p))
        sev = str(rng.choice(SEVERITY_ORDER, p=sev_p.get(dmg, [1 / 3] * 3)))
        conf = float(np.clip(rng.beta(8, 2), 0.35, 0.995))
        vtype = str(rng.choice(sc["vehicle_types"]))
        low, high = estimate_cost_range(dmg, sev, cfg, vehicle_type=vtype)
        est_mid = rng.uniform(low, high)
        # Claimed amount: mostly near the estimate, occasionally inflated.
        factor = rng.lognormal(0.0, 0.18) * (rng.uniform(1.4, 2.0) if rng.random() < 0.07 else 1.0)
        claimed = round(est_mid * factor, -2)
        rows.append(
            {
                "claim_id": f"SYN-{i + 1:05d}",
                "claim_date": (end - pd.Timedelta(days=int(rng.integers(0, 180)))).date().isoformat(),
                "vehicle_type": vtype,
                "policy_type": str(rng.choice(sc["policy_types"])),
                "damage_class": dmg,
                "severity": sev,
                "confidence": round(conf, 4),
                "priority": priority_for(sev, conf, cfg),
                "est_cost_low": low,
                "est_cost_high": high,
                "claimed_amount": claimed,
                "amount_flag": amount_flag(claimed, low, high),
                "data_source": "synthetic",
            }
        )
    return pd.DataFrame(rows, columns=CLAIM_COLUMNS).sort_values("claim_date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim analytics utilities")
    parser.add_argument("--generate", action="store_true", help="write synthetic claims CSV")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.generate:
        df = generate_synthetic_claims(args.n, cfg)
        out = resolve_path(cfg["paths"]["claims_csv"])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Wrote %d SYNTHETIC claims to %s", len(df), out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
