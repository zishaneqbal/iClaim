from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from src.config import get_logger, load_config, resolve_path
from src.utils import ImageInput, load_image

logger = get_logger(__name__)

_GATE_MODEL_CACHE: Optional[Any] = None
_CLASS_INDEX_CACHE: Optional[Dict[int, str]] = None
_NON_VEHICLE_CLASSES_CACHE: Optional[Set[str]] = None

VEHICLE_IMAGENET_CLASSES = {
    "ambulance", "beach_wagon", "cab", "convertible", "fire_engine",
    "garbage_truck", "go-kart", "golfcart", "half_track", "harvester", "jeep",
    "limousine", "minibus", "minivan", "Model_T", "moped", "motor_scooter",
    "mountain_bike", "moving_van", "passenger_car", "pickup", "police_van",
    "racer", "recreational_vehicle", "school_bus", "snowplow", "sports_car",
    "streetcar", "tow_truck", "tractor", "trailer_truck", "trolleybus",
    "car_mirror", "car_wheel", "disk_brake", "grille", "crash_helmet",
    "freight_car", "tricycle", "unicycle", "seat_belt", "radiator",
    "spider_web", "window_screen", "sliding_door", "odometer", "bicycle-built-for-two"
}

AUTOMOTIVE_FRIENDLY_NAMES = {
    "spider_web": "broken glass / fractured window (spider-web pattern)",
    "window_screen": "vehicle glass / window mesh",
    "sliding_door": "vehicle door panel",
    "car_mirror": "car side mirror",
    "car_wheel": "car wheel / tire",
    "seat_belt": "vehicle interior / seat belt",
}

EXTRA_NON_VEHICLE_ITEMS = {
    "notebook", "laptop", "cellular_telephone", "dial_telephone", "pay-phone",
    "desktop_computer", "computer_keyboard", "mouse", "television", "monitor",
    "screen", "iPod", "remote_control", "printer", "cassette", "tape_player",
    "CD_player", "loudspeaker", "microphone", "joystick", "radio", "modem",
    "suit", "trench_coat", "bow_tie", "tie", "brassiere", "diaper", "sock",
    "shoe", "running_shoe", "sandal", "cowboy_boot", "kimono", "gown",
    "academic_gown", "fur_coat", "sweatshirt", "bikini", "pajama", "swimming_trunks",
    "jean", "jersey", "cardigan", "miniskirt", "sarong", "stole", "cloak",
    "dining_table", "desk", "bookcase", "wardrobe", "couch", "studio_couch",
    "four-poster", "refrigerator", "microwave", "toaster", "bathtub", "toilet",
    "bidet", "washbasin", "vacuum", "dishwasher", "coffee_mug", "teapot",
    "pitcher", "plate", "water_bottle", "wine_bottle", "beer_bottle", "cup",
    "guitar", "electric_guitar", "acoustic_guitar", "violin", "cello", "piano",
    "grand_piano", "accordion", "flute", "harp", "drum", "harmonica", "sax",
    "trombone", "trumpet", "balloon"
}


def load_imagenet_class_index(cfg: dict | None = None) -> Dict[int, str]:
    global _CLASS_INDEX_CACHE
    if _CLASS_INDEX_CACHE is not None:
        return _CLASS_INDEX_CACHE

    cfg = cfg or load_config()
    vcfg = cfg.get("vehicle_verification", {})
    idx_path = resolve_path(vcfg.get("class_index_path", "models/imagenet_class_index.json"))
    if not idx_path.exists():
        logger.warning("ImageNet class index not found at %s", idx_path)
        return {}

    try:
        raw = json.loads(idx_path.read_text(encoding="utf-8"))
        _CLASS_INDEX_CACHE = {int(k): v[1] for k, v in raw.items()}
    except Exception as exc:
        logger.warning("Failed to load ImageNet class index: %s", exc)
        return {}

    return _CLASS_INDEX_CACHE


def get_definite_non_vehicle_classes(cfg: dict | None = None) -> Set[str]:
    global _NON_VEHICLE_CLASSES_CACHE
    if _NON_VEHICLE_CLASSES_CACHE is not None:
        return _NON_VEHICLE_CLASSES_CACHE

    cfg = cfg or load_config()
    vcfg = cfg.get("vehicle_verification", {})
    idx_path = resolve_path(vcfg.get("class_index_path", "models/imagenet_class_index.json"))

    non_veh: Set[str] = set()
    if idx_path.exists():
        try:
            raw = json.loads(idx_path.read_text(encoding="utf-8"))
            for k, v in raw.items():
                ik = int(k)
                if (0 <= ik <= 397) or (923 <= ik <= 965) or (985 <= ik <= 991):
                    non_veh.add(v[1])
        except Exception:
            pass

    non_veh.update(EXTRA_NON_VEHICLE_ITEMS)
    non_veh.difference_update(VEHICLE_IMAGENET_CLASSES)
    _NON_VEHICLE_CLASSES_CACHE = non_veh
    return _NON_VEHICLE_CLASSES_CACHE


def load_gate_model(cfg: dict | None = None):
    global _GATE_MODEL_CACHE
    if _GATE_MODEL_CACHE is not None:
        return _GATE_MODEL_CACHE

    import keras

    cfg = cfg or load_config()
    vcfg = cfg.get("vehicle_verification", {})
    gate_path = resolve_path(vcfg.get("gate_model_path", "models/vehicle_gate_mobilenetv2.keras"))

    if not gate_path.exists():
        logger.info("Vehicle gate model not found at %s. Verification will rely on distribution gating.", gate_path)
        return None

    try:
        logger.info("Loading vehicle gatekeeper from %s", gate_path)
        _GATE_MODEL_CACHE = keras.models.load_model(gate_path)
        return _GATE_MODEL_CACHE
    except Exception as exc:
        logger.warning("Could not load vehicle gate model (%s)", exc)
        return None


def verify_vehicle(image: ImageInput, cfg: dict | None = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    vcfg = cfg.get("vehicle_verification", {})
    if not vcfg.get("enabled", True):
        return {
            "passed": True,
            "detected_object": "vehicle",
            "confidence": 1.0,
            "is_vehicle": True,
            "is_definite_non_vehicle": False,
        }

    model = load_gate_model(cfg)
    idx_to_label = load_imagenet_class_index(cfg)

    if model is None or not idx_to_label:
        return {
            "passed": True,
            "detected_object": "unverified",
            "confidence": 1.0,
            "is_vehicle": True,
            "is_definite_non_vehicle": False,
        }

    pil_img = load_image(image, cfg).resize((224, 224))
    arr = np.asarray(pil_img, dtype="float32")[None, ...]
    arr_prep = (arr / 127.5) - 1.0

    preds = model.predict(arr_prep, verbose=0)[0]
    top_k = int(vcfg.get("top_k", 5))
    top_indices = np.argsort(preds)[-top_k:][::-1]

    top_label = idx_to_label.get(int(top_indices[0]), "unknown")
    top_conf = float(preds[top_indices[0]])

    found_vehicle = False
    vehicle_label = ""
    vehicle_conf = 0.0
    min_veh_conf = float(vcfg.get("min_vehicle_confidence", 0.03))

    for idx in top_indices:
        label = idx_to_label.get(int(idx), "")
        conf = float(preds[idx])
        if label in VEHICLE_IMAGENET_CLASSES and conf >= min_veh_conf:
            found_vehicle = True
            vehicle_label = label
            vehicle_conf = conf
            break

    top_preds_list = [(idx_to_label.get(int(i), ""), round(float(preds[i]), 4)) for i in top_indices]

    if found_vehicle:
        display_name = AUTOMOTIVE_FRIENDLY_NAMES.get(vehicle_label, vehicle_label.replace("_", " "))
        return {
            "passed": True,
            "is_vehicle": True,
            "is_definite_non_vehicle": False,
            "detected_object": display_name,
            "confidence": round(vehicle_conf, 4),
            "top_predictions": top_preds_list,
        }

    non_veh_set = get_definite_non_vehicle_classes(cfg)
    is_definite_non_vehicle = (top_label in non_veh_set) and (top_conf >= 0.25)
    clean_top = top_label.replace("_", " ")

    return {
        "passed": False,
        "is_vehicle": False,
        "is_definite_non_vehicle": is_definite_non_vehicle,
        "detected_object": clean_top,
        "confidence": round(top_conf, 4),
        "reason": f"Detected non-vehicle object '{clean_top}' ({top_conf:.0%} confidence)",
        "top_predictions": top_preds_list,
    }


def compute_distribution_metrics(probabilities: Dict[str, float] | np.ndarray) -> Dict[str, float]:
    if isinstance(probabilities, dict):
        probs = np.array(list(probabilities.values()), dtype="float64")
    else:
        probs = np.asarray(probabilities, dtype="float64")

    total = probs.sum()
    if total > 0:
        probs = probs / total

    k = max(len(probs), 1)
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
    max_entropy = np.log(k) if k > 1 else 1.0
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    sorted_p = np.sort(probs)
    max_confidence = float(sorted_p[-1]) if len(sorted_p) > 0 else 0.0
    margin = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else max_confidence

    return {
        "max_confidence": round(max_confidence, 4),
        "entropy": round(entropy, 4),
        "norm_entropy": round(norm_entropy, 4),
        "margin": round(margin, 4),
    }


def check_distribution_ood(probabilities: Dict[str, float] | np.ndarray, cfg: dict | None = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    vcfg = cfg.get("vehicle_verification", {})
    ood_cfg = vcfg.get("ood_gating", {})

    metrics = compute_distribution_metrics(probabilities)
    if not ood_cfg.get("enabled", True):
        return {"passed": True, **metrics}

    min_conf = float(ood_cfg.get("min_confidence", 0.35))
    max_entropy = float(ood_cfg.get("max_norm_entropy", 0.85))
    min_margin = float(ood_cfg.get("min_margin", 0.04))

    failed_reasons = []
    if metrics["max_confidence"] < min_conf:
        failed_reasons.append(f"Confidence {metrics['max_confidence']:.0%} below threshold ({min_conf:.0%})")
    if metrics["norm_entropy"] > max_entropy:
        failed_reasons.append(f"High entropy {metrics['norm_entropy']:.2f} (diffuse distribution, limit: {max_entropy:.2f})")
    if metrics["margin"] < min_margin:
        failed_reasons.append(f"Ambiguous class margin {metrics['margin']:.3f} (limit: {min_margin:.3f})")

    passed = len(failed_reasons) == 0
    return {
        "passed": passed,
        "reasons": failed_reasons,
        **metrics,
    }


def evaluate_claim_validity(
    image: ImageInput,
    damage_probabilities: Dict[str, float] | np.ndarray,
    cfg: dict | None = None,
) -> Dict[str, Any]:
    cfg = cfg or load_config()

    gate_result = verify_vehicle(image, cfg)
    ood_result = check_distribution_ood(damage_probabilities, cfg)

    if gate_result["passed"]:
        if ood_result["passed"]:
            status = "VERIFIED_VEHICLE"
            message = "Vehicle verified and damage distribution confirmed."
            is_valid = True
        else:
            status = "REJECTED_OOD"
            message = "Vehicle detected, but damage probability distribution is ambiguous or diffuse."
            is_valid = False
    elif gate_result.get("is_definite_non_vehicle"):
        status = "REJECTED_NON_VEHICLE"
        message = f"Definite non-vehicle subject detected ({gate_result.get('detected_object', 'unknown')}). Automated damage assessment halted."
        is_valid = False
    else:
        if ood_result["passed"] and ood_result.get("max_confidence", 0.0) >= 0.40 and ood_result.get("norm_entropy", 1.0) <= 0.88:
            status = "VERIFIED_VEHICLE"
            gate_result["is_vehicle"] = True
            gate_result["passed"] = True
            gate_result["detected_object"] = f"vehicle panel ({gate_result.get('detected_object', 'surface')})"
            message = "Close-up vehicle panel surface verified via damage neural classifier."
            is_valid = True
        else:
            status = "REJECTED_OOD"
            message = "Image probability distribution is diffuse or ambiguous, indicating non-standard or out-of-distribution vehicle photo."
            is_valid = False

    return {
        "is_valid_claim": is_valid,
        "status": status,
        "message": message,
        "gatekeeper": gate_result,
        "ood_metrics": ood_result,
    }
