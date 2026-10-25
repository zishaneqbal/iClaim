"""Dataset preparation (audit, dedupe, stratified split) and tf.data pipelines.

CLI:
    python -m src.data_loader --audit      # print class counts / image stats
    python -m src.data_loader --prepare    # create train/val/test folders
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config import get_logger, load_config, resolve_path, set_seed

logger = get_logger(__name__)

SPLITS = ("train", "val", "test")


def _canonical_class(folder_name: str, cfg: dict) -> str | None:
    """Map a raw folder name to a configured class name (or None if unknown)."""
    classes = cfg["data"]["class_names"]
    aliases = {k.lower(): v for k, v in (cfg["data"].get("folder_aliases") or {}).items()}
    key = folder_name.strip().lower()
    if key in aliases:
        return aliases[key]
    normalised = key.replace(" ", "_").replace("-", "_")
    return normalised if normalised in classes else None


def list_raw_images(raw_dir: Path, cfg: dict) -> Dict[str, List[Path]]:
    """Return {class_name: [image paths]} from data/raw/<class_folder>/..."""
    exts = {e.lower() for e in cfg["data"]["allowed_extensions"]}
    found: Dict[str, List[Path]] = {c: [] for c in cfg["data"]["class_names"]}
    if not raw_dir.exists():
        return found
    for folder in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        cls = _canonical_class(folder.name, cfg)
        if cls is None:
            logger.warning("Skipping unknown folder '%s' (add it to data.folder_aliases)", folder.name)
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                found[cls].append(path)
    return found


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def audit_dataset(cfg: dict | None = None) -> pd.DataFrame:
    """Audit raw data: counts per class, duplicates, unreadable images.

    Returns a DataFrame with one row per class. Also logs a summary.
    """
    from PIL import Image, UnidentifiedImageError

    cfg = cfg or load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_data"])
    images = list_raw_images(raw_dir, cfg)

    rows = []
    seen: Dict[str, Path] = {}
    for cls, paths in images.items():
        dup = bad = 0
        sizes = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    sizes.append(im.size)
            except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
                bad += 1
                continue
            digest = _md5(p)
            if digest in seen:
                dup += 1
            else:
                seen[digest] = p
        rows.append(
            {
                "class": cls,
                "images": len(paths),
                "duplicates": dup,
                "unreadable": bad,
                "median_width": int(pd.Series([s[0] for s in sizes]).median()) if sizes else 0,
                "median_height": int(pd.Series([s[1] for s in sizes]).median()) if sizes else 0,
            }
        )
    df = pd.DataFrame(rows)
    total = int(df["images"].sum()) if not df.empty else 0
    logger.info("Audit complete: %d images across %d classes\n%s", total, len(df), df.to_string(index=False))
    if total:
        ratio = df["images"].max() / max(df["images"].replace(0, pd.NA).dropna().min(), 1)
        if ratio > 2:
            logger.warning("Class imbalance ratio %.1f:1 - enable training.use_class_weights.", ratio)
    return df


def prepare_splits(cfg: dict | None = None, overwrite: bool = False) -> pd.DataFrame:
    """Create stratified train/val/test folders under data/processed.

    Steps: collect images -> drop exact duplicates (prevents train/test leakage) ->
    shuffle with a fixed seed -> split per class -> copy files -> write a manifest.
    """
    cfg = cfg or load_config()
    seed = set_seed(cfg["project"]["seed"])
    rng = random.Random(seed)
    raw_dir = resolve_path(cfg["paths"]["raw_data"])
    out_dir = resolve_path(cfg["paths"]["processed_data"])
    split_cfg = cfg["data"]["split"]
    if abs(sum(split_cfg.values()) - 1.0) > 1e-6:
        raise ValueError("data.split fractions must sum to 1.0")

    images = list_raw_images(raw_dir, cfg)
    if sum(len(v) for v in images.values()) == 0:
        raise FileNotFoundError(
            f"No images found in {raw_dir}. Follow docs/DATASET_GUIDE.md to download the dataset."
        )

    already = any((out_dir / s).exists() and any((out_dir / s).iterdir()) for s in SPLITS)
    if already and not overwrite:
        raise FileExistsError(
            f"{out_dir} already contains splits. Re-run with --overwrite to rebuild them."
        )
    if already and overwrite:
        for s in SPLITS:
            shutil.rmtree(out_dir / s, ignore_errors=True)

    manifest = []
    seen_hashes = set()
    for cls, paths in images.items():
        unique: List[Path] = []
        for p in sorted(paths):
            digest = _md5(p)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            unique.append(p)
        rng.shuffle(unique)
        n = len(unique)
        n_train = int(round(n * split_cfg["train"]))
        n_val = int(round(n * split_cfg["val"]))
        buckets = {
            "train": unique[:n_train],
            "val": unique[n_train : n_train + n_val],
            "test": unique[n_train + n_val :],
        }
        for split, items in buckets.items():
            dest = out_dir / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(items):
                target = dest / f"{cls}_{split}_{i:05d}{src.suffix.lower()}"
                shutil.copy2(src, target)
                manifest.append({"split": split, "class": cls, "source": str(src), "file": str(target)})

    df = pd.DataFrame(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "split_manifest.csv", index=False)
    summary = df.groupby(["split", "class"]).size().unstack(fill_value=0)
    logger.info("Splits created in %s\n%s", out_dir, summary.to_string())
    return df


def build_augmentation(cfg: dict | None = None):
    """Keras augmentation pipeline. Apply to the TRAINING set only."""
    from tensorflow import keras

    cfg = cfg or load_config()
    a = cfg["data"]["augmentation"]
    layers = []
    if a.get("horizontal_flip"):
        layers.append(keras.layers.RandomFlip("horizontal"))
    layers.append(keras.layers.RandomRotation(a["rotation_factor"]))
    layers.append(keras.layers.RandomZoom(a["zoom_factor"]))
    layers.append(keras.layers.RandomContrast(a["contrast_factor"]))
    layers.append(keras.layers.RandomBrightness(a["brightness_factor"], value_range=(0, 255)))
    return keras.Sequential(layers, name="augmentation")


def load_datasets(cfg: dict | None = None, data_dir: str | Path | None = None):
    """Return (train_ds, val_ds, test_ds, class_names) as batched tf.data datasets.

    Augmentation is applied to train only. Images are float32 in 0-255 range;
    backbone-specific scaling happens inside the model.
    """
    import tensorflow as tf
    from tensorflow import keras

    cfg = cfg or load_config()
    seed = cfg["project"]["seed"]
    root = Path(data_dir) if data_dir else resolve_path(cfg["paths"]["processed_data"])
    class_names = cfg["data"]["class_names"]
    h, w = cfg["data"]["image_size"]
    bs = cfg["training"]["batch_size"]

    def _make(split: str, shuffle: bool):
        path = root / split
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python -m src.data_loader --prepare")
        return keras.utils.image_dataset_from_directory(
            path,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=(h, w),
            batch_size=bs,
            shuffle=shuffle,
            seed=seed,
        )

    train_ds = _make("train", True)
    val_ds = _make("val", False)
    test_ds = _make("test", False)

    aug = build_augmentation(cfg)
    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
        class_names,
    )


def class_counts(data_dir: str | Path, split: str, class_names: List[str]) -> Dict[str, int]:
    """Count images per class for one split folder."""
    root = Path(data_dir) / split
    return {c: len(list((root / c).glob("*"))) if (root / c).exists() else 0 for c in class_names}


def compute_class_weights(cfg: dict | None = None, data_dir: str | Path | None = None) -> Dict[int, float]:
    """Balanced class weights {class_index: weight} from the training split."""
    cfg = cfg or load_config()
    root = Path(data_dir) if data_dir else resolve_path(cfg["paths"]["processed_data"])
    names = cfg["data"]["class_names"]
    counts = class_counts(root, "train", names)
    total = sum(counts.values())
    k = len(names)
    return {i: (total / (k * counts[c])) if counts[c] else 1.0 for i, c in enumerate(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset audit and preparation")
    parser.add_argument("--audit", action="store_true", help="audit raw data")
    parser.add_argument("--prepare", action="store_true", help="create train/val/test splits")
    parser.add_argument("--overwrite", action="store_true", help="rebuild existing splits")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.audit:
        audit_dataset(cfg)
    if args.prepare:
        prepare_splits(cfg, overwrite=args.overwrite)
    if not (args.audit or args.prepare):
        parser.print_help()


if __name__ == "__main__":
    main()
