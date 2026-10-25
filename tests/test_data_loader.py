"""Dataset preparation: stratified splits, deduplication, leakage protection."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data_loader import audit_dataset, compute_class_weights, list_raw_images, prepare_splits


def _make_raw(root: Path, counts: dict, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for cls, n in counts.items():
        folder = root / cls
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = rng.integers(0, 255, size=(72, 72, 3), dtype=np.uint8)
            Image.fromarray(arr).save(folder / f"{cls}_{i}.jpg")


@pytest.fixture()
def data_cfg(cfg, tmp_path):
    cfg["paths"]["raw_data"] = str(tmp_path / "raw")
    cfg["paths"]["processed_data"] = str(tmp_path / "processed")
    return cfg


def test_prepare_splits_ratios_and_no_overlap(data_cfg, tmp_path):
    counts = {c: 20 for c in data_cfg["data"]["class_names"]}
    _make_raw(tmp_path / "raw", counts)
    manifest = prepare_splits(data_cfg)
    assert len(manifest) == 100

    by_split = manifest.groupby("split").size()
    assert by_split["train"] == 70 and by_split["val"] == 15 and by_split["test"] == 15

    # Every class appears in every split (stratified)
    pivot = manifest.groupby(["split", "class"]).size().unstack(fill_value=0)
    assert (pivot > 0).all().all()

    # Source files are never reused across splits
    assert manifest["source"].is_unique


def test_duplicates_are_removed_to_prevent_leakage(data_cfg, tmp_path):
    raw = tmp_path / "raw"
    _make_raw(raw, {c: 10 for c in data_cfg["data"]["class_names"]})
    dup_src = raw / "scratch" / "scratch_0.jpg"
    for i in range(3):
        (raw / "dent" / f"copy_{i}.jpg").write_bytes(dup_src.read_bytes())
    manifest = prepare_splits(data_cfg)
    digests = [hashlib.md5(Path(f).read_bytes()).hexdigest() for f in manifest["file"]]
    assert len(digests) == len(set(digests))


def test_existing_splits_are_protected(data_cfg, tmp_path):
    _make_raw(tmp_path / "raw", {c: 10 for c in data_cfg["data"]["class_names"]})
    prepare_splits(data_cfg)
    with pytest.raises(FileExistsError):
        prepare_splits(data_cfg)
    assert len(prepare_splits(data_cfg, overwrite=True)) == 50


def test_missing_raw_data_gives_helpful_error(data_cfg):
    with pytest.raises(FileNotFoundError, match="DATASET_GUIDE"):
        prepare_splits(data_cfg)


def test_split_is_reproducible(data_cfg, tmp_path):
    _make_raw(tmp_path / "raw", {c: 12 for c in data_cfg["data"]["class_names"]})
    a = prepare_splits(data_cfg)
    b = prepare_splits(data_cfg, overwrite=True)
    pd.testing.assert_series_equal(
        a["source"].reset_index(drop=True), b["source"].reset_index(drop=True)
    )


def test_folder_aliases(data_cfg, tmp_path):
    raw = tmp_path / "raw"
    _make_raw(raw, {"Scratches": 4, "Dents": 4})
    data_cfg["data"]["folder_aliases"] = {"Scratches": "scratch", "Dents": "dent"}
    found = list_raw_images(raw, data_cfg)
    assert len(found["scratch"]) == 4 and len(found["dent"]) == 4


def test_audit_counts_and_flags_unreadable(data_cfg, tmp_path):
    raw = tmp_path / "raw"
    _make_raw(raw, {"scratch": 5, "dent": 5})
    (raw / "dent" / "broken.jpg").write_bytes(b"not an image")
    df = audit_dataset(data_cfg).set_index("class")
    assert df.loc["scratch", "images"] == 5
    assert df.loc["dent", "unreadable"] == 1


def test_class_weights_favour_rare_classes(data_cfg, tmp_path):
    root = tmp_path / "processed"
    _make_raw(root / "train", {"scratch": 30, "dent": 10, "crack": 10, "broken_glass": 10, "major_body_damage": 10})
    w = compute_class_weights(data_cfg, root)
    assert w[0] < w[1]  # scratch (common) weighs less than dent (rare)
