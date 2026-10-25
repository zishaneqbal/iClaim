"""Evaluate a trained model on the held-out test set and update reports/results.md.

CLI:
    python -m src.evaluate --model models/best_model.keras
    python -m src.evaluate --model models/resnet50.keras --name resnet50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow import keras

from src.config import get_logger, load_config, resolve_path
from src.data_loader import load_datasets

logger = get_logger(__name__)

RESULTS_START = "<!-- MODEL_COMPARISON_START -->"
RESULTS_END = "<!-- MODEL_COMPARISON_END -->"


def plot_confusion_matrix(cm: np.ndarray, class_names, out_path: Path, normalise: bool = True) -> None:
    """Save a confusion-matrix heatmap."""
    data = cm.astype("float")
    if normalise:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        data, annot=True, fmt=".2f" if normalise else ".0f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalise else ""))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_model(
    model_path: str | Path | None = None,
    name: str | None = None,
    cfg: dict | None = None,
    data_dir: str | Path | None = None,
) -> Dict:
    """Evaluate on the TEST split only. Returns a metrics dict and saves figures/JSON."""
    cfg = cfg or load_config()
    model_path = Path(model_path) if model_path else resolve_path(cfg["paths"]["best_model"])
    if not model_path.is_absolute():
        model_path = resolve_path(str(model_path))
    name = name or model_path.stem
    figures = resolve_path(cfg["paths"]["figures_dir"])

    model = keras.models.load_model(model_path)
    _, _, test_ds, class_names = load_datasets(cfg, data_dir)

    y_true, y_prob = [], []
    started = time.time()
    n_images = 0
    for x, y in test_ds:
        y_prob.append(model.predict(x, verbose=0))
        y_true.append(np.argmax(y.numpy(), axis=1))
        n_images += len(x)
    elapsed = time.time() - started
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "model": name,
        "test_images": int(n_images),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "parameters": int(model.count_params()),
        "ms_per_image": float(elapsed / max(n_images, 1) * 1000),
        "per_class": {c: report[c] for c in class_names},
        "confusion_matrix": cm.tolist(),
    }

    plot_confusion_matrix(cm, class_names, figures / f"confusion_matrix_{name}.png")
    (figures / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Test accuracy=%.4f macro-F1=%.4f (%s)", metrics["accuracy"], metrics["macro_f1"], name)
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))
    update_results_md(cfg)
    return metrics


def update_results_md(cfg: dict | None = None) -> Path:
    """Regenerate the model comparison table in reports/results.md from metrics_*.json."""
    cfg = cfg or load_config()
    figures = resolve_path(cfg["paths"]["figures_dir"])
    results_path = resolve_path(cfg["paths"]["results_md"])

    rows = []
    for f in sorted(figures.glob("metrics_*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        rows.append(m)
    lines = [
        "| Model | Accuracy | Macro-F1 | Weighted-F1 | Parameters | ms / image |",
        "|-------|---------:|---------:|------------:|-----------:|-----------:|",
    ]
    for m in rows:
        lines.append(
            f"| {m['model']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} "
            f"| {m['parameters']:,} | {m['ms_per_image']:.1f} |"
        )
    table = "\n".join(lines) if rows else "_No evaluations have been run yet._"
    block = f"{RESULTS_START}\n{table}\n{RESULTS_END}"

    text = results_path.read_text(encoding="utf-8") if results_path.exists() else ""
    if RESULTS_START in text and RESULTS_END in text:
        before = text.split(RESULTS_START)[0]
        after = text.split(RESULTS_END)[1]
        text = before + block + after
    else:
        text = text.rstrip() + "\n\n## Model comparison\n\n" + block + "\n"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(text, encoding="utf-8")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the test set")
    parser.add_argument("--model", default=None, help="path to .keras model")
    parser.add_argument("--name", default=None, help="label used in reports (default: file name)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    evaluate_model(args.model, args.name, load_config(args.config), args.data_dir)


if __name__ == "__main__":
    main()
