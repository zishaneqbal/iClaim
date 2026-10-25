"""Two-phase training: (1) train the new head, (2) fine-tune the top backbone layers.

CLI:
    python -m src.train --config config.yaml
    python -m src.train --backbone resnet50 --output models/resnet50.keras
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from src.config import get_logger, load_config, resolve_path, set_seed
from src.data_loader import compute_class_weights, load_datasets
from src.model import build_model, compile_model, get_backbone_layer, unfreeze_for_finetuning

logger = get_logger(__name__)


def _callbacks(cfg: dict, checkpoint_path: Path):
    t = cfg["training"]
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=t["early_stopping_patience"], restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=t["reduce_lr_patience"], min_lr=1e-7
        ),
    ]


def plot_history(history: dict, out_path: Path) -> None:
    """Save accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        ax.plot(history.get(key, []), label="train")
        ax.plot(history.get(f"val_{key}", []), label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train(
    cfg: dict | None = None,
    backbone: str | None = None,
    output: str | Path | None = None,
    data_dir: str | Path | None = None,
    epochs_head: int | None = None,
    epochs_finetune: int | None = None,
) -> Path:
    """Train a model and return the path of the best saved checkpoint."""
    cfg = cfg or load_config()
    set_seed(cfg["project"]["seed"])
    t = cfg["training"]
    backbone = (backbone or cfg["model"]["backbone"]).lower()
    out_path = Path(output) if output else resolve_path(cfg["paths"]["best_model"])
    if not out_path.is_absolute():
        out_path = resolve_path(str(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _test_ds, class_names = load_datasets(cfg, data_dir)
    class_weight = compute_class_weights(cfg, data_dir) if t["use_class_weights"] else None
    logger.info("Backbone=%s | classes=%s | class_weight=%s", backbone, class_names, class_weight)

    model = build_model(backbone, cfg)
    ckpt = out_path
    history_all: dict = {}
    started = time.time()

    # Phase 1: train the head (backbone frozen). Baseline trains all layers here.
    compile_model(model, t["lr_head"])
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head if epochs_head is not None else t["epochs_head"],
        class_weight=class_weight,
        callbacks=_callbacks(cfg, ckpt),
    )
    for k, v in h1.history.items():
        history_all.setdefault(k, []).extend(v)

    # Phase 2: fine-tune top backbone layers at a low learning rate.
    n_ft = epochs_finetune if epochs_finetune is not None else t["epochs_finetune"]
    if get_backbone_layer(model) is not None and n_ft > 0:
        unfreeze_for_finetuning(model, t["unfreeze_last_layers"])
        compile_model(model, t["lr_finetune"])
        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=n_ft,
            class_weight=class_weight,
            callbacks=_callbacks(cfg, ckpt),
        )
        for k, v in h2.history.items():
            history_all.setdefault(k, []).extend(v)

    minutes = (time.time() - started) / 60
    figures = resolve_path(cfg["paths"]["figures_dir"])
    plot_history(history_all, figures / f"training_curves_{backbone}.png")

    # Reload the best checkpoint so the saved file is the best-validation model.
    if not out_path.exists():
        model.save(out_path)
    class_json = resolve_path(cfg["paths"]["class_names_json"])
    class_json.parent.mkdir(parents=True, exist_ok=True)
    class_json.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (figures / f"history_{backbone}.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in history_all.items()}, indent=2),
        encoding="utf-8",
    )
    logger.info("Training finished in %.1f min. Best model: %s", minutes, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the vehicle damage classifier")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--backbone", default=None, choices=["baseline", "mobilenetv2", "resnet50", "efficientnetb0"])
    parser.add_argument("--output", default=None, help="where to save the best model (.keras)")
    parser.add_argument("--data-dir", default=None, help="override processed data directory")
    parser.add_argument("--epochs-head", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    args = parser.parse_args()
    train(
        load_config(args.config),
        backbone=args.backbone,
        output=args.output,
        data_dir=args.data_dir,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
    )


if __name__ == "__main__":
    main()
