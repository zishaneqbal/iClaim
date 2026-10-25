"""Configuration loader. All settings live in config.yaml at the project root."""

from __future__ import annotations

import logging
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=4)
def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load config.yaml and return it as a dict.

    Args:
        path: Optional path to a YAML config. Defaults to <project_root>/config.yaml.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    return _load(str(cfg_path))


def resolve_path(relative: str) -> Path:
    """Resolve a path from the config relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int | None = None) -> int:
    """Seed python, numpy and (if installed) tensorflow for reproducibility."""
    if seed is None:
        seed = int(load_config()["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # TensorFlow is optional for modules that only need config
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:  # pragma: no cover
        pass
    return seed


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
