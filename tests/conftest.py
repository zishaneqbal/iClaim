"""Shared pytest fixtures. Tests never need the real dataset or a trained model."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


@pytest.fixture()
def cfg() -> dict:
    """A private copy of the config with ImageNet downloads disabled."""
    c = copy.deepcopy(load_config())
    c["model"]["use_imagenet_weights"] = False
    return c


@pytest.fixture()
def rgb_image() -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture(scope="session")
def baseline_model():
    """Small randomly initialised CNN (session scoped to keep tests fast)."""
    from src.model import build_model

    c = copy.deepcopy(load_config())
    c["model"]["use_imagenet_weights"] = False
    return build_model("baseline", c)
