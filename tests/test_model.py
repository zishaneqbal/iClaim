"""Model builders and serialisation."""

import numpy as np
import pytest

from src.model import BACKBONES, build_model, get_backbone_layer, unfreeze_for_finetuning


@pytest.mark.parametrize("backbone", BACKBONES)
def test_output_shape_and_softmax(cfg, backbone):
    model = build_model(backbone, cfg)
    h, w = cfg["data"]["image_size"]
    out = model.predict(np.random.rand(2, h, w, 3).astype("float32") * 255, verbose=0)
    assert out.shape == (2, len(cfg["data"]["class_names"]))
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)


def test_unknown_backbone_raises(cfg):
    with pytest.raises(ValueError):
        build_model("vgg99", cfg)


def test_backbone_frozen_by_default(cfg):
    model = build_model("mobilenetv2", cfg)
    assert get_backbone_layer(model).trainable is False


def test_unfreeze_makes_last_layers_trainable_but_not_batchnorm(cfg):
    from tensorflow.keras import layers

    model = build_model("mobilenetv2", cfg)
    n = unfreeze_for_finetuning(model, 20)
    assert n > 0
    base = get_backbone_layer(model)
    assert not any(isinstance(lyr, layers.BatchNormalization) and lyr.trainable for lyr in base.layers)


def test_baseline_has_no_backbone(cfg):
    model = build_model("baseline", cfg)
    assert get_backbone_layer(model) is None
    assert unfreeze_for_finetuning(model, 10) == 0


def test_resnet_save_load_roundtrip(cfg, tmp_path):
    """The custom ResNet preprocessing layer must survive .keras save/load."""
    from tensorflow import keras

    model = build_model("resnet50", cfg)
    path = tmp_path / "resnet.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    h, w = cfg["data"]["image_size"]
    x = np.random.rand(1, h, w, 3).astype("float32") * 255
    assert np.allclose(model.predict(x, verbose=0), loaded.predict(x, verbose=0), atol=1e-4)
