"""Model builders: baseline CNN and transfer-learning backbones.

Every model takes raw 0-255 RGB pixels and applies the correct backbone-specific
scaling internally, so predict() and the Streamlit app never need to care.
"""

from __future__ import annotations

from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers

from src.config import get_logger, load_config

logger = get_logger(__name__)

BACKBONES = ("baseline", "mobilenetv2", "resnet50", "efficientnetb0")


@keras.utils.register_keras_serializable(package="vehicle_damage")
class ResNetPreprocess(layers.Layer):
    """Caffe-style ResNet50 preprocessing: RGB -> BGR and mean subtraction."""

    def call(self, x):
        import tensorflow as tf

        x = x[..., ::-1]
        mean = tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)
        return x - mean

    def compute_output_shape(self, input_shape):
        return input_shape


def _input_shape(cfg: dict) -> Tuple[int, int, int]:
    h, w = cfg["data"]["image_size"]
    return (h, w, cfg["data"]["channels"])


def build_baseline(num_classes: int, cfg: dict) -> keras.Model:
    """Simple CNN used as a performance floor."""
    inputs = keras.Input(shape=_input_shape(cfg), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    for i, filters in enumerate((32, 64, 128, 128), start=1):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(cfg["model"]["dense_units"], activation="relu", name="fc")(x)
    x = layers.Dropout(cfg["model"]["dropout"], name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="damage_class")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def _make_backbone(name: str, input_shape, use_imagenet: bool) -> keras.Model:
    weights = "imagenet" if use_imagenet else None
    if name == "mobilenetv2":
        return keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)
    if name == "resnet50":
        return keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights=weights)
    if name == "efficientnetb0":
        return keras.applications.EfficientNetB0(input_shape=input_shape, include_top=False, weights=weights)
    raise ValueError(f"Unknown backbone '{name}'. Choose from {BACKBONES}")


def build_transfer_model(backbone: str, num_classes: int, cfg: dict) -> keras.Model:
    """Backbone (frozen by default) + pooling + dense head."""
    input_shape = _input_shape(cfg)
    base = _make_backbone(backbone, input_shape, cfg["model"]["use_imagenet_weights"])
    base.trainable = False

    inputs = keras.Input(shape=input_shape, name="image")
    if backbone == "mobilenetv2":
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(inputs)
    elif backbone == "resnet50":
        x = ResNetPreprocess(name="preprocess")(inputs)
    else:  # EfficientNet rescales internally and expects 0-255
        x = inputs
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(cfg["model"]["dropout"], name="dropout")(x)
    x = layers.Dense(cfg["model"]["dense_units"], activation="relu", name="fc")(x)
    x = layers.Dropout(cfg["model"]["dropout"], name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="damage_class")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_damage_classifier")


def build_model(backbone: str | None = None, cfg: dict | None = None) -> keras.Model:
    """Build an (uncompiled) damage classifier for the configured backbone."""
    cfg = cfg or load_config()
    backbone = (backbone or cfg["model"]["backbone"]).lower()
    num_classes = len(cfg["data"]["class_names"])
    if backbone == "baseline":
        return build_baseline(num_classes, cfg)
    return build_transfer_model(backbone, num_classes, cfg)


def get_backbone_layer(model: keras.Model) -> keras.Model | None:
    """Return the nested backbone model (None for the baseline CNN)."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    return None


def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    """Compile with Adam and categorical cross-entropy."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_for_finetuning(model: keras.Model, last_n: int) -> int:
    """Unfreeze the last `last_n` backbone layers (BatchNorm stays frozen).

    Returns the number of backbone layers that are now trainable.
    """
    base = get_backbone_layer(model)
    if base is None:
        return 0
    base.trainable = True
    cutoff = max(len(base.layers) - last_n, 0)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)
    trainable = sum(1 for lyr in base.layers if lyr.trainable)
    logger.info("Fine-tuning: %d/%d backbone layers trainable", trainable, len(base.layers))
    return trainable
