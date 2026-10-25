"""Grad-CAM explainability: heatmaps that show where the CNN looked.

Works for the baseline CNN (last Conv2D) and the transfer-learning models
(output of the nested backbone, which is its last convolutional feature map).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.config import get_logger, load_config

logger = get_logger(__name__)


def find_feature_layer(model):
    """Return the layer whose output is the last convolutional feature map."""
    from tensorflow import keras

    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):  # nested backbone
            return layer
        if isinstance(layer, keras.layers.Conv2D):
            return layer
    raise ValueError("No convolutional layer found for Grad-CAM")


def make_gradcam_heatmap(model, img_array: np.ndarray, class_index: Optional[int] = None) -> np.ndarray:
    """Compute a Grad-CAM heatmap in [0, 1].

    Args:
        model: Trained Keras model.
        img_array: Preprocessed batch of shape (1, H, W, 3), 0-255 pixels.
        class_index: Class to explain; defaults to the predicted class.

    Returns:
        2D float32 array (feature-map resolution) scaled to [0, 1].
    """
    import tensorflow as tf
    from tensorflow import keras

    feature_layer = find_feature_layer(model)
    x = tf.convert_to_tensor(img_array, dtype=tf.float32)
    conv_out = None
    # Models built by src.model are linear chains, so we replay them layer by layer and
    # capture the feature map on the way. (A nested backbone's `.output` is wired to its own
    # input, not to the outer model's, so building a sub-model from it does not work.)
    with tf.GradientTape() as tape:
        for layer in model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            x = layer(x, training=False)
            if layer is feature_layer:
                conv_out = x
        preds = x
        if conv_out is None:
            raise ValueError("Feature layer was not reached during the forward pass")
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heat = tf.nn.relu(heat)
    heat = heat / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy().astype("float32")


def damage_area_ratio(heatmap: np.ndarray, threshold: float | None = None) -> float:
    """Fraction of the image where activation exceeds the threshold (0-1).

    Used by the rule-based severity module as a proxy for damage extent.
    """
    if threshold is None:
        threshold = load_config()["severity"]["gradcam_activation_threshold"]
    if heatmap.size == 0:
        return 0.0
    return float((heatmap >= threshold).mean())


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a heatmap on the original image. Returns an RGB uint8 array."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0, 1)
    colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, colored, alpha, 0)
