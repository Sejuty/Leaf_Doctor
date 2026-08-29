"""Preprocessing and prediction, shared by the Streamlit app, the tests and the notebook.

Keeping preprocessing here (rather than inline in app.py) is what makes the
channel-order handling testable — see tests/test_inference.py.
"""

import cv2
import numpy as np

from .config import CLASS_NAMES, IMG_SIZE, MODEL_PATH


class InvalidImageError(ValueError):
    """Raised when uploaded bytes can't be decoded as an image."""


def load_model(path=None):
    """Load the trained Keras model. Import of tensorflow is deferred — it is a
    multi-second import, and callers like the tests don't always need it."""
    import tensorflow as tf

    path = path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train one with `python -m leafdoctor.train`."
        )
    return tf.keras.models.load_model(path)


def decode_image(data):
    """Decode raw uploaded bytes into an RGB uint8 array.

    cv2.imdecode returns BGR, but the model was trained on RGB batches produced
    by Keras' ImageDataGenerator (which loads via PIL). Converting here is the
    fix for a real inference bug: the app previously fed BGR straight to a model
    trained on RGB, silently degrading every prediction.
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # also flattens RGBA/greyscale to 3ch
    if bgr is None:
        raise InvalidImageError("Could not decode the file as an image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess(rgb):
    """RGB uint8 array -> normalised (1, 128, 128, 3) float32 batch.

    Mirrors training exactly: resize to IMG_SIZE, scale to [0, 1] (the
    ``rescale=1./255`` in the training ImageDataGenerator).
    """
    resized = cv2.resize(rgb, IMG_SIZE)
    normed = resized.astype(np.float32) / 255.0
    return np.expand_dims(normed, axis=0)


def predict(model, rgb, top_k=3):
    """Return the top_k (raw_label, probability) pairs, most likely first."""
    probs = model.predict(preprocess(rgb), verbose=0)[0]
    if len(probs) != len(CLASS_NAMES):
        raise RuntimeError(
            f"Model outputs {len(probs)} classes but {len(CLASS_NAMES)} labels are "
            f"configured. models/class_names.json is out of sync with the weights."
        )
    order = np.argsort(probs)[::-1][:top_k]
    return [(CLASS_NAMES[i], float(probs[i])) for i in order]
