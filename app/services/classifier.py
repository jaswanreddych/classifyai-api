"""The image classifier.

The API depends on the small ``Classifier`` protocol, not on TensorFlow. That
keeps the tests fast (they use a fake) and means TensorFlow is only imported
when the real model is loaded.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

ARCHITECTURE_FILE = "model_architecture.json"
WEIGHTS_FILE = "model_weights.weights.h5"


class Classifier(Protocol):
    name: str

    def predict(self, image: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return the ``top_k`` (label, probability) pairs, best first."""
        ...


class KerasClassifier:
    """Xception transfer-learning model, rebuilt from JSON architecture + weights."""

    name = "xception-medicinal-leaf-78"

    def __init__(self, model, labels: list[str]) -> None:  # noqa: ANN001
        output_units = model.output_shape[-1]
        if output_units != len(labels):
            raise ValueError(
                f"Model has {output_units} outputs but labels.json has {len(labels)}"
            )
        self._model = model
        self._labels = labels
        # One prediction at a time. Keeps memory use and latency predictable.
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model_dir: Path, labels_path: Path) -> "KerasClassifier":
        import tensorflow as tf  # imported here so the rest of the app never needs it

        architecture = (model_dir / ARCHITECTURE_FILE).read_text(encoding="utf-8")
        weights = model_dir / WEIGHTS_FILE
        if not weights.exists():
            raise FileNotFoundError(
                f"{weights} not found. Run: python scripts/download_model.py"
            )

        model = tf.keras.models.model_from_json(architecture)
        model.load_weights(weights)
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        classifier = cls(model, labels)
        classifier._warm_up()
        return classifier

    def _warm_up(self) -> None:
        """Run one dummy prediction so the first real request is not slow."""
        self.predict(np.zeros((299, 299, 3), dtype=np.float32), top_k=1)
        logger.info("Model warmed up")

    def predict(self, image: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        batch = image[np.newaxis, ...]
        with self._lock:
            probabilities = self._model([batch], training=False).numpy()[0]
        best = np.argsort(probabilities)[::-1][:top_k]
        return [(self._labels[i], float(probabilities[i])) for i in best]
