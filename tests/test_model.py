"""Tests that load the real model. Skipped when TensorFlow or the weights are missing,
so CI (which installs neither) stays fast. Run locally with: pytest -m model"""

import numpy as np
import pytest

from app.config import BASE_DIR
from app.services.classifier import WEIGHTS_FILE

pytestmark = pytest.mark.model

MODEL_DIR = BASE_DIR / "models"


@pytest.fixture(scope="module")
def classifier():
    pytest.importorskip("tensorflow")
    if not (MODEL_DIR / WEIGHTS_FILE).exists():
        pytest.skip("model weights not downloaded")
    from app.services.classifier import KerasClassifier

    return KerasClassifier.load(MODEL_DIR, BASE_DIR / "app" / "data" / "labels.json")


def test_predict_returns_sorted_probabilities(classifier):
    image = np.random.default_rng(0).integers(0, 256, (299, 299, 3)).astype(np.float32)
    result = classifier.predict(image, top_k=5)

    assert len(result) == 5
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)
    assert 0 < sum(scores) <= 1.0001


def test_label_count_mismatch_is_rejected():
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    from app.services.classifier import KerasClassifier

    tiny = tf.keras.Sequential([tf.keras.Input((4,)), tf.keras.layers.Dense(3)])
    with pytest.raises(ValueError, match="outputs"):
        KerasClassifier(tiny, labels=["a", "b"])
