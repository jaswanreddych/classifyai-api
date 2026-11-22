import io
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app


class FakeClassifier:
    """Returns fixed results so API tests do not need TensorFlow or the model."""

    name = "fake-model"

    def __init__(self, confidence: float = 0.9) -> None:
        self.confidence = confidence
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def predict(self, image: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        self.calls.append((image.shape, top_k))
        ranked = [("Tulsi", self.confidence), ("Neem", 0.05), ("Mint", 0.03)]
        return ranked[:top_k]


def make_image(fmt: str = "PNG", mode: str = "RGB", size=(64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=0).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def settings() -> Settings:
    return Settings(load_model=False, max_upload_mb=1)


@pytest.fixture
def fake_classifier() -> FakeClassifier:
    return FakeClassifier()


@pytest.fixture
def client(settings, fake_classifier) -> Iterator[TestClient]:
    with TestClient(create_app(settings, classifier=fake_classifier)) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
