import pytest

from tests.conftest import make_image


def upload(client, data: bytes, content_type: str = "image/png", **params):
    return client.post(
        "/api/v1/predictions",
        files={"file": ("leaf.png", data, content_type)},
        params=params,
    )


# ---------- health ----------


def test_live(client):
    assert client.get("/health/live").json() == {"status": "ok"}


def test_ready_when_model_loaded(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_loaded": True, "plants": 78}


def test_not_ready_without_model(client_without_model):
    response = client_without_model.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False


# ---------- plants ----------


def test_list_plants(client):
    body = client.get("/api/v1/plants").json()
    assert len(body) == 78
    assert set(body[0]) == {"id", "name", "scientific_name", "family"}


def test_search_plants(client):
    body = client.get("/api/v1/plants", params={"q": "tulsi"}).json()
    assert [p["id"] for p in body] == ["Tulsi"]


def test_get_plant(client):
    body = client.get("/api/v1/plants/Tulsi").json()
    assert body["id"] == "Tulsi"
    assert body["uses"]["medicinal"]


def test_unknown_plant_uses_error_shape(client):
    response = client.get("/api/v1/plants/nope")
    assert response.status_code == 404
    assert response.json() == {
        "error": {"code": "plant_not_found", "message": "No plant with id 'nope'."}
    }


# ---------- predictions ----------


def test_prediction_success(client, fake_classifier):
    response = upload(client, make_image())
    assert response.status_code == 200
    body = response.json()

    assert body["model"] == "fake-model"
    assert [p["plant_id"] for p in body["predictions"]] == ["Tulsi", "Neem", "Mint"]
    assert body["predictions"][0]["confidence"] == 0.9
    assert body["plant"]["id"] == "Tulsi"
    assert body["low_confidence"] is False
    assert body["inference_ms"] >= 0
    # The classifier receives a 299x299 RGB array.
    assert fake_classifier.calls == [((299, 299, 3), 3)]


def test_top_k_limits_results(client):
    body = upload(client, make_image(), top_k=1).json()
    assert len(body["predictions"]) == 1


@pytest.mark.parametrize("top_k", [0, 6, -1])
def test_top_k_out_of_range_is_rejected(client, top_k):
    assert upload(client, make_image(), top_k=top_k).status_code == 422


def test_low_confidence_is_flagged(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import FakeClassifier

    with TestClient(create_app(settings, FakeClassifier(confidence=0.2))) as client:
        assert upload(client, make_image()).json()["low_confidence"] is True


def test_wrong_content_type_is_415(client):
    response = upload(client, b"hello", content_type="text/plain")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_corrupt_image_is_400(client):
    response = upload(client, b"not really a png", content_type="image/png")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"


def test_oversized_upload_is_413(client):
    too_big = b"\x00" * (1024 * 1024 + 1)  # test settings allow 1 MB
    response = upload(client, too_big)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_missing_file_is_422(client):
    assert client.post("/api/v1/predictions").status_code == 422


def test_prediction_without_model_is_503(client_without_model):
    response = upload(client_without_model, make_image())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_request_id_header_is_set_and_echoed(client):
    assert client.get("/health/live").headers["x-request-id"]
    echoed = client.get("/health/live", headers={"x-request-id": "abc123"})
    assert echoed.headers["x-request-id"] == "abc123"
