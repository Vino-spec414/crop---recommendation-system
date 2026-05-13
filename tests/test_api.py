"""
test_api.py
===========
Integration tests for the Flask REST API (app/app.py).

These tests use Flask's built-in test client, which runs the application
in a single thread without starting a real HTTP server.  They verify the
full request → validation → prediction → response cycle.

The trained_artefacts fixture from conftest.py is used to inject a real
(but small) model so we exercise the actual inference path, not a mock.
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch

import pytest

# Make src/ importable
SRC = Path(__file__).resolve().parent.parent / "src"
APP = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(APP))


@pytest.fixture(scope="module")
def test_client(trained_artefacts):
    """
    Create a Flask test client with real artefacts injected into predict.py.
    Module scope: the client is created once per test file.
    """
    import predict as predict_module

    model, scaler, le = trained_artefacts
    predict_module._model         = model
    predict_module._scaler        = scaler
    predict_module._label_encoder = le

    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

    # Cleanup
    predict_module._model         = None
    predict_module._scaler        = None
    predict_module._label_encoder = None


VALID_PAYLOAD = {
    "N": 90.0, "P": 42.0, "K": 43.0,
    "temperature": 20.87, "humidity": 82.0,
    "ph": 6.5, "rainfall": 202.93,
}


# ══════════════════════════════════════════════════════════════════════════════
# GET /
# ══════════════════════════════════════════════════════════════════════════════

class TestIndexRoute:

    def test_returns_200(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200

    def test_returns_html(self, test_client):
        response = test_client.get("/")
        assert b"html" in response.data.lower() or b"CropSense" in response.data


# ══════════════════════════════════════════════════════════════════════════════
# GET /health
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthRoute:

    def test_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_returns_json(self, test_client):
        response = test_client.get("/health")
        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_status_is_ok(self, test_client):
        response = test_client.get("/health")
        data = json.loads(response.data)
        assert data["status"] == "ok"

    def test_service_name_present(self, test_client):
        response = test_client.get("/health")
        data = json.loads(response.data)
        assert "service" in data


# ══════════════════════════════════════════════════════════════════════════════
# GET /model-info
# ══════════════════════════════════════════════════════════════════════════════

class TestModelInfoRoute:

    def test_returns_200(self, test_client):
        response = test_client.get("/model-info")
        assert response.status_code == 200

    def test_returns_json(self, test_client):
        response = test_client.get("/model-info")
        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_has_model_type(self, test_client):
        response = test_client.get("/model-info")
        data = json.loads(response.data)
        assert "model_type" in data

    def test_has_n_classes(self, test_client):
        response = test_client.get("/model-info")
        data = json.loads(response.data)
        assert "n_classes" in data
        assert data["n_classes"] == 22

    def test_has_features(self, test_client):
        response = test_client.get("/model-info")
        data = json.loads(response.data)
        assert "features" in data
        assert len(data["features"]) == 7


# ══════════════════════════════════════════════════════════════════════════════
# POST /predict — happy path
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictRouteSuccess:

    def _post(self, test_client, payload=None):
        return test_client.post(
            "/predict",
            data=json.dumps(payload or VALID_PAYLOAD),
            content_type="application/json",
        )

    def test_returns_200(self, test_client):
        assert self._post(test_client).status_code == 200

    def test_returns_json(self, test_client):
        response = self._post(test_client)
        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_status_is_success(self, test_client):
        response = self._post(test_client)
        data = json.loads(response.data)
        assert data["status"] == "success"

    def test_crop_key_present(self, test_client):
        response = self._post(test_client)
        data = json.loads(response.data)
        assert "crop" in data

    def test_crop_is_non_empty_string(self, test_client):
        response = self._post(test_client)
        data = json.loads(response.data)
        assert isinstance(data["crop"], str)
        assert len(data["crop"]) > 0

    def test_confidence_key_present(self, test_client):
        response = self._post(test_client)
        data = json.loads(response.data)
        assert "confidence" in data

    def test_confidence_in_valid_range(self, test_client):
        response = self._post(test_client)
        data = json.loads(response.data)
        assert 0.0 <= data["confidence"] <= 100.0

    def test_top3_key_present(self, test_client):
        response = self._post(test_client)
        data = json.loads(response.data)
        assert "top3" in data

    def test_top3_has_three_items(self, test_client):
        response = self._post(test_client)
        data = json.loads(response.data)
        assert len(data["top3"]) == 3

    def test_top3_items_schema(self, test_client):
        response = self._post(test_client)
        data = json.loads(response.data)
        for item in data["top3"]:
            assert "crop"        in item
            assert "probability" in item

    def test_same_input_deterministic(self, test_client):
        r1 = json.loads(self._post(test_client).data)
        r2 = json.loads(self._post(test_client).data)
        assert r1["crop"] == r2["crop"]

    def test_integer_values_accepted(self, test_client):
        int_payload = {k: int(v) for k, v in VALID_PAYLOAD.items()}
        response = self._post(test_client, int_payload)
        assert response.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# POST /predict — error handling
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictRouteErrors:

    def _post(self, test_client, payload, content_type="application/json"):
        return test_client.post(
            "/predict",
            data=json.dumps(payload),
            content_type=content_type,
        )

    def test_missing_feature_returns_400(self, test_client):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "N"}
        response = self._post(test_client, bad)
        assert response.status_code == 400

    def test_missing_feature_error_message(self, test_client):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "N"}
        data = json.loads(self._post(test_client, bad).data)
        assert data["status"] == "error"
        assert "message" in data

    def test_out_of_range_ph_returns_400(self, test_client):
        bad = {**VALID_PAYLOAD, "ph": 20.0}
        response = self._post(test_client, bad)
        assert response.status_code == 400

    def test_negative_rainfall_returns_400(self, test_client):
        bad = {**VALID_PAYLOAD, "rainfall": -50.0}
        response = self._post(test_client, bad)
        assert response.status_code == 400

    def test_non_json_content_type_returns_400(self, test_client):
        response = test_client.post(
            "/predict",
            data="not json at all",
            content_type="text/plain",
        )
        assert response.status_code == 400

    def test_empty_body_returns_400(self, test_client):
        response = test_client.post(
            "/predict",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_error_response_has_status_key(self, test_client):
        bad = {**VALID_PAYLOAD, "ph": -1}
        data = json.loads(self._post(test_client, bad).data)
        assert "status" in data
        assert data["status"] == "error"

    def test_error_response_has_message_key(self, test_client):
        bad = {**VALID_PAYLOAD, "humidity": 200}
        data = json.loads(self._post(test_client, bad).data)
        assert "message" in data
        assert len(data["message"]) > 0
