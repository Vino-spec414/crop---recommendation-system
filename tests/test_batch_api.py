"""
test_batch_api.py
=================
Integration tests for the extended API endpoints:
  POST /batch-predict
  GET  /history
"""

import sys
import json
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
APP = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(APP))

VALID = {
    "N": 90.0, "P": 42.0, "K": 43.0,
    "temperature": 20.87, "humidity": 82.0,
    "ph": 6.5, "rainfall": 202.93,
}


@pytest.fixture(scope="module")
def client(trained_artefacts):
    import predict as pm
    model, scaler, le = trained_artefacts
    pm._model = model
    pm._scaler = scaler
    pm._label_encoder = le
    from app import app
    app.config["TESTING"] = True
    # Redirect log to /tmp to avoid writing in project dir during tests
    import app as app_mod
    app_mod.LOG_FILE = Path("/tmp/test_prediction_log.jsonl")
    app_mod.LOG_FILE.unlink(missing_ok=True)
    with app.test_client() as c:
        yield c
    pm._model = None
    pm._scaler = None
    pm._label_encoder = None


def _batch_post(client, samples):
    return client.post(
        "/batch-predict",
        data=json.dumps({"samples": samples}),
        content_type="application/json",
    )


# ══════════════════════════════════════════════════════════════
# /batch-predict — happy path
# ══════════════════════════════════════════════════════════════

class TestBatchPredictSuccess:

    def test_single_sample_returns_200(self, client):
        assert _batch_post(client, [VALID]).status_code == 200

    def test_three_samples_all_succeed(self, client):
        r = json.loads(_batch_post(client, [VALID, VALID, VALID]).data)
        assert r["status"] == "success"
        assert r["succeeded"] == 3
        assert r["failed"] == 0

    def test_result_count_matches_input(self, client):
        samples = [VALID] * 5
        r = json.loads(_batch_post(client, samples).data)
        assert len(r["results"]) == 5

    def test_each_result_has_index(self, client):
        r = json.loads(_batch_post(client, [VALID, VALID]).data)
        indices = [item["index"] for item in r["results"]]
        assert indices == [0, 1]

    def test_each_result_has_crop(self, client):
        r = json.loads(_batch_post(client, [VALID]).data)
        assert "crop" in r["results"][0]

    def test_each_result_has_confidence(self, client):
        r = json.loads(_batch_post(client, [VALID]).data)
        assert "confidence" in r["results"][0]

    def test_response_has_latency(self, client):
        r = json.loads(_batch_post(client, [VALID]).data)
        assert "latency_ms" in r
        assert r["latency_ms"] >= 0

    def test_mixed_valid_invalid(self, client):
        bad = {**VALID, "ph": 99}
        r = json.loads(_batch_post(client, [VALID, bad, VALID]).data)
        assert r["succeeded"] == 2
        assert r["failed"] == 1

    def test_failed_item_has_error_status(self, client):
        bad = {**VALID, "humidity": 200}
        r = json.loads(_batch_post(client, [bad]).data)
        assert r["results"][0]["status"] == "error"

    def test_failed_item_has_message(self, client):
        bad = {**VALID, "humidity": 200}
        r = json.loads(_batch_post(client, [bad]).data)
        assert "message" in r["results"][0]

    def test_has_request_id(self, client):
        r = json.loads(_batch_post(client, [VALID]).data)
        assert "request_id" in r


# ══════════════════════════════════════════════════════════════
# /batch-predict — error handling
# ══════════════════════════════════════════════════════════════

class TestBatchPredictErrors:

    def test_missing_samples_key_returns_400(self, client):
        r = client.post("/batch-predict",
                        data=json.dumps({"data": [VALID]}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_empty_samples_list_returns_400(self, client):
        r = client.post("/batch-predict",
                        data=json.dumps({"samples": []}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_non_json_returns_400(self, client):
        r = client.post("/batch-predict",
                        data="not json",
                        content_type="text/plain")
        assert r.status_code == 400

    def test_batch_over_limit_returns_400(self, client):
        samples = [VALID] * 51
        r = client.post("/batch-predict",
                        data=json.dumps({"samples": samples}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_exactly_at_limit_succeeds(self, client):
        samples = [VALID] * 50
        r = client.post("/batch-predict",
                        data=json.dumps({"samples": samples}),
                        content_type="application/json")
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════
# /history
# ══════════════════════════════════════════════════════════════

class TestHistory:

    def test_returns_200(self, client):
        assert client.get("/history").status_code == 200

    def test_returns_json(self, client):
        r = json.loads(client.get("/history").data)
        assert isinstance(r, dict)

    def test_has_status_ok(self, client):
        r = json.loads(client.get("/history").data)
        assert r["status"] == "ok"

    def test_has_count_and_records(self, client):
        r = json.loads(client.get("/history").data)
        assert "count" in r
        assert "records" in r

    def test_records_is_list(self, client):
        r = json.loads(client.get("/history").data)
        assert isinstance(r["records"], list)

    def test_count_matches_records_length(self, client):
        r = json.loads(client.get("/history").data)
        assert r["count"] == len(r["records"])

    def test_records_populated_after_predict(self, client):
        # Make a prediction first
        client.post("/predict",
                    data=json.dumps(VALID),
                    content_type="application/json")
        r = json.loads(client.get("/history").data)
        assert r["count"] > 0

    def test_record_has_expected_fields(self, client):
        client.post("/predict",
                    data=json.dumps(VALID),
                    content_type="application/json")
        r = json.loads(client.get("/history").data)
        if r["records"]:
            record = r["records"][0]
            for field in ("request_id", "timestamp", "prediction", "confidence", "latency_ms"):
                assert field in record, f"Missing field: {field}"

    def test_n_param_limits_results(self, client):
        # Make several predictions
        for _ in range(5):
            client.post("/predict", data=json.dumps(VALID), content_type="application/json")
        r = json.loads(client.get("/history?n=2").data)
        assert r["count"] <= 2

    def test_invalid_n_returns_400(self, client):
        r = client.get("/history?n=notanumber")
        assert r.status_code == 400
