"""
app.py  (v2 — Extended API)
===========================
Flask backend for the Crop Recommendation System.

Routes:
  GET  /              → Serves the main web UI
  POST /predict       → Single prediction (JSON input)
  POST /batch-predict → Multiple predictions in one request
  GET  /history       → Last N predictions from the request log
  GET  /model-info    → Model metadata
  GET  /health        → Health-check endpoint
"""

import sys
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import predict_crop, get_model_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LOG_FILE    = PROJECT_ROOT / "prediction_log.jsonl"
BATCH_LIMIT = 50

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)


def _log_prediction(request_id, features, result, latency_ms, endpoint="/predict"):
    record = {
        "request_id": request_id,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "endpoint":   endpoint,
        "inputs":     features,
        "prediction": result.get("crop"),
        "confidence": result.get("confidence"),
        "latency_ms": round(latency_ms, 2),
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.warning("Could not write to prediction log: %s", e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """Live model metrics dashboard."""
    return render_template("dashboard.html")


@app.route("/api/dashboard-metrics")
def dashboard_metrics():
    """Return the metrics JSON generated at training time."""
    metrics_path = PROJECT_ROOT / "models" / "dashboard_metrics.json"
    if not metrics_path.exists():
        return jsonify({
            "status":  "error",
            "message": "No dashboard metrics found. Run training first to generate dashboard_metrics.json.",
        }), 404
    return jsonify(json.loads(metrics_path.read_text(encoding="utf-8"))), 200


@app.route("/predict", methods=["POST"])
def predict():
    request_id = str(uuid.uuid4())[:8]
    if not request.is_json:
        return jsonify({"status": "error", "request_id": request_id,
                        "message": "Request Content-Type must be application/json."}), 400
    features = request.get_json()
    t0 = time.perf_counter()
    try:
        result = predict_crop(features)
        latency_ms = (time.perf_counter() - t0) * 1000
        _log_prediction(request_id, features, result, latency_ms)
        return jsonify({
            "status":                "success",
            "request_id":            request_id,
            "crop":                  result["crop"],
            "confidence":            result["confidence"],
            "margin":                result.get("margin"),
            "confidence_tier":       result.get("confidence_tier"),
            "top3":                  result["top3"],
            "feature_contributions": result.get("feature_contributions"),
            "latency_ms":            round(latency_ms, 2),
        }), 200
    except ValueError as e:
        return jsonify({"status": "error", "request_id": request_id, "message": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"status": "error", "request_id": request_id,
                        "message": "Model not trained. Run 'python src/train_model.py' first."}), 503
    except Exception:
        logger.exception("[%s] Unexpected error", request_id)
        return jsonify({"status": "error", "request_id": request_id,
                        "message": "Internal server error."}), 500


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    request_id = str(uuid.uuid4())[:8]
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON."}), 400
    body = request.get_json()
    if not isinstance(body, dict) or "samples" not in body:
        return jsonify({"status": "error",
                        "message": "Body must contain a 'samples' key with a list."}), 400
    samples = body["samples"]
    if not isinstance(samples, list) or len(samples) == 0:
        return jsonify({"status": "error", "message": "'samples' must be a non-empty list."}), 400
    if len(samples) > BATCH_LIMIT:
        return jsonify({"status": "error",
                        "message": f"Batch size {len(samples)} exceeds limit of {BATCH_LIMIT}."}), 400

    t0 = time.perf_counter()
    results = []
    for i, features in enumerate(samples):
        try:
            result = predict_crop(features)
            results.append({"index": i, "status": "success", "crop": result["crop"],
                             "confidence": result["confidence"], "top3": result["top3"]})
        except (ValueError, TypeError) as e:
            results.append({"index": i, "status": "error", "message": str(e)})
        except Exception:
            results.append({"index": i, "status": "error", "message": "Prediction failed."})

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info("[%s] Batch %d/%d in %.1fms", request_id, success_count, len(samples), latency_ms)
    for r in results:
        if r["status"] == "success":
            _log_prediction(request_id, samples[r["index"]], r,
                            latency_ms / len(samples), "/batch-predict")
    return jsonify({"status": "success", "request_id": request_id, "total": len(samples),
                    "succeeded": success_count, "failed": len(samples) - success_count,
                    "results": results, "latency_ms": latency_ms}), 200


@app.route("/history")
def history():
    try:
        n = min(int(request.args.get("n", 20)), 200)
    except ValueError:
        return jsonify({"status": "error", "message": "'n' must be an integer."}), 400
    records = []
    if LOG_FILE.exists():
        try:
            lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
            for line in lines[-n:]:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        except OSError:
            pass
    return jsonify({"status": "ok", "count": len(records),
                    "records": list(reversed(records))}), 200


@app.route("/model-info")
def model_info():
    try:
        info = get_model_info()
        return jsonify({"status": "ok", **info}), 200
    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 503


@app.route("/health")
def health():
    log_entries = 0
    if LOG_FILE.exists():
        try:
            log_entries = sum(1 for _ in open(LOG_FILE, encoding="utf-8"))
        except OSError:
            pass
    return jsonify({"status": "ok", "service": "crop-recommendation-api",
                    "log_entries": log_entries}), 200


if __name__ == "__main__":
    print(f"\n{'='*55}\n  🌾  Crop Recommendation System — API Server v2")
    print(f"  URL: http://localhost:5000  |  Batch: POST /batch-predict")
    print(f"{'='*55}\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

