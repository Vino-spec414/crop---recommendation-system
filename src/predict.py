"""
predict.py
==========
Inference module with per-prediction SHAP explanations and
confidence-tier classification.

Returns (per call):
  - top-1 crop and confidence
  - top-3 alternatives
  - SHAP feature contributions (which features pushed towards/against the prediction)
  - confidence_tier:  'high' | 'medium' | 'low'
  - margin:           gap between top-1 and top-2 confidence
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import joblib

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

MODEL_PATH  = MODELS_DIR / "crop_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_PATH  = MODELS_DIR / "label_encoder.pkl"

FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

FEATURE_LABELS = {
    "N":           "Nitrogen",
    "P":           "Phosphorus",
    "K":           "Potassium",
    "temperature": "Temperature",
    "humidity":    "Humidity",
    "ph":          "Soil pH",
    "rainfall":    "Rainfall",
}

# Domain bounds for feature validation
BOUNDS = {
    "N":           (0, 200),
    "P":           (0, 200),
    "K":           (0, 250),
    "temperature": (0, 50),
    "humidity":    (0, 100),
    "ph":          (0, 14),
    "rainfall":    (0, 3000),
}

# Confidence thresholds (research-grade defaults)
CONFIDENCE_HIGH_THRESHOLD = 70.0   # ≥70% → high confidence
CONFIDENCE_LOW_THRESHOLD  = 40.0   # <40% → low confidence
MARGIN_LOW_THRESHOLD      = 15.0   # margin < 15 percentage points → low confidence

# Module-level cache (singleton pattern)
_model         = None
_scaler        = None
_label_encoder = None
_shap_explainer = None


# ── Artefact loader ───────────────────────────────────────────────────────────

def _load_artefacts():
    global _model, _scaler, _label_encoder
    if _model is not None:
        return

    missing = [p for p in (MODEL_PATH, SCALER_PATH, LABEL_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Trained model artefacts not found:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n\nRun 'python src/train_model.py' first."
        )

    _model         = joblib.load(MODEL_PATH)
    _scaler        = joblib.load(SCALER_PATH)
    _label_encoder = joblib.load(LABEL_PATH)
    logger.info("Loaded model: %s", type(_model).__name__)


def _get_shap_explainer():
    """Lazy-load the SHAP TreeExplainer (only for tree-based models)."""
    global _shap_explainer
    if _shap_explainer is not None:
        return _shap_explainer

    _load_artefacts()
    model_name = type(_model).__name__
    tree_supported = ("RandomForestClassifier", "DecisionTreeClassifier", "ExtraTreesClassifier")

    if model_name not in tree_supported:
        return None  # graceful degradation — SHAP unavailable

    try:
        import shap
        _shap_explainer = shap.TreeExplainer(_model)
        logger.info("SHAP TreeExplainer ready")
        return _shap_explainer
    except ImportError:
        return None


# ── Validation ────────────────────────────────────────────────────────────────

def validate_input(features: dict) -> np.ndarray:
    """Validate dict of features → (1, 7) numpy array."""
    missing = [k for k in FEATURE_ORDER if k not in features]
    if missing:
        raise ValueError(f"Missing input features: {missing}")

    row, errors = [], []
    for feat in FEATURE_ORDER:
        try:
            val = float(features[feat])
        except (TypeError, ValueError):
            errors.append(f"'{feat}' must be numeric, got: {features[feat]!r}")
            continue
        lo, hi = BOUNDS[feat]
        if not (lo <= val <= hi):
            errors.append(f"'{feat}' = {val} is outside valid range [{lo}, {hi}]")
        row.append(val)

    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    return np.array(row, dtype=np.float64).reshape(1, -1)


# ── Confidence tier classification ───────────────────────────────────────────

def _classify_confidence(top1_pct: float, margin: float) -> dict:
    """
    Return a confidence assessment based on top-1 probability AND margin.

    Why margin matters: a top-1 of 45% with second-place at 44% (margin = 1%)
    is far less reliable than 45% with second-place at 5% (margin = 40%).
    """
    if top1_pct >= CONFIDENCE_HIGH_THRESHOLD and margin >= MARGIN_LOW_THRESHOLD:
        tier    = "high"
        message = "Strong recommendation — input conditions clearly match this crop's profile."
        emoji   = "✓"
    elif top1_pct < CONFIDENCE_LOW_THRESHOLD or margin < MARGIN_LOW_THRESHOLD:
        tier    = "low"
        message = ("Low confidence — your input falls between multiple suitable crops. "
                  "Review the alternatives below before deciding.")
        emoji   = "⚠"
    else:
        tier    = "medium"
        message = "Moderate confidence — top alternatives may also be suitable."
        emoji   = "•"

    return {"tier": tier, "message": message, "emoji": emoji}


# ── SHAP feature contributions ────────────────────────────────────────────────

def _get_feature_contributions(x_scaled: np.ndarray, pred_idx: int) -> Optional[list]:
    """
    Return per-feature SHAP contributions for the predicted class.
    Each item: { feature, label, value, contribution, direction }
    """
    explainer = _get_shap_explainer()
    if explainer is None:
        return None

    try:
        shap_vals = explainer.shap_values(x_scaled)
        sv_array = np.array(shap_vals)

        # Normalise to (1, n_features) for the predicted class
        if sv_array.ndim == 3:
            sv = sv_array[0, :, pred_idx]
        elif isinstance(shap_vals, list):
            sv = shap_vals[pred_idx][0]
        else:
            sv = sv_array[0]

        contributions = []
        for i, feat in enumerate(FEATURE_ORDER):
            contrib = float(sv[i])
            contributions.append({
                "feature":      feat,
                "label":        FEATURE_LABELS[feat],
                "value":        float(x_scaled[0, i]),
                "contribution": round(contrib, 4),
                "direction":    "positive" if contrib > 0 else "negative",
                "abs":          abs(contrib),
            })

        # Sort by absolute contribution (highest first)
        contributions.sort(key=lambda c: c["abs"], reverse=True)
        for c in contributions:
            del c["abs"]   # internal field, not for response

        return contributions
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def predict_crop(features: dict) -> dict:
    """
    Main prediction function.

    Returns:
        {
          "crop":        "Rice",
          "confidence":  94.3,
          "top3":        [...],
          "confidence_tier":   {"tier": "high", "message": "...", "emoji": "✓"},
          "margin":      67.2,
          "feature_contributions": [...]   # SHAP — None if unavailable
        }
    """
    _load_artefacts()

    X_raw    = validate_input(features)
    X_scaled = _scaler.transform(X_raw)

    pred_idx  = int(_model.predict(X_scaled)[0])
    crop_name = _label_encoder.inverse_transform([pred_idx])[0]

    # Probabilities and top-3
    top3 = []
    confidence = None
    margin = 0.0
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X_scaled)[0]
        confidence = round(float(proba[pred_idx]) * 100, 2)

        sorted_idx = np.argsort(proba)[::-1][:3]
        top3 = [
            {
                "crop":        _label_encoder.inverse_transform([i])[0].capitalize(),
                "probability": round(float(proba[i]) * 100, 2),
            }
            for i in sorted_idx
        ]

        if len(top3) >= 2:
            margin = round(top3[0]["probability"] - top3[1]["probability"], 2)

    # Confidence tier classification
    conf_tier = _classify_confidence(confidence or 0, margin)

    # SHAP feature contributions
    contributions = _get_feature_contributions(X_scaled, pred_idx)

    return {
        "crop":                  crop_name.capitalize(),
        "confidence":            confidence,
        "margin":                margin,
        "confidence_tier":       conf_tier,
        "top3":                  top3,
        "feature_contributions": contributions,
    }


def get_model_info() -> dict:
    _load_artefacts()
    return {
        "model_type":  type(_model).__name__,
        "n_classes":   len(_label_encoder.classes_),
        "class_names": list(_label_encoder.classes_),
        "features":    FEATURE_ORDER,
    }


# Quick test
if __name__ == "__main__":
    sample = {"N": 90, "P": 42, "K": 43, "temperature": 20.87,
              "humidity": 82.0, "ph": 6.5, "rainfall": 202.93}
    result = predict_crop(sample)
    import json
    print(json.dumps(result, indent=2))
