"""
test_predict.py
===============
Unit tests for src/predict.py — the inference module.

Tests are structured to avoid touching the real /models/ directory.
We inject trained artefacts directly via monkeypatching (pytest's built-in
mechanism for temporarily replacing module-level attributes in tests).
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import predict as predict_module
from predict import validate_input, FEATURE_ORDER


# ══════════════════════════════════════════════════════════════════════════════
# validate_input
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateInput:
    """Tests for the input validation step (no model needed)."""

    def test_returns_numpy_array(self, valid_input):
        result = validate_input(valid_input)
        assert isinstance(result, np.ndarray)

    def test_correct_shape(self, valid_input):
        result = validate_input(valid_input)
        assert result.shape == (1, len(FEATURE_ORDER))

    def test_correct_dtype(self, valid_input):
        result = validate_input(valid_input)
        assert result.dtype == np.float64

    def test_values_preserved(self, valid_input):
        result = validate_input(valid_input)
        for i, key in enumerate(FEATURE_ORDER):
            assert result[0, i] == pytest.approx(valid_input[key])

    def test_raises_on_missing_key(self, valid_input):
        bad = {k: v for k, v in valid_input.items() if k != "N"}
        with pytest.raises(ValueError, match="N"):
            validate_input(bad)

    def test_raises_on_multiple_missing_keys(self):
        with pytest.raises(ValueError):
            validate_input({"N": 50})

    def test_raises_on_non_numeric_value(self, valid_input):
        bad = {**valid_input, "N": "lots"}
        with pytest.raises(ValueError, match="N"):
            validate_input(bad)

    def test_raises_on_none_value(self, valid_input):
        bad = {**valid_input, "temperature": None}
        with pytest.raises(ValueError, match="temperature"):
            validate_input(bad)

    # ── boundary / domain tests ──────────────────────────────────────────────

    def test_accepts_minimum_boundary_values(self):
        """Exact lower-bound values must pass validation."""
        min_input = {
            "N": 0.0, "P": 0.0, "K": 0.0,
            "temperature": 0.0, "humidity": 0.0,
            "ph": 0.0, "rainfall": 0.0,
        }
        result = validate_input(min_input)
        assert result.shape == (1, 7)

    def test_accepts_maximum_boundary_values(self):
        """Exact upper-bound values must pass validation."""
        max_input = {
            "N": 200.0, "P": 200.0, "K": 250.0,
            "temperature": 50.0, "humidity": 100.0,
            "ph": 14.0, "rainfall": 3000.0,
        }
        result = validate_input(max_input)
        assert result.shape == (1, 7)

    def test_raises_when_n_exceeds_upper_bound(self, valid_input):
        bad = {**valid_input, "N": 201.0}
        with pytest.raises(ValueError, match="N"):
            validate_input(bad)

    def test_raises_when_ph_exceeds_14(self, valid_input):
        bad = {**valid_input, "ph": 14.1}
        with pytest.raises(ValueError, match="ph"):
            validate_input(bad)

    def test_raises_when_ph_below_0(self, valid_input):
        bad = {**valid_input, "ph": -0.1}
        with pytest.raises(ValueError, match="ph"):
            validate_input(bad)

    def test_raises_when_humidity_above_100(self, valid_input):
        bad = {**valid_input, "humidity": 101.0}
        with pytest.raises(ValueError, match="humidity"):
            validate_input(bad)

    def test_raises_when_temperature_below_0(self, valid_input):
        bad = {**valid_input, "temperature": -1.0}
        with pytest.raises(ValueError, match="temperature"):
            validate_input(bad)

    def test_multiple_violations_reported_together(self):
        """All violations should be collected and raised together, not one at a time."""
        bad = {
            "N": 999, "P": -5, "K": 0,
            "temperature": 0, "humidity": 0,
            "ph": 0, "rainfall": 0,
        }
        with pytest.raises(ValueError) as exc_info:
            validate_input(bad)
        msg = str(exc_info.value)
        assert "N" in msg
        assert "P" in msg

    def test_integer_inputs_accepted(self, valid_input):
        """Integer inputs should be accepted and coerced to float."""
        int_input = {k: int(v) for k, v in valid_input.items()}
        result = validate_input(int_input)
        assert result.dtype == np.float64

    def test_string_numeric_accepted(self, valid_input):
        """Numeric strings ('90.0') should be coerced, not rejected."""
        str_input = {k: str(v) for k, v in valid_input.items()}
        result = validate_input(str_input)
        assert result.shape == (1, 7)

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            validate_input({})

    def test_extra_keys_ignored(self, valid_input):
        """Extra keys beyond FEATURE_ORDER should not cause failure."""
        extra = {**valid_input, "extra_field": 999}
        result = validate_input(extra)
        assert result.shape == (1, 7)


# ══════════════════════════════════════════════════════════════════════════════
# predict_crop  (uses monkeypatched artefacts)
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictCrop:
    """
    Tests for predict_crop() using the trained_artefacts fixture.
    We monkeypatch the module-level cache variables so the real /models/
    directory is never touched.
    """

    @pytest.fixture(autouse=True)
    def inject_artefacts(self, trained_artefacts):
        """
        Before each test in this class: inject the session-trained artefacts
        into predict.py's module-level cache.
        After each test: restore the original (None) state.
        """
        model, scaler, le = trained_artefacts
        predict_module._model         = model
        predict_module._scaler        = scaler
        predict_module._label_encoder = le
        yield
        predict_module._model         = None
        predict_module._scaler        = None
        predict_module._label_encoder = None

    def test_returns_dict(self, valid_input):
        from predict import predict_crop
        result = predict_crop(valid_input)
        assert isinstance(result, dict)

    def test_result_has_crop_key(self, valid_input):
        from predict import predict_crop
        result = predict_crop(valid_input)
        assert "crop" in result

    def test_result_has_confidence_key(self, valid_input):
        from predict import predict_crop
        result = predict_crop(valid_input)
        assert "confidence" in result

    def test_result_has_top3_key(self, valid_input):
        from predict import predict_crop
        result = predict_crop(valid_input)
        assert "top3" in result

    def test_crop_is_string(self, valid_input):
        from predict import predict_crop
        result = predict_crop(valid_input)
        assert isinstance(result["crop"], str)

    def test_crop_is_capitalised(self, valid_input):
        from predict import predict_crop
        result = predict_crop(valid_input)
        assert result["crop"][0].isupper()

    def test_confidence_is_float_between_0_and_100(self, valid_input):
        from predict import predict_crop
        result = predict_crop(valid_input)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 100.0

    def test_top3_has_three_items(self, valid_input):
        from predict import predict_crop
        result = predict_crop(valid_input)
        assert len(result["top3"]) == 3

    def test_top3_items_have_crop_and_probability(self, valid_input):
        from predict import predict_crop
        result = predict_crop(valid_input)
        for item in result["top3"]:
            assert "crop"        in item
            assert "probability" in item

    def test_top3_probabilities_are_between_0_and_100(self, valid_input):
        from predict import predict_crop
        result = predict_crop(valid_input)
        for item in result["top3"]:
            assert 0.0 <= item["probability"] <= 100.0

    def test_top3_sorted_descending(self, valid_input):
        from predict import predict_crop
        result = predict_crop(valid_input)
        probs = [item["probability"] for item in result["top3"]]
        assert probs == sorted(probs, reverse=True), "top3 must be sorted highest → lowest"

    def test_top_prediction_matches_top3_first(self, valid_input):
        from predict import predict_crop
        result = predict_crop(valid_input)
        # crop is capitalised, top3 items may differ in case — compare normalised
        assert result["crop"].lower() == result["top3"][0]["crop"].lower()

    def test_raises_on_invalid_input(self, valid_input):
        from predict import predict_crop
        bad = {**valid_input, "ph": 99}
        with pytest.raises(ValueError):
            predict_crop(bad)

    def test_raises_when_no_artefacts(self, valid_input):
        """When module cache is cleared AND no .pkl files exist, must raise FileNotFoundError."""
        import predict as pm
        from unittest.mock import patch
        predict_module._model         = None
        predict_module._scaler        = None
        predict_module._label_encoder = None
        # Patch Path.exists to simulate missing files
        with patch("predict.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                pm._load_artefacts()

    def test_deterministic(self, valid_input):
        """Same input should always return the same crop."""
        from predict import predict_crop
        r1 = predict_crop(valid_input)
        r2 = predict_crop(valid_input)
        assert r1["crop"] == r2["crop"]
        assert r1["confidence"] == r2["confidence"]


# ══════════════════════════════════════════════════════════════════════════════
# get_model_info
# ══════════════════════════════════════════════════════════════════════════════

class TestGetModelInfo:

    @pytest.fixture(autouse=True)
    def inject_artefacts(self, trained_artefacts):
        model, scaler, le = trained_artefacts
        predict_module._model         = model
        predict_module._scaler        = scaler
        predict_module._label_encoder = le
        yield
        predict_module._model         = None
        predict_module._scaler        = None
        predict_module._label_encoder = None

    def test_returns_dict(self):
        from predict import get_model_info
        info = get_model_info()
        assert isinstance(info, dict)

    def test_has_model_type(self):
        from predict import get_model_info
        info = get_model_info()
        assert "model_type" in info
        assert info["model_type"] == "RandomForestClassifier"

    def test_has_n_classes(self):
        from predict import get_model_info
        info = get_model_info()
        assert "n_classes" in info
        assert info["n_classes"] == 22

    def test_has_features(self):
        from predict import get_model_info
        info = get_model_info()
        assert "features" in info
        assert info["features"] == FEATURE_ORDER

    def test_has_class_names(self):
        from predict import get_model_info
        info = get_model_info()
        assert "class_names" in info
        assert isinstance(info["class_names"], list)
        assert len(info["class_names"]) == 22
