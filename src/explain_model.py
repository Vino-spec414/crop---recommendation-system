"""
explain_model.py
================
Model explainability using SHAP (SHapley Additive exPlanations).

EXPLAINER COMPATIBILITY:
  ┌─────────────────────────────┬──────────────────────────────────────┐
  │ Model                       │ SHAP Explainer used                  │
  ├─────────────────────────────┼──────────────────────────────────────┤
  │ RandomForestClassifier      │ TreeExplainer  (exact, fast)         │
  │ DecisionTreeClassifier      │ TreeExplainer  (exact, fast)         │
  │ GradientBoostingClassifier  │ KernelExplainer (model-agnostic)     │
  │                             │ ← TreeExplainer only supports binary │
  └─────────────────────────────┴──────────────────────────────────────┘

Run:
    python src/explain_model.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preprocessing import (
    load_crop_dataset, validate_schema, clean_dataset,
    FEATURE_COLUMNS, TARGET_COLUMN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

FEATURE_LABELS = {
    "N":           "Nitrogen (N)",
    "P":           "Phosphorus (P)",
    "K":           "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity":    "Humidity (%)",
    "ph":          "Soil pH",
    "rainfall":    "Rainfall (mm)",
}

# Only these support TreeExplainer for multiclass
_TREE_SUPPORTED = (
    "RandomForestClassifier",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
)

def load_artefacts():
    for name in ("crop_model.pkl", "scaler.pkl", "label_encoder.pkl"):
        if not (MODELS_DIR / name).exists():
            raise FileNotFoundError(
                f"Missing: {MODELS_DIR / name}  —  run 'python src/train_model.py' first."
            )
    model  = joblib.load(MODELS_DIR / "crop_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    le     = joblib.load(MODELS_DIR / "label_encoder.pkl")
    logger.info("Loaded model: %s", type(model).__name__)
    return model, scaler, le


def _get_explainer(model, X_background: np.ndarray):
    """
    Route to the correct SHAP explainer based on model type.
    Returns (explainer, use_kernel_flag).

    For KernelExplainer, the background must have at least as many rows
    as the number of k-means clusters we want.  For very small backgrounds
    (e.g. when called for a single prediction) we pass the data directly.
    """
    import shap
    model_name = type(model).__name__

    if model_name in _TREE_SUPPORTED:
        logger.info("Using TreeExplainer for %s", model_name)
        return shap.TreeExplainer(model), False
    else:
        logger.info(
            "%s not supported by TreeExplainer for multiclass — "
            "using KernelExplainer (slower). "
            "Retrain with Random Forest for best SHAP performance.",
            model_name,
        )
        # k-means requires n_samples >= n_clusters.  Use min(30, n_rows).
        n_clusters = min(30, len(X_background))
        if n_clusters < 2:
            # Only 1 background row — pass it directly as the background
            background = X_background
        else:
            background = shap.kmeans(X_background, n_clusters)
        return shap.KernelExplainer(model.predict_proba, background), True


def _to_3d(shap_values) -> np.ndarray:
    """
    Normalise any SHAP output format to shape (n_samples, n_features, n_classes).

    SHAP returns different shapes depending on model + explainer version:
      TreeExplainer + RF  → list of (n_samples, n_features) — one per class
      TreeExplainer + RF  → (n_samples, n_features, n_classes)  newer versions
      KernelExplainer     → list of (n_samples, n_features) — one per class
    """
    if isinstance(shap_values, list):
        # List of 2-D arrays — stack along class axis
        return np.stack(shap_values, axis=2)   # (n, feats, classes)

    sv = np.array(shap_values)
    if sv.ndim == 3:
        return sv                              # already (n, feats, classes)
    if sv.ndim == 2:
        return sv[:, :, np.newaxis]           # binary — add class dim
    raise ValueError(f"Unexpected SHAP value shape: {sv.shape}")


def compute_shap_values(model, X_scaled: np.ndarray, sample_size: int = 300):
    """Compute SHAP values with the correct explainer for this model."""
    try:
        import shap
    except ImportError:
        raise ImportError("Run:  pip install shap")

    rng      = np.random.default_rng(42)
    n        = min(sample_size, len(X_scaled))
    idx      = rng.choice(len(X_scaled), size=n, replace=False)
    X_sample = X_scaled[idx]

    logger.info("Computing SHAP values on %d samples...", n)
    explainer, use_kernel = _get_explainer(model, X_scaled)

    if use_kernel:
        # KernelExplainer is O(n²) — use a smaller sample
        kernel_n  = min(100, n)
        X_sample  = X_sample[:kernel_n]
        shap_vals = explainer.shap_values(X_sample, nsamples=100)
        logger.info("KernelExplainer finished on %d samples.", kernel_n)
    else:
        shap_vals = explainer.shap_values(X_sample)

    logger.info("SHAP values computed.")
    return shap_vals, X_sample, explainer


def plot_shap_summary(shap_values, X_sample, class_names, save_dir):
    """Global feature importance: mean |SHAP| across all samples and classes."""
    feature_labels = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLUMNS]

    sv_3d = _to_3d(shap_values)                      # (n, feats, classes)
    mean_importance = np.abs(sv_3d).mean(axis=(0, 2)) # (n_features,)

    sorted_idx = np.argsort(mean_importance)

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = sns.color_palette("viridis", len(FEATURE_COLUMNS))
    bars = ax.barh(
        [feature_labels[i] for i in sorted_idx],
        mean_importance[sorted_idx],
        color=[colours[i] for i in sorted_idx],
        edgecolor="white",
    )
    ax.set_xlabel("Mean |SHAP Value| — average impact on model output")
    ax.set_title("Global Feature Importance — SHAP Values", fontsize=13, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    offset = max(mean_importance) * 0.01
    for bar in bars:
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va="center", fontsize=9)

    plt.tight_layout()
    path = save_dir / "shap_global_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", path)


def plot_shap_class_breakdown(shap_values, class_names, save_dir):
    """Per-class feature importance heatmap."""
    feature_labels = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLUMNS]

    sv_3d    = _to_3d(shap_values)                   # (n, feats, classes)
    mean_abs = np.abs(sv_3d).mean(axis=0).T           # (classes, feats)

    if mean_abs.shape[0] != len(class_names):
        logger.warning(
            "SHAP class count (%d) != class_names (%d) — skipping heatmap.",
            mean_abs.shape[0], len(class_names),
        )
        return

    df_heat = pd.DataFrame(mean_abs, index=class_names, columns=feature_labels)
    df_norm = df_heat.div(df_heat.max(axis=0).replace(0, 1))

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(
        df_norm, annot=df_heat.round(4), fmt=".4f", cmap="YlGn",
        linewidths=0.4, ax=ax,
        cbar_kws={"label": "Normalised SHAP Importance"},
        annot_kws={"size": 7},
    )
    ax.set_title("SHAP Feature Importance by Crop Class", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Crop")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = save_dir / "shap_class_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", path)


def explain_single_prediction(model, scaler, le, features: dict, save_dir: Path,
                               X_background: np.ndarray = None) -> str:
    """
    Waterfall-style bar chart for a single prediction.

    Args:
        X_background: Optional reference dataset for KernelExplainer's
                      background distribution.  If None, a tiny synthetic
                      background is created from the input row.
    """
    try:
        import shap
    except ImportError:
        return ""

    feature_labels = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLUMNS]

    x_raw          = np.array([features[f] for f in FEATURE_COLUMNS]).reshape(1, -1)
    x_scaled       = scaler.transform(x_raw)
    pred_class_idx = int(model.predict(x_scaled)[0])
    pred_crop      = le.inverse_transform([pred_class_idx])[0].capitalize()

    # Use the supplied background (proper full dataset) if available
    bg = X_background if X_background is not None else x_scaled
    explainer, use_kernel = _get_explainer(model, bg)
    if use_kernel:
        shap_vals = explainer.shap_values(x_scaled, nsamples=200)
    else:
        shap_vals = explainer.shap_values(x_scaled)

    sv_3d = _to_3d(shap_vals)    # (1, n_features, n_classes)
    class_axis = min(pred_class_idx, sv_3d.shape[2] - 1)
    sv = sv_3d[0, :, class_axis]

    sorted_idx = np.argsort(np.abs(sv))
    colours    = ["#5C8A44" if v > 0 else "#C4692A" for v in sv[sorted_idx]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feature_labels[i] for i in sorted_idx], sv[sorted_idx],
            color=colours, edgecolor="white")
    ax.axvline(0, color="#2C1A0E", linewidth=0.8)
    ax.set_title(f"SHAP Single Prediction — Predicted: {pred_crop}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("SHAP Value  (green = towards crop,  orange = away from crop)")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = save_dir / f"shap_single_{pred_crop.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s  (predicted: %s)", path, pred_crop)
    return pred_crop


def run_explainability(data_path=None) -> None:
    print("\n" + "=" * 60)
    print("  🔍  MODEL EXPLAINABILITY — SHAP ANALYSIS")
    print("=" * 60 + "\n")

    model, scaler, le = load_artefacts()
    class_names = list(le.classes_)
    model_name  = type(model).__name__

    print(f"  Model  : {model_name}")
    print(f"  Classes: {len(class_names)}")

    if model_name not in _TREE_SUPPORTED:
        print(
            f"\n  ⚠️  {model_name} requires KernelExplainer (slower fallback).\n"
            "     For faster SHAP, retrain with Random Forest:\n"
            "       python src/train_model.py\n"
        )

    raw = load_crop_dataset(data_path)
    validate_schema(raw)
    df = clean_dataset(raw)

    X        = df[FEATURE_COLUMNS].values.astype(np.float64)
    X_scaled = scaler.transform(X)

    shap_values, X_sample, _ = compute_shap_values(model, X_scaled, sample_size=300)

    plot_shap_summary(shap_values, X_sample, class_names, MODELS_DIR)
    plot_shap_class_breakdown(shap_values, class_names, MODELS_DIR)

    sample_input = {
        "N": 90, "P": 42, "K": 43,
        "temperature": 22.5, "humidity": 85.0,
        "ph": 6.5, "rainfall": 220.0,
    }
    predicted = explain_single_prediction(
        model, scaler, le, sample_input, MODELS_DIR,
        X_background=X_scaled,
    )
    if predicted:
        print(f"\n  Sample prediction: {predicted}")

    print(f"\n✅  Done. Charts saved to: {MODELS_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_explainability()

