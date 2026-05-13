"""
train_model.py
==============
Full ML training pipeline for the Crop Recommendation System.

Pipeline stages:
  1. Load & clean data          (via data_preprocessing)
  2. Feature preparation        (scaling)
  3. Train/test split
  4. Train multiple models
  5. Evaluate & compare
  6. Persist best model artefacts

Run directly:
    python src/train_model.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# ── Ensure src/ is importable regardless of working directory ─────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preprocessing import (
    load_crop_dataset, validate_schema, clean_dataset,
    get_summary_stats, get_class_distribution,
    FEATURE_COLUMNS, TARGET_COLUMN, DATA_DIR,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH  = MODELS_DIR / "crop_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_PATH  = MODELS_DIR / "label_encoder.pkl"

# ── Hyperparameters (centralised — easy to tune) ──────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

MODELS = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
    ),
}


# ── Helper utilities ──────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    """
    Split the DataFrame into features (X) and encoded target (y).
    Also returns the fitted LabelEncoder so we can decode predictions later.
    """
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COLUMN])
    return X, y, le


def evaluate_model(model, X_test, y_test, class_names) -> dict:
    """
    Run all evaluation metrics on a trained model.
    Returns a results dictionary.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="macro", zero_division=0),
        "y_pred":    y_pred,
        "report":    classification_report(y_test, y_pred, target_names=class_names),
        "cm":        confusion_matrix(y_test, y_pred),
    }


# ── Visualisation helpers ─────────────────────────────────────────────────────

def plot_comparison_bar(results: dict, save_dir: Path) -> None:
    """Bar chart comparing all models across 4 metrics."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Model Comparison — Evaluation Metrics", fontsize=14, fontweight="bold")

    colours = sns.color_palette("viridis", len(model_names))

    for ax, metric in zip(axes, metrics):
        values = [results[m][metric] for m in model_names]
        bars = ax.bar(model_names, values, color=colours, edgecolor="white", linewidth=0.8)
        ax.set_title(metric.capitalize(), fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    plt.tight_layout()
    path = save_dir / "model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved model comparison chart → %s", path)


def plot_confusion_matrix(cm, class_names: list, model_name: str, save_dir: Path) -> None:
    """Annotated heatmap confusion matrix."""
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = save_dir / f"confusion_matrix_{safe_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix → %s", path)


def plot_feature_importance(model, model_name: str, save_dir: Path) -> None:
    """Horizontal bar chart of feature importances (tree-based models only)."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)
    features = [FEATURE_COLUMNS[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features, importances[indices], color=sns.color_palette("viridis", len(features)))
    ax.set_title(f"Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = save_dir / f"feature_importance_{safe_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved feature importance chart → %s", path)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(data_path=None) -> dict:
    """
    Execute the full ML pipeline end-to-end.

    Returns:
        dict with best_model, scaler, label_encoder, results, class_names
    """
    print("\n" + "=" * 60)
    print("  🌾  CROP RECOMMENDATION — ML TRAINING PIPELINE")
    print("=" * 60 + "\n")

    # ── 1. Load & clean ───────────────────────────────────────────
    logger.info("STAGE 1: Data Loading & Cleaning")
    raw_df = load_crop_dataset(data_path)
    validate_schema(raw_df)
    df = clean_dataset(raw_df)

    print("\n── Summary Statistics ──")
    print(get_summary_stats(df).to_string())

    print("\n── Class Distribution ──")
    print(get_class_distribution(df).to_string())

    # ── 2. Feature preparation ────────────────────────────────────
    logger.info("STAGE 2: Feature Preparation")
    X, y, label_encoder = prepare_features(df)
    class_names = list(label_encoder.classes_)
    logger.info("Classes: %s", class_names)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 3. Train / test split ─────────────────────────────────────
    logger.info("STAGE 3: Train/Test Split (%.0f%% test)", TEST_SIZE * 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,           # preserves class proportions in both splits
    )
    logger.info("Train: %d samples | Test: %d samples", len(X_train), len(X_test))

    # ── 4. Train & evaluate all models ───────────────────────────
    logger.info("STAGE 4: Model Training & Evaluation")
    results = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in MODELS.items():
        print(f"\n  ▶ Training {name} ...")
        model.fit(X_train, y_train)

        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        logger.info(
            "%s | CV accuracy: %.4f ± %.4f",
            name, cv_scores.mean(), cv_scores.std(),
        )

        # Hold-out test set evaluation
        res = evaluate_model(model, X_test, y_test, class_names)
        res["cv_mean"] = cv_scores.mean()
        res["cv_std"]  = cv_scores.std()
        res["model"]   = model
        results[name]  = res

        print(f"     Accuracy:  {res['accuracy']:.4f}")
        print(f"     F1 Score:  {res['f1']:.4f}")
        print(f"     CV Score:  {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")

    # ── 5. Model comparison table ─────────────────────────────────
    logger.info("STAGE 5: Model Comparison")
    comparison = pd.DataFrame([
        {
            "Model":     name,
            "Accuracy":  f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall":    f"{r['recall']:.4f}",
            "F1 Score":  f"{r['f1']:.4f}",
            "CV Mean":   f"{r['cv_mean']:.4f}",
            "CV Std":    f"±{r['cv_std']:.4f}",
        }
        for name, r in results.items()
    ])
    print("\n── Model Comparison ──")
    print(comparison.to_string(index=False))

    # ── 6. Select best model (by F1 macro) ───────────────────────
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_result = results[best_name]
    best_model  = best_result["model"]
    logger.info("Best model: %s (F1 = %.4f)", best_name, best_result["f1"])

    print(f"\n🏆  Best Model: {best_name}")
    print(f"\n── Classification Report ({best_name}) ──")
    print(best_result["report"])

    # ── 7. Visualisations ─────────────────────────────────────────
    logger.info("STAGE 6: Generating Visualisations → %s", MODELS_DIR)
    plot_comparison_bar(results, MODELS_DIR)
    for name, res in results.items():
        plot_confusion_matrix(res["cm"], class_names, name, MODELS_DIR)
        plot_feature_importance(res["model"], name, MODELS_DIR)

    # ── 8. Save artefacts ─────────────────────────────────────────
    logger.info("STAGE 7: Saving Model Artefacts")
    joblib.dump(best_model,    MODEL_PATH)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(label_encoder, LABEL_PATH)

    logger.info("Model saved  → %s", MODEL_PATH)
    logger.info("Scaler saved → %s", SCALER_PATH)
    logger.info("Labels saved → %s", LABEL_PATH)

    print("\n✅  Pipeline complete. Artefacts saved to /models/")
    print("=" * 60 + "\n")

    return {
        "best_model":     best_model,
        "best_name":      best_name,
        "scaler":         scaler,
        "label_encoder":  label_encoder,
        "results":        results,
        "class_names":    class_names,
    }


if __name__ == "__main__":
    # Allow optional path argument:  python src/train_model.py path/to/data.csv
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(data_path)
    
