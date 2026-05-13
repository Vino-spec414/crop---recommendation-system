"""
tune_model.py
=============
Hyperparameter tuning for the Crop Recommendation System.

Uses GridSearchCV and RandomizedSearchCV to find optimal parameters
for each model. Results are logged and the best tuned model is saved.

WHY THIS MATTERS FOR DISSERTATION:
  Demonstrating hyperparameter tuning shows rigorous ML methodology.
  It separates a proper research-grade pipeline from a basic demo.

Run:
    python src/tune_model.py
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, train_test_split,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preprocessing import (
    load_crop_dataset, validate_schema, clean_dataset,
    FEATURE_COLUMNS, TARGET_COLUMN,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5


# ── Parameter Grids ───────────────────────────────────────────────────────────
# Note: For dissertation use GridSearchCV on smaller grids for reproducibility.
# For production, use RandomizedSearchCV on wider ranges.

PARAM_GRIDS = {
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "method": "grid",
        "params": {
            "max_depth":        [5, 10, 15, None],
            "min_samples_split":[2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion":        ["gini", "entropy"],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "method": "random",          # RandomizedSearchCV — faster for large grids
        "n_iter": 30,
        "params": {
            "n_estimators":      [100, 200, 300, 500],
            "max_depth":         [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "max_features":      ["sqrt", "log2", None],
        },
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "method": "random",
        "n_iter": 20,
        "params": {
            "n_estimators":  [100, 200, 300],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth":     [3, 5, 7],
            "subsample":     [0.8, 1.0],
        },
    },
}


def run_tuning(data_path=None) -> dict:
    """
    Run hyperparameter search for all models and return tuned estimators.
    """
    print("\n" + "=" * 60)
    print("  🔬  HYPERPARAMETER TUNING PIPELINE")
    print("=" * 60 + "\n")

    # ── Data ──────────────────────────────────────────────────────
    df = clean_dataset(validate_schema(load_crop_dataset(data_path)) or load_crop_dataset(data_path))
    # Simpler inline:
    raw = load_crop_dataset(data_path)
    validate_schema(raw)
    df = clean_dataset(raw)

    le = LabelEncoder()
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = le.fit_transform(df[TARGET_COLUMN])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Tuning ────────────────────────────────────────────────────
    results = {}

    for name, cfg in PARAM_GRIDS.items():
        logger.info("Tuning %s (%s search)...", name, cfg["method"].upper())
        print(f"\n  ▶ {name} — {cfg['method'].upper()} search")

        if cfg["method"] == "grid":
            search = GridSearchCV(
                cfg["estimator"], cfg["params"],
                cv=cv, scoring="f1_macro",
                n_jobs=-1, verbose=0,
            )
        else:
            search = RandomizedSearchCV(
                cfg["estimator"], cfg["params"],
                n_iter=cfg.get("n_iter", 20),
                cv=cv, scoring="f1_macro",
                n_jobs=-1, verbose=0,
                random_state=RANDOM_STATE,
            )

        search.fit(X_train, y_train)

        best = search.best_estimator_
        y_pred = best.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average="macro")

        logger.info(
            "%s | Best CV F1: %.4f | Test F1: %.4f",
            name, search.best_score_, test_f1
        )
        print(f"     Best params : {search.best_params_}")
        print(f"     CV F1 Score : {search.best_score_:.4f}")
        print(f"     Test F1     : {test_f1:.4f}")

        results[name] = {
            "model":       best,
            "best_params": search.best_params_,
            "cv_f1":       search.best_score_,
            "test_f1":     test_f1,
            "cv_results":  pd.DataFrame(search.cv_results_),
        }

    # ── Select & save best ────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["test_f1"])
    best_model = results[best_name]["model"]

    print(f"\n🏆  Best tuned model: {best_name} (Test F1 = {results[best_name]['test_f1']:.4f})")
    print("\n── Final Classification Report ──")
    y_pred_final = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_final, target_names=le.classes_))

    # Save tuned artefacts (overwrites baseline)
    joblib.dump(best_model, MODELS_DIR / "crop_model.pkl")
    joblib.dump(scaler,     MODELS_DIR / "scaler.pkl")
    joblib.dump(le,         MODELS_DIR / "label_encoder.pkl")

    # Save tuning summary
    summary = pd.DataFrame([
        {
            "Model":       n,
            "Best CV F1":  f"{r['cv_f1']:.4f}",
            "Test F1":     f"{r['test_f1']:.4f}",
            "Best Params": str(r["best_params"]),
        }
        for n, r in results.items()
    ])
    summary.to_csv(MODELS_DIR / "tuning_summary.csv", index=False)

    # Plot tuning results
    _plot_tuning_comparison(results, MODELS_DIR)

    print("\n✅  Tuning complete. Best model saved to /models/")
    print("=" * 60 + "\n")
    return results


def _plot_tuning_comparison(results: dict, save_dir: Path) -> None:
    """Bar chart: baseline vs tuned F1 scores."""
    names    = list(results.keys())
    cv_f1s   = [results[n]["cv_f1"]   for n in names]
    test_f1s = [results[n]["test_f1"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, cv_f1s,   width, label="CV F1",   color="#5C8A44", edgecolor="white")
    bars2 = ax.bar(x + width/2, test_f1s, width, label="Test F1", color="#3D7A8A", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("F1 Score (Macro)")
    ax.set_title("Tuned Model Performance — CV vs Test", fontsize=13, fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)

    for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.003, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.003, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = save_dir / "tuning_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved tuning comparison chart → %s", path)


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_tuning(data_path)
