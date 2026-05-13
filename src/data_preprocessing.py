"""
data_preprocessing.py
=====================
Handles all data loading, cleaning, and validation for the
Crop Recommendation System.

Design principle: Each function does ONE thing and is independently testable.
This makes debugging easier and the code reusable across notebooks and scripts.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"

# Sensible domain bounds for validation (agronomic literature)
FEATURE_BOUNDS = {
    "N":           (0, 200),
    "P":           (0, 200),
    "K":           (0, 250),
    "temperature": (0, 50),
    "humidity":    (0, 100),
    "ph":          (0, 14),
    "rainfall":    (0, 3000),
}


# ── Public API ────────────────────────────────────────────────────────────────

def load_crop_dataset(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load the primary Crop Recommendation dataset.

    Args:
        filepath: Optional explicit path. Defaults to data/crop_dataset.csv.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If the CSV cannot be found.
    """
    path = Path(filepath) if filepath else DATA_DIR / "crop_dataset.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please download it from:\n"
            "  https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset\n"
            f"and place it in: {DATA_DIR}"
        )

    df = pd.read_csv(path)
    logger.info("Loaded crop dataset: %d rows × %d columns from '%s'", *df.shape, path.name)
    return df


def load_soil_dataset(filepath: str | Path | None = None) -> pd.DataFrame | None:
    """
    Load the secondary Soil Physical & Chemical Properties dataset.
    Returns None (with a warning) if the file is absent, so the rest of
    the pipeline can still run on the primary dataset alone.
    """
    path = Path(filepath) if filepath else DATA_DIR / "soil_dataset.csv"

    if not path.exists():
        logger.warning(
            "Secondary soil dataset not found at '%s'. "
            "EDA soil analysis will be skipped.",
            path,
        )
        return None

    df = pd.read_csv(path)
    logger.info("Loaded soil dataset: %d rows × %d columns", *df.shape)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Assert that all required columns are present.

    Raises:
        ValueError: If any expected column is missing.
    """
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    logger.info("Schema validation passed ✓")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a reproducible cleaning pipeline:
      1. Drop exact duplicate rows
      2. Drop rows with missing values in key columns
      3. Strip whitespace from the target label
      4. Flag and remove out-of-bound feature values

    Args:
        df: Raw DataFrame (must pass validate_schema first).

    Returns:
        Cleaned DataFrame with a reset index.
    """
    initial_count = len(df)
    logger.info("Starting cleaning — %d rows", initial_count)

    # 1. Duplicates
    df = df.drop_duplicates()
    logger.info("After duplicate removal: %d rows (dropped %d)", len(df), initial_count - len(df))

    # 2. Missing values
    key_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    before = len(df)
    df = df.dropna(subset=key_cols)
    logger.info("After missing-value drop: %d rows (dropped %d)", len(df), before - len(df))

    # 3. Clean target label
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.lower()

    # 4. Out-of-bound values
    mask = pd.Series([True] * len(df), index=df.index)
    for col, (lo, hi) in FEATURE_BOUNDS.items():
        if col in df.columns:
            out = ~df[col].between(lo, hi)
            if out.sum():
                logger.warning("  '%s': %d out-of-bound values removed", col, out.sum())
            mask &= ~out

    df = df[mask].reset_index(drop=True)
    logger.info(
        "Cleaning complete. Final: %d rows (total removed: %d)",
        len(df),
        initial_count - len(df),
    )
    return df


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for numeric feature columns.
    Useful for EDA and dissertation reporting.
    """
    stats = df[FEATURE_COLUMNS].describe().T
    stats["skewness"] = df[FEATURE_COLUMNS].skew()
    stats["kurtosis"] = df[FEATURE_COLUMNS].kurt()
    return stats.round(4)


def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    """Return count and percentage of each crop class."""
    counts = df[TARGET_COLUMN].value_counts()
    pct = (counts / len(df) * 100).round(2)
    summary = pd.DataFrame({"count": counts, "pct_%": pct})
    return summary


# ── Main guard (quick sanity check) ───────────────────────────────────────────
if __name__ == "__main__":
    print("\n📂 Running data_preprocessing.py as a standalone sanity check...\n")
    try:
        raw = load_crop_dataset()
        validate_schema(raw)
        clean = clean_dataset(raw)

        print("\n── Summary Statistics ──")
        print(get_summary_stats(clean))

        print("\n── Class Distribution ──")
        print(get_class_distribution(clean))

    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("\nGenerating a tiny synthetic sample so you can test the code:\n")

        # Synthetic stand-in so the rest of the pipeline is testable
        rng = np.random.default_rng(42)
        crops = ["rice", "maize", "wheat", "mango", "cotton"]
        n = 500
        synthetic = pd.DataFrame({
            "N":           rng.integers(0, 140, n),
            "P":           rng.integers(5, 145, n),
            "K":           rng.integers(5, 205, n),
            "temperature": rng.uniform(8, 44, n).round(2),
            "humidity":    rng.uniform(14, 100, n).round(2),
            "ph":          rng.uniform(3.5, 9.5, n).round(2),
            "rainfall":    rng.uniform(20, 300, n).round(2),
            "label":       rng.choice(crops, n),
        })
        save_path = DATA_DIR / "crop_dataset_synthetic.csv"
        DATA_DIR.mkdir(exist_ok=True)
        synthetic.to_csv(save_path, index=False)
        print(f"Synthetic dataset saved to: {save_path}")
        print("Use this for development. Replace with the real dataset before submission.")

