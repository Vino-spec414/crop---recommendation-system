"""
conftest.py
===========
Shared pytest fixtures used across all test modules.

Design principle:
  Fixtures are the single source of truth for test data.
  Each test function receives clean, isolated state — no shared mutable globals.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Make src/ importable from tests/ ─────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_preprocessing import FEATURE_COLUMNS, TARGET_COLUMN


# ── Dataset fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_df():
    """
    A minimal but realistic synthetic DataFrame that mirrors the real dataset.
    'session' scope means it is created once and reused across all tests in
    the session — cheap to compute and safe because no test mutates it.
    """
    rng = np.random.default_rng(42)
    crops = [
        "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas",
        "mothbeans", "mungbean", "blackgram", "lentil", "pomegranate",
        "banana", "mango", "grapes", "watermelon", "muskmelon",
        "apple", "orange", "papaya", "coconut", "cotton", "jute", "coffee",
    ]
    n_per = 10          # 10 records per class → 220 rows total
    rows = []
    for crop in crops:
        for _ in range(n_per):
            rows.append({
                "N":           float(rng.integers(0, 140)),
                "P":           float(rng.integers(5, 145)),
                "K":           float(rng.integers(5, 205)),
                "temperature": round(float(rng.uniform(8, 44)), 4),
                "humidity":    round(float(rng.uniform(14, 100)), 4),
                "ph":          round(float(rng.uniform(3.5, 9.5)), 4),
                "rainfall":    round(float(rng.uniform(20, 300)), 4),
                "label":       crop,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def dirty_df(sample_df):
    """
    Introduces controlled impurities into the clean sample fixture:
      - 3 duplicate rows (copied from rows 50-52, well clear of NaN-injected rows)
      - 3 NaN values in 'N' (rows 0, 10, 20 — these are NOT the duplicated rows)
      - 2 out-of-bound pH values
      - Whitespace in some label values
    """
    df = sample_df.copy()
    # Duplicates — copy rows 50-52 (far from NaN rows 0,10,20)
    df = pd.concat([df, df.iloc[50:53]], ignore_index=True)
    # Missing values in rows that are NOT the duplicates
    df.loc[[0, 10, 20], "N"] = np.nan
    # Out-of-bound pH
    df.loc[[5, 15], "ph"] = 15.5
    # Dirty labels
    df.loc[[1, 2], "label"] = "  Rice  "
    return df


@pytest.fixture(scope="session")
def valid_input():
    """A single valid prediction input (rice-like conditions)."""
    return {
        "N":           90.0,
        "P":           42.0,
        "K":           43.0,
        "temperature": 20.87,
        "humidity":    82.0,
        "ph":          6.5,
        "rainfall":    202.93,
    }


@pytest.fixture(scope="session")
def trained_artefacts(sample_df):
    """
    Train a minimal Random Forest on the sample data and return
    (model, scaler, label_encoder).  Used by predict tests.
    This fixture is session-scoped so training happens only once.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from data_preprocessing import clean_dataset, validate_schema

    validate_schema(sample_df)
    df = clean_dataset(sample_df)

    le = LabelEncoder()
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = le.fit_transform(df[TARGET_COLUMN])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
    model.fit(X_scaled, y)

    return model, scaler, le
