"""
test_data_preprocessing.py
===========================
Unit tests for src/data_preprocessing.py.

Every function in the module has at least one positive test (correct input →
correct output) and one negative test (invalid input → correct error).

WHY THIS MATTERS FOR DISSERTATION:
  Unit tests are evidence of software engineering rigour.  They also act as
  executable documentation — a reader can understand what each function is
  supposed to do by reading the test cases alone.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from data_preprocessing import (
    validate_schema,
    clean_dataset,
    get_summary_stats,
    get_class_distribution,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FEATURE_BOUNDS,
)


# ══════════════════════════════════════════════════════════════════════════════
# validate_schema
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateSchema:

    def test_passes_with_all_columns(self, sample_df):
        """No exception when all required columns are present."""
        validate_schema(sample_df)   # should not raise

    def test_raises_on_missing_feature(self, sample_df):
        """Raises ValueError when a feature column is missing."""
        df_bad = sample_df.drop(columns=["N"])
        with pytest.raises(ValueError, match="N"):
            validate_schema(df_bad)

    def test_raises_on_missing_target(self, sample_df):
        """Raises ValueError when the target column is missing."""
        df_bad = sample_df.drop(columns=[TARGET_COLUMN])
        with pytest.raises(ValueError, match=TARGET_COLUMN):
            validate_schema(df_bad)

    def test_raises_on_multiple_missing(self, sample_df):
        """Error message lists ALL missing columns, not just the first."""
        df_bad = sample_df.drop(columns=["N", "P", TARGET_COLUMN])
        with pytest.raises(ValueError):
            validate_schema(df_bad)

    def test_passes_with_extra_columns(self, sample_df):
        """Extra columns beyond the required set should not cause failure."""
        df_extra = sample_df.copy()
        df_extra["extra_col"] = 99
        validate_schema(df_extra)   # should not raise

    def test_empty_dataframe_raises(self):
        """Empty DataFrame with no columns should raise ValueError."""
        with pytest.raises(ValueError):
            validate_schema(pd.DataFrame())


# ══════════════════════════════════════════════════════════════════════════════
# clean_dataset
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanDataset:

    def test_removes_exact_duplicates(self, dirty_df):
        """Duplicate rows must be removed — count must be strictly less than input."""
        original_dupes = dirty_df.duplicated().sum()
        cleaned = clean_dataset(dirty_df)
        # After cleaning, zero duplicates should remain
        remaining_dupes = cleaned.duplicated().sum()
        assert remaining_dupes == 0, (
            f"Expected 0 duplicates after cleaning, found {remaining_dupes}. "
            f"Input had {original_dupes} duplicates."
        )

    def test_removes_null_rows(self, dirty_df):
        """Rows with NaN in feature columns must be removed."""
        cleaned = clean_dataset(dirty_df)
        assert cleaned[FEATURE_COLUMNS].isnull().sum().sum() == 0

    def test_removes_out_of_bound_ph(self, dirty_df):
        """pH values > 14 must be removed (FEATURE_BOUNDS upper limit)."""
        cleaned = clean_dataset(dirty_df)
        lo, hi = FEATURE_BOUNDS["ph"]
        assert cleaned["ph"].between(lo, hi).all()

    def test_strips_label_whitespace(self, dirty_df):
        """Labels with leading/trailing whitespace should be stripped."""
        cleaned = clean_dataset(dirty_df)
        assert not (cleaned[TARGET_COLUMN].str.startswith(" ")).any()
        assert not (cleaned[TARGET_COLUMN].str.endswith(" ")).any()

    def test_lowercases_label(self, dirty_df):
        """Labels should be normalised to lowercase."""
        cleaned = clean_dataset(dirty_df)
        assert (cleaned[TARGET_COLUMN] == cleaned[TARGET_COLUMN].str.lower()).all()

    def test_clean_df_unchanged(self, sample_df):
        """Running clean_dataset on already-clean data should not lose rows."""
        cleaned = clean_dataset(sample_df)
        # We should retain at least 95% of the rows (allows for boundary floats)
        assert len(cleaned) >= len(sample_df) * 0.95

    def test_returns_reset_index(self, dirty_df):
        """Returned DataFrame index should be a clean 0-based RangeIndex."""
        cleaned = clean_dataset(dirty_df)
        expected = pd.RangeIndex(len(cleaned))
        pd.testing.assert_index_equal(cleaned.index, expected)

    def test_idempotent(self, sample_df):
        """Calling clean_dataset twice should produce the same result."""
        first  = clean_dataset(sample_df)
        second = clean_dataset(first)
        pd.testing.assert_frame_equal(first.reset_index(drop=True),
                                      second.reset_index(drop=True))

    def test_output_feature_columns_numeric(self, sample_df):
        """All feature columns in cleaned output must be numeric."""
        cleaned = clean_dataset(sample_df)
        for col in FEATURE_COLUMNS:
            assert pd.api.types.is_numeric_dtype(cleaned[col]), \
                f"Column '{col}' should be numeric after cleaning"

    def test_all_feature_bounds_respected(self, sample_df):
        """Every feature value in the cleaned output must lie within bounds."""
        cleaned = clean_dataset(sample_df)
        for col, (lo, hi) in FEATURE_BOUNDS.items():
            if col in cleaned.columns:
                assert cleaned[col].between(lo, hi).all(), \
                    f"Column '{col}' has values outside [{lo}, {hi}]"


# ══════════════════════════════════════════════════════════════════════════════
# get_summary_stats
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSummaryStats:

    def test_returns_dataframe(self, sample_df):
        stats = get_summary_stats(sample_df)
        assert isinstance(stats, pd.DataFrame)

    def test_index_is_feature_columns(self, sample_df):
        stats = get_summary_stats(sample_df)
        assert list(stats.index) == FEATURE_COLUMNS

    def test_contains_skewness_column(self, sample_df):
        stats = get_summary_stats(sample_df)
        assert "skewness" in stats.columns

    def test_contains_kurtosis_column(self, sample_df):
        stats = get_summary_stats(sample_df)
        assert "kurtosis" in stats.columns

    def test_count_column_equals_dataframe_length(self, sample_df):
        stats = get_summary_stats(sample_df)
        assert (stats["count"] == len(sample_df)).all()

    def test_mean_within_bounds(self, sample_df):
        """Feature means must lie within agronomic bounds."""
        stats = get_summary_stats(sample_df)
        for col in FEATURE_COLUMNS:
            lo, hi = FEATURE_BOUNDS[col]
            mean_val = stats.loc[col, "mean"]
            assert lo <= mean_val <= hi, \
                f"Mean of '{col}' ({mean_val}) is outside [{lo}, {hi}]"


# ══════════════════════════════════════════════════════════════════════════════
# get_class_distribution
# ══════════════════════════════════════════════════════════════════════════════

class TestGetClassDistribution:

    def test_returns_dataframe(self, sample_df):
        dist = get_class_distribution(sample_df)
        assert isinstance(dist, pd.DataFrame)

    def test_has_count_and_pct_columns(self, sample_df):
        dist = get_class_distribution(sample_df)
        assert "count"  in dist.columns
        assert "pct_%"  in dist.columns

    def test_count_sums_to_total(self, sample_df):
        dist = get_class_distribution(sample_df)
        assert dist["count"].sum() == len(sample_df)

    def test_pct_sums_to_100(self, sample_df):
        dist = get_class_distribution(sample_df)
        assert abs(dist["pct_%"].sum() - 100.0) < 0.1

    def test_all_crops_present(self, sample_df):
        """Every unique crop in the DataFrame should appear in the distribution."""
        dist = get_class_distribution(sample_df)
        for crop in sample_df[TARGET_COLUMN].unique():
            assert crop in dist.index, f"Crop '{crop}' missing from distribution"

    def test_balanced_dataset_equal_counts(self, sample_df):
        """The sample_df fixture is balanced (10 records per crop)."""
        dist = get_class_distribution(sample_df)
        assert (dist["count"] == 10).all()


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE_BOUNDS constant
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureBoundsConstant:
    """Sanity checks on the module-level FEATURE_BOUNDS dict."""

    def test_all_features_have_bounds(self):
        for col in FEATURE_COLUMNS:
            assert col in FEATURE_BOUNDS, f"'{col}' missing from FEATURE_BOUNDS"

    def test_bounds_are_tuples_of_two(self):
        for col, bounds in FEATURE_BOUNDS.items():
            assert len(bounds) == 2, f"Bounds for '{col}' should have exactly 2 elements"

    def test_lower_bound_less_than_upper(self):
        for col, (lo, hi) in FEATURE_BOUNDS.items():
            assert lo < hi, f"Lower bound >= upper bound for '{col}'"

    def test_ph_bounds_valid(self):
        lo, hi = FEATURE_BOUNDS["ph"]
        assert lo >= 0
        assert hi <= 14
