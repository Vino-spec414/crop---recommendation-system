"""
eda.py
======
Standalone Exploratory Data Analysis script.

Run this before training to understand your data.
All plots are saved to /models/ so they can be referenced
in the dissertation alongside the ML results.

Run:
    python src/eda.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preprocessing import (
    load_crop_dataset, load_soil_dataset,
    validate_schema, clean_dataset,
    get_summary_stats, get_class_distribution,
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
MODELS_DIR.mkdir(exist_ok=True)

# ── Consistent plot style ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({
    "figure.dpi":    130,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PALETTE = sns.color_palette("viridis", 7)
BG      = "#F7F3EC"


def plot_class_distribution(df: pd.DataFrame) -> None:
    dist = df[TARGET_COLUMN].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=BG)

    colours = sns.color_palette("viridis", len(dist))
    dist.sort_values().plot(kind="barh", ax=axes[0], color=colours, edgecolor="none")
    axes[0].set_title("Crop Class Counts", fontweight="bold", pad=10)
    axes[0].set_xlabel("Records")
    axes[0].set_facecolor(BG)
    for bar, val in zip(axes[0].patches, dist.sort_values().values):
        axes[0].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     str(val), va="center", fontsize=8)

    wedge = dict(width=0.45, edgecolor="white", linewidth=1.5)
    axes[1].pie(dist.values, labels=dist.index, autopct="%1.0f%%",
                startangle=140, colors=colours, pctdistance=0.78,
                textprops={"fontsize": 7.5}, wedgeprops=wedge)
    axes[1].set_title("Class Balance (%)", fontweight="bold", pad=10)
    axes[1].set_facecolor(BG)

    plt.suptitle("Target Variable — Crop Class Distribution",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "eda_class_distribution.png")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLUMNS):
        sns.histplot(df[col], kde=True, ax=axes[i], color=PALETTE[i], alpha=0.75, linewidth=0)
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("")
        skew = df[col].skew()
        axes[i].text(0.97, 0.95, f"skew={skew:.2f}",
                     transform=axes[i].transAxes,
                     ha="right", va="top", fontsize=8, color="#666")
    axes[-1].axis("off")
    plt.suptitle("Feature Distributions with KDE", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "eda_feature_distributions.png")


def plot_correlation(df: pd.DataFrame) -> None:
    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5,
        square=True, ax=ax, cbar_kws={"shrink": .7},
        annot_kws={"fontsize": 9},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    _save(fig, "eda_correlation.png")


def plot_box_by_crop(df: pd.DataFrame) -> None:
    """Box plots for the 3 most discriminating features."""
    key_feats = ["N", "temperature", "rainfall"]
    fig, axes = plt.subplots(len(key_feats), 1, figsize=(16, 12))
    for ax, feat in zip(axes, key_feats):
        order = df.groupby(TARGET_COLUMN)[feat].median().sort_values().index
        sns.boxplot(data=df, x=TARGET_COLUMN, y=feat,
                    order=order, ax=ax, palette="viridis",
                    linewidth=0.7, fliersize=2)
        ax.set_title(f"{feat} by Crop Class", fontweight="bold", pad=8)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40)
    plt.suptitle("Key Features per Crop — sorted by median",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "eda_boxplots.png")


def plot_violin(df: pd.DataFrame) -> None:
    """pH and Humidity violins — useful for dissertation discussion."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, feat in zip(axes, ["ph", "humidity"]):
        order = df.groupby(TARGET_COLUMN)[feat].median().sort_values().index
        sns.violinplot(
            data=df, x=TARGET_COLUMN, y=feat,
            order=order, ax=ax, palette="viridis",
            inner="quartile", linewidth=0.6, scale="width",
        )
        ax.set_title(f"{feat} distribution by Crop", fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    plt.suptitle("pH & Humidity Violin Plots", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "eda_violin.png")


def plot_pairplot(df: pd.DataFrame) -> None:
    """Pairplot on a readable subset of crops."""
    subset_crops = ["rice", "maize", "wheat", "mango", "cotton", "coffee"]
    sub = df[df[TARGET_COLUMN].isin(subset_crops)]
    pair_cols = ["N", "P", "K", "temperature", "rainfall", TARGET_COLUMN]
    pp = sns.pairplot(
        sub[pair_cols], hue=TARGET_COLUMN,
        palette="tab10", diag_kind="kde",
        plot_kws={"alpha": 0.45, "s": 12}, diag_kws={"linewidth": 1},
    )
    pp.fig.suptitle("Pairplot — Selected Crops & Key Features",
                    y=1.02, fontsize=12, fontweight="bold")
    _save(pp.fig, "eda_pairplot.png")


def run_soil_eda(soil_df: pd.DataFrame) -> None:
    """EDA for the secondary soil dataset."""
    numeric_cols = soil_df.select_dtypes(include=[np.number]).columns.tolist()
    n_cols = min(8, len(numeric_cols))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    soil_pal = sns.color_palette("YlOrBr", n_cols)

    for i in range(n_cols):
        col = numeric_cols[i]
        data = soil_df[col].dropna()
        sns.histplot(data, kde=True, ax=axes[i], color=soil_pal[i], alpha=0.75, linewidth=0)
        axes[i].set_title(col, fontweight="bold", fontsize=9)
        axes[i].tick_params(labelsize=7)

    for j in range(n_cols, 8):
        axes[j].axis("off")

    plt.suptitle("Soil Property Distributions (Secondary Dataset)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "eda_soil_distributions.png")

    # Correlation heatmap for soil properties
    corr = soil_df[numeric_cols[:10]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="BrBG", center=0, linewidths=0.5, ax=ax,
                annot_kws={"fontsize": 8})
    ax.set_title("Soil Properties Correlation Matrix",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    _save(fig, "eda_soil_correlation.png")


def _save(fig, filename: str) -> None:
    path = MODELS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def run_eda(data_path=None, soil_path=None) -> None:
    print("\n" + "=" * 55)
    print("  📊  EXPLORATORY DATA ANALYSIS")
    print("=" * 55 + "\n")

    # Primary dataset
    raw = load_crop_dataset(data_path)
    validate_schema(raw)
    df = clean_dataset(raw)

    print("── Summary Statistics ──")
    print(get_summary_stats(df).to_string())

    print("\n── Class Distribution ──")
    print(get_class_distribution(df).to_string())

    logger.info("Generating EDA plots → %s", MODELS_DIR)
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_correlation(df)
    plot_box_by_crop(df)
    plot_violin(df)
    plot_pairplot(df)

    # Secondary dataset
    soil_df = load_soil_dataset(soil_path)
    if soil_df is not None:
        run_soil_eda(soil_df)

    print("\n✅  EDA complete. All charts saved to /models/")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_eda()
