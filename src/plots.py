"""Reusable plotting helpers for LOS notebook and dashboard."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_los_distribution(df: pd.DataFrame, los_col: str = "los_days") -> plt.Figure:
    """Histogram + KDE for LOS distribution."""

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df[los_col].dropna(), bins=50, kde=True, ax=ax)
    ax.set_title("Length of Stay Distribution")
    ax.set_xlabel("LOS (days)")
    ax.set_ylabel("Admissions")
    fig.tight_layout()
    return fig


def plot_long_stay_rate(df: pd.DataFrame, group_col: str, target_col: str = "is_long_stay") -> plt.Figure:
    """Bar chart for long-stay share by a categorical grouping."""

    grouped = (
        df.groupby(group_col, dropna=False)[target_col]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=grouped, x=group_col, y=target_col, ax=ax)
    ax.set_title(f"Long-Stay Share by {group_col}")
    ax.set_ylabel("Long-stay share")
    ax.set_xlabel(group_col)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    metrics: Sequence[str] = ("roc_auc", "pr_auc", "f1", "recall", "precision"),
) -> plt.Figure:
    """Plot grouped bars for model metric comparison."""

    # Keep metric set compact so interview walkthroughs remain readable.
    available_metrics = [m for m in metrics if m in metrics_df.columns]
    melted = metrics_df.melt(
        id_vars=["model"],
        value_vars=available_metrics,
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="metric", y="score", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm_df: pd.DataFrame, title: str = "Confusion Matrix") -> plt.Figure:
    """Plot confusion matrix heatmap from a matrix dataframe."""

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    value_col: str,
    top_n: int = 15,
    title: str = "Feature Importance",
) -> plt.Figure:
    """Plot horizontal bar chart for top feature importances."""

    subset = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=subset, x=value_col, y="feature", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


def plot_precision_recall_curves(curve_data: dict[str, pd.DataFrame]) -> plt.Figure:
    """Plot precision-recall curves from threshold tables per model."""

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model_name, table in curve_data.items():
        ax.plot(table["recall"], table["precision"], label=model_name)

    ax.set_title("Precision-Recall Tradeoff")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_threshold_tradeoff(threshold_df: pd.DataFrame) -> plt.Figure:
    """Plot threshold vs precision/recall/F1 tradeoff."""

    # Designed to explain operating-policy choice to non-technical stakeholders.
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="precision")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="recall")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], label="f1")
    ax.set_title("Threshold Tradeoff (Final Model)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_calibration_curve(calibration_df: pd.DataFrame) -> plt.Figure:
    """Plot predicted vs observed event rates by probability bins."""

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(
        calibration_df["predicted_risk_mean"],
        calibration_df["observed_event_rate"],
        marker="o",
        label="model",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.set_title("Calibration Check (Decile Bins)")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig
