"""EDA helpers focused on LOS distribution and long-stay definitions."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.config import LONG_STAY_DAYS_THRESHOLD, LONG_STAY_PERCENTILE
from src.utils import safe_rate


def los_summary_table(df: pd.DataFrame, los_col: str = "los_days") -> pd.DataFrame:
    """Return core LOS descriptive statistics."""

    los = df[los_col].dropna()
    summary = {
        "n_admissions": int(los.shape[0]),
        "mean_los_days": float(los.mean()),
        "median_los_days": float(los.median()),
        "std_los_days": float(los.std(ddof=1)),
        "min_los_days": float(los.min()),
        "p25_los_days": float(los.quantile(0.25)),
        "p75_los_days": float(los.quantile(0.75)),
        "p90_los_days": float(los.quantile(0.90)),
        "p95_los_days": float(los.quantile(0.95)),
        "p99_los_days": float(los.quantile(0.99)),
        "max_los_days": float(los.max()),
    }
    return pd.DataFrame([summary])


def los_percentile_table(
    df: pd.DataFrame,
    los_col: str = "los_days",
    percentiles: Iterable[float] = (0.50, 0.75, 0.90, 0.95, 0.99),
) -> pd.DataFrame:
    """Return LOS values at requested percentiles."""

    los = df[los_col].dropna()
    rows = []
    for p in percentiles:
        rows.append({"percentile": p, "los_days": float(los.quantile(p))})
    return pd.DataFrame(rows)


def add_long_stay_labels(
    df: pd.DataFrame,
    los_col: str = "los_days",
    fixed_day_threshold: float = LONG_STAY_DAYS_THRESHOLD,
    percentile_threshold: float = LONG_STAY_PERCENTILE,
) -> tuple[pd.DataFrame, float]:
    """Add long-stay labels for both candidate definitions.

    Returns:
    - labeled dataframe
    - numeric LOS threshold corresponding to the percentile definition
    """

    labeled = df.copy()
    pctl_value = float(labeled[los_col].quantile(percentile_threshold))
    labeled["long_stay_gt_7d"] = (labeled[los_col] > fixed_day_threshold).astype(int)
    labeled["long_stay_gte_p90"] = (labeled[los_col] >= pctl_value).astype(int)
    return labeled, pctl_value


def long_stay_definition_comparison(
    df: pd.DataFrame,
    los_col: str = "los_days",
    fixed_day_threshold: float = LONG_STAY_DAYS_THRESHOLD,
    percentile_threshold: float = LONG_STAY_PERCENTILE,
) -> pd.DataFrame:
    """Compare class balance between fixed and percentile long-stay definitions."""

    labeled, pctl_value = add_long_stay_labels(
        df,
        los_col=los_col,
        fixed_day_threshold=fixed_day_threshold,
        percentile_threshold=percentile_threshold,
    )

    total = int(labeled.shape[0])

    gt_7_count = int(labeled["long_stay_gt_7d"].sum())
    gte_p90_count = int(labeled["long_stay_gte_p90"].sum())

    comparison = pd.DataFrame(
        [
            {
                "definition": f"LOS > {fixed_day_threshold:.0f} days",
                "threshold_value_days": fixed_day_threshold,
                "positive_count": gt_7_count,
                "positive_rate": safe_rate(gt_7_count, total),
            },
            {
                "definition": f"LOS >= p{int(percentile_threshold * 100)}",
                "threshold_value_days": pctl_value,
                "positive_count": gte_p90_count,
                "positive_rate": safe_rate(gte_p90_count, total),
            },
        ]
    )
    return comparison


def subgroup_long_stay_rates(
    df: pd.DataFrame,
    group_columns: Iterable[str],
    target_columns: Iterable[str] = ("long_stay_gt_7d", "long_stay_gte_p90"),
) -> pd.DataFrame:
    """Compute long-stay rates by subgroup and label definition."""

    rows: list[dict[str, object]] = []

    for target_col in target_columns:
        if target_col not in df.columns:
            continue

        for group_col in group_columns:
            if group_col not in df.columns:
                continue

            grouped = (
                df.groupby(group_col, dropna=False)[target_col]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"count": "n_admissions", "mean": "long_stay_rate"})
            )

            for _, row in grouped.iterrows():
                rows.append(
                    {
                        "definition": target_col,
                        "group_column": group_col,
                        "group_value": row[group_col],
                        "n_admissions": int(row["n_admissions"]),
                        "long_stay_rate": float(row["long_stay_rate"]),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["definition", "group_column", "long_stay_rate"],
        ascending=[True, True, False],
    )


def recommend_long_stay_definition(comparison_df: pd.DataFrame) -> str:
    """Return a simple recommendation note for notebook interpretation.

    Placeholder logic for this phase:
    - Prefer fixed threshold when prevalence is still operationally interpretable
      (roughly 5% to 50%) because it is easier to explain to non-technical users.
    - Otherwise prefer percentile-based definition for distribution-relative targeting.
    """

    fixed_row = comparison_df[comparison_df["definition"].str.contains("LOS >", regex=False)]
    if fixed_row.empty:
        return "Insufficient comparison data to recommend a definition."

    fixed_rate = float(fixed_row.iloc[0]["positive_rate"])

    if 0.05 <= fixed_rate <= 0.50:
        return (
            "Recommend primary label: LOS > 7 days, because prevalence is operationally "
            "manageable and interpretation is straightforward for non-technical stakeholders."
        )

    return (
        "Recommend primary label: LOS >= p90, because fixed-threshold prevalence appears "
        "too extreme for balanced monitoring/modeling in this cohort."
    )
