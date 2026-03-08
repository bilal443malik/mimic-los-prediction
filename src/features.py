"""Feature engineering for long-stay LOS classification."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.config import LONG_STAY_DAYS_THRESHOLD, LONG_STAY_PERCENTILE


def get_excluded_columns_with_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Return excluded columns and rationale for admission-time prediction.

    Exclusions intentionally include identifiers and outcome-adjacent fields.
    """

    exclusion_map = {
        "subject_id": "Identifier column; can cause memorization and is not clinically meaningful.",
        "hadm_id": "Admission identifier; should never be used as predictive signal.",
        "admittime": "Raw timestamp; replaced by engineered time features.",
        "dischtime": "Not known at admission; direct outcome leakage.",
        "deathtime": "Future event relative to admission; leakage.",
        "los_days": "Target-defining variable; direct leakage.",
        "discharge_location": "Determined near/after discharge; leakage for admission-time prediction.",
        "hospital_expire_flag": "Outcome-adjacent endpoint; excluded from default model inputs.",
        "anchor_year": "De-identification anchor metadata; not clinically actionable.",
        "long_stay_gt_7d": "Target label, not predictor.",
        "long_stay_gte_p90": "Target label, not predictor.",
        "target_long_stay": "Target label, not predictor.",
    }

    rows: list[dict[str, str]] = []
    for col, reason in exclusion_map.items():
        if col in df.columns:
            rows.append({"column": col, "reason": reason})

    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def engineer_admission_time_features(df: pd.DataFrame, admit_col: str = "admittime") -> pd.DataFrame:
    """Create admission-time calendar features available at prediction time."""

    out = df.copy()
    out[admit_col] = pd.to_datetime(out[admit_col], errors="coerce")

    out["admit_hour"] = out[admit_col].dt.hour
    out["admit_dayofweek"] = out[admit_col].dt.dayofweek
    out["admit_month"] = out[admit_col].dt.month
    out["is_weekend_admit"] = out["admit_dayofweek"].isin([5, 6]).astype(int)
    return out


def engineer_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cleaned demographic features and coarse age bands."""

    out = df.copy()

    if "age_at_admit" in out.columns:
        out["age_at_admit"] = pd.to_numeric(out["age_at_admit"], errors="coerce")
    elif "anchor_age" in out.columns:
        out["age_at_admit"] = pd.to_numeric(out["anchor_age"], errors="coerce")
    else:
        out["age_at_admit"] = pd.NA

    bins = [0, 39, 49, 59, 69, 79, 120]
    labels = ["18-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    out["age_band"] = pd.cut(out["age_at_admit"], bins=bins, labels=labels, include_lowest=True)
    out["age_band"] = out["age_band"].astype("object").fillna("Unknown")

    return out


def get_safe_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return admission-time-safe columns for default modeling."""

    candidate_columns = [
        "age_at_admit",
        "age_band",
        "gender",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "admit_hour",
        "admit_dayofweek",
        "is_weekend_admit",
        "admit_month",
    ]

    # Explicitly enforce admission-time feature policy for interview clarity.
    leakage_like = set(get_excluded_columns_with_reasons(df)["column"].tolist())
    return [col for col in candidate_columns if col in df.columns and col not in leakage_like]


def build_modeling_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Build engineered dataset and return safe feature policy metadata.

    Returns:
    - engineered dataframe
    - safe feature columns for default admission-time model
    - excluded columns with reasons
    """

    featured = engineer_admission_time_features(df)
    featured = engineer_demographic_features(featured)

    safe_cols = get_safe_feature_columns(featured)
    excluded = get_excluded_columns_with_reasons(featured)

    return featured, safe_cols, excluded


def create_label_columns(
    df: pd.DataFrame,
    los_col: str = "los_days",
    fixed_day_threshold: float = LONG_STAY_DAYS_THRESHOLD,
    percentile_threshold: float = LONG_STAY_PERCENTILE,
) -> tuple[pd.DataFrame, float]:
    """Add both candidate labels for sensitivity analysis.

    Label A: LOS > fixed day threshold (default 7 days)
    Label B: LOS >= percentile threshold (default p90)
    """

    labeled = df.copy()
    # Keep both definitions to support sensitivity analysis and business discussion.
    pctl_value = float(labeled[los_col].quantile(percentile_threshold))

    labeled["long_stay_gt_7d"] = (labeled[los_col] > fixed_day_threshold).astype(int)
    labeled["long_stay_gte_p90"] = (labeled[los_col] >= pctl_value).astype(int)

    return labeled, pctl_value


def get_feature_overview(df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """Summarize candidate model columns for notebook display."""

    rows: list[dict[str, object]] = []
    for col in feature_columns:
        series = df[col]
        rows.append(
            {
                "feature": col,
                "dtype": str(series.dtype),
                "missing_rate": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
