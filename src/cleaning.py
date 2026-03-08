"""Data quality checks and cleaning helpers for LOS cohort phase."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.utils import safe_rate


def missing_value_report(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Return missing-value counts and rates by column."""

    cols = list(columns) if columns is not None else list(df.columns)
    report = pd.DataFrame(
        {
            "column": cols,
            "missing_count": [int(df[col].isna().sum()) for col in cols],
            "row_count": [int(df.shape[0]) for _ in cols],
        }
    )
    report["missing_rate"] = report.apply(
        lambda row: safe_rate(row["missing_count"], row["row_count"]),
        axis=1,
    )
    return report.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)


def duplicate_admission_report(df: pd.DataFrame, id_col: str = "hadm_id") -> pd.DataFrame:
    """Summarize duplicate admission IDs for one-row-per-admission integrity."""

    total_rows = int(df.shape[0])
    unique_ids = int(df[id_col].nunique(dropna=True))
    duplicate_rows = total_rows - unique_ids

    return pd.DataFrame(
        [
            {
                "id_column": id_col,
                "total_rows": total_rows,
                "unique_ids": unique_ids,
                "duplicate_rows": duplicate_rows,
                "duplicate_rate": safe_rate(duplicate_rows, total_rows),
            }
        ]
    )


def sanity_summary(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    los_col: str = "los_days",
) -> pd.DataFrame:
    """Return high-level sanity metrics after cohort filtering."""

    raw_rows = int(raw_df.shape[0])
    clean_rows = int(clean_df.shape[0])
    dropped_rows = raw_rows - clean_rows

    out = {
        "raw_rows": raw_rows,
        "clean_rows": clean_rows,
        "dropped_rows": dropped_rows,
        "drop_rate": safe_rate(dropped_rows, raw_rows),
        "clean_unique_hadm_id": int(clean_df["hadm_id"].nunique(dropna=True)) if "hadm_id" in clean_df else pd.NA,
        "clean_unique_subject_id": int(clean_df["subject_id"].nunique(dropna=True)) if "subject_id" in clean_df else pd.NA,
        "los_min": float(clean_df[los_col].min()) if los_col in clean_df else pd.NA,
        "los_median": float(clean_df[los_col].median()) if los_col in clean_df else pd.NA,
        "los_max": float(clean_df[los_col].max()) if los_col in clean_df else pd.NA,
    }
    return pd.DataFrame([out])


def normalize_categorical_labels(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Fill missing categorical values and trim whitespace."""

    cleaned = df.copy()
    target_columns = (
        list(columns)
        if columns is not None
        else ["gender", "race", "insurance", "marital_status", "admission_type", "language"]
    )

    for col in target_columns:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna("Unknown").astype(str).str.strip()
            cleaned[col] = cleaned[col].replace({"": "Unknown"})

    return cleaned
