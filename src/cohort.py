"""Cohort construction utilities for inpatient LOS analytics."""

from __future__ import annotations

import pandas as pd

from src.config import REQUIRED_ADMISSIONS_COLUMNS, REQUIRED_PATIENTS_COLUMNS
from src.utils import validate_required_columns


def merge_admissions_patients(admissions: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Merge admissions with key demographic anchor fields from patients."""

    validate_required_columns(admissions, REQUIRED_ADMISSIONS_COLUMNS, "admissions")
    validate_required_columns(patients, REQUIRED_PATIENTS_COLUMNS, "patients")

    patient_cols = ["subject_id", "gender", "anchor_age", "anchor_year"]
    patient_demo = patients[patient_cols].drop_duplicates(subset=["subject_id"], keep="first")

    merged = admissions.merge(
        patient_demo,
        how="left",
        on="subject_id",
        validate="many_to_one",
    )
    return merged


def calculate_los_days(
    df: pd.DataFrame,
    admit_col: str = "admittime",
    discharge_col: str = "dischtime",
    output_col: str = "los_days",
) -> pd.DataFrame:
    """Calculate LOS in days from admission/discharge datetimes."""

    out = df.copy()
    out[admit_col] = pd.to_datetime(out[admit_col], errors="coerce")
    out[discharge_col] = pd.to_datetime(out[discharge_col], errors="coerce")
    # Canonical LOS formula used across SQL and Python modules for consistency.
    out[output_col] = (out[discharge_col] - out[admit_col]).dt.total_seconds() / 86400.0
    return out


def derive_age_at_admit(
    df: pd.DataFrame,
    admit_col: str = "admittime",
    output_col: str = "age_at_admit",
) -> pd.DataFrame:
    """Derive approximate age at admission using MIMIC anchor metadata.

    MIMIC-IV uses de-identified shifted dates. This age is suitable for analytic
    segmentation but should not be interpreted as true calendar age.
    """

    out = df.copy()
    if {"anchor_age", "anchor_year", admit_col}.issubset(out.columns):
        admit_year = pd.to_datetime(out[admit_col], errors="coerce").dt.year
        # MIMIC-safe approximation: anchor_age advanced by shifted admit year delta.
        out[output_col] = out["anchor_age"] + (admit_year - out["anchor_year"])
    else:
        out[output_col] = pd.NA
    return out


def filter_invalid_admissions(
    df: pd.DataFrame,
    admit_col: str = "admittime",
    discharge_col: str = "dischtime",
    los_col: str = "los_days",
) -> pd.DataFrame:
    """Filter invalid admissions and enforce one row per `hadm_id`.

    Rules:
    - drop rows with missing `hadm_id`, `admittime`, or `dischtime`
    - drop rows with non-positive LOS
    - defensively deduplicate `hadm_id` by earliest available timestamps
    """

    out = df.copy()
    out = out.dropna(subset=["hadm_id", admit_col, discharge_col])

    # Defensive guard only; admissions should usually be unique by hadm_id in MIMIC.
    out = out.sort_values(["hadm_id", admit_col, discharge_col], kind="stable")
    out = out.drop_duplicates(subset=["hadm_id"], keep="first")

    # Remove invalid stays that would break LOS summaries and labels.
    out = out[out[los_col] > 0].copy()
    return out


def build_inpatient_cohort(admissions: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Build a clean admissions cohort with one row per hospital admission.

    This phase intentionally depends only on `admissions` and `patients` tables.
    Additional enrichments (diagnoses, ICU, procedures) can be joined later.
    """

    merged = merge_admissions_patients(admissions, patients)
    merged = calculate_los_days(merged)
    merged = derive_age_at_admit(merged)
    cohort = filter_invalid_admissions(merged)

    preferred_columns = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "hospital_expire_flag",
        "gender",
        "anchor_age",
        "anchor_year",
        "age_at_admit",
        "los_days",
    ]
    output_columns = [col for col in preferred_columns if col in cohort.columns]

    return cohort[output_columns].copy()
