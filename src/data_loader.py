"""Data loading helpers for MIMIC-IV LOS analysis."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.config import (
    ADMISSIONS_TABLE_BASENAME,
    PATIENTS_TABLE_BASENAME,
    REQUIRED_ADMISSIONS_COLUMNS,
    REQUIRED_PATIENTS_COLUMNS,
    SUPPORTED_TABLE_EXTENSIONS,
)
from src.utils import validate_required_columns


def resolve_table_path(source: str | Path, table_basename: str) -> Path:
    """Resolve a concrete table path from either a file path or a directory.

    If `source` is a directory, this searches for `<table_basename>` with a supported
    extension in this order: `.csv.gz`, `.csv`, `.parquet`.
    """

    source_path = Path(source)
    env_override = os.getenv("MIMIC_HOSP_DIR")

    # If the provided source path is missing, allow environment override.
    if not source_path.exists() and env_override:
        env_path = Path(env_override)
        if env_path.exists():
            source_path = env_path

    if source_path.is_file():
        return source_path

    if source_path.is_dir():
        for ext in SUPPORTED_TABLE_EXTENSIONS:
            candidate = source_path / f"{table_basename}{ext}"
            if candidate.exists():
                return candidate

        expected_files = [f"{table_basename}{ext}" for ext in SUPPORTED_TABLE_EXTENSIONS]
        raise FileNotFoundError(
            f"Could not find `{table_basename}` table in directory: {source_path}\n"
            f"Expected one of: {expected_files}\n"
            "You can either:\n"
            "1) Place files under `<project>/data/raw/mimiciv/hosp/`, or\n"
            "2) Pass an explicit directory/file path to load_admissions/load_patients, or\n"
            "3) Set environment variable `MIMIC_HOSP_DIR` to your local MIMIC hosp folder."
        )

    raise FileNotFoundError(
        f"Path does not exist: {source_path}\n"
        "Expected MIMIC hosp files under `<project>/data/raw/mimiciv/hosp/` by default.\n"
        "If your files are elsewhere, pass that path directly or set `MIMIC_HOSP_DIR`."
    )


def _read_table(table_path: Path) -> pd.DataFrame:
    """Read CSV/CSV.GZ/Parquet based on file suffix."""

    suffixes = table_path.suffixes
    if suffixes[-2:] == [".csv", ".gz"] or table_path.suffix == ".csv":
        return pd.read_csv(table_path, low_memory=False)
    if table_path.suffix == ".parquet":
        return pd.read_parquet(table_path)

    raise ValueError(
        f"Unsupported file extension for {table_path}. "
        "Supported: .csv, .csv.gz, .parquet"
    )


def _parse_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Parse datetime columns when present, coercing invalid values to NaT."""

    parsed = df.copy()
    for col in columns:
        if col in parsed.columns:
            parsed[col] = pd.to_datetime(parsed[col], errors="coerce")
    return parsed


def load_admissions(source: str | Path) -> pd.DataFrame:
    """Load admissions from a file or directory and validate schema."""

    admissions_path = resolve_table_path(source, ADMISSIONS_TABLE_BASENAME)
    admissions = _read_table(admissions_path)
    admissions = _parse_datetime_columns(
        admissions,
        ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
    )
    validate_required_columns(admissions, REQUIRED_ADMISSIONS_COLUMNS, "admissions")
    return admissions


def load_patients(source: str | Path) -> pd.DataFrame:
    """Load patients from a file or directory and validate schema."""

    patients_path = resolve_table_path(source, PATIENTS_TABLE_BASENAME)
    patients = _read_table(patients_path)
    validate_required_columns(patients, REQUIRED_PATIENTS_COLUMNS, "patients")
    return patients


def load_core_admissions_dataset(
    admissions_source: str | Path,
    patients_source: str | Path,
) -> pd.DataFrame:
    """Load admissions + patients and merge demographic anchor columns.

    This returns a base joined table; cohort filtering is handled in `src/cohort.py`.
    """

    admissions = load_admissions(admissions_source)
    patients = load_patients(patients_source)

    demographics = patients[["subject_id", "gender", "anchor_age", "anchor_year"]].drop_duplicates(
        subset=["subject_id"],
        keep="first",
    )

    merged = admissions.merge(
        demographics,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )
    return merged
