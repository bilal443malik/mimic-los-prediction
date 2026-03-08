"""Project-level configuration for MIMIC-IV LOS analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Long-stay candidate defaults used consistently across SQL, Python, and notebook.
LONG_STAY_DAYS_THRESHOLD: float = 7.0
LONG_STAY_PERCENTILE: float = 0.90
DEFAULT_LABEL_MODE: str = "gt_7d"
DEFAULT_MODEL_RANDOM_STATE: int = 42

# Phase-4 final-selection defaults.
FINAL_LABEL_MODE: str = "gt_7d"
FINAL_MODEL_NAME: str = "random_forest"
RECALL_PRIORITY_MIN_RECALL: float = 0.80
# Submission-readiness defaults for documentation/dashboard fallback.
# These do not replace data-driven threshold tuning in the notebook;
# they are used only when artifacts are missing or incomplete.
FINAL_BALANCED_THRESHOLD: float = 0.40
FINAL_RECALL_PRIORITY_THRESHOLD: float = 0.40

# Expected table names under `data/raw/mimiciv/hosp/`.
ADMISSIONS_TABLE_BASENAME = "admissions"
PATIENTS_TABLE_BASENAME = "patients"
SUPPORTED_TABLE_EXTENSIONS: tuple[str, ...] = (".csv.gz", ".csv", ".parquet")

# Minimal schema required for this phase.
REQUIRED_ADMISSIONS_COLUMNS: tuple[str, ...] = (
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
)

REQUIRED_PATIENTS_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "gender",
    "anchor_age",
    "anchor_year",
)


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized filesystem paths used by analysis modules."""

    root: Path

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def mimic_hosp_dir(self) -> Path:
        return self.data_raw / "mimiciv" / "hosp"

    @property
    def outputs_figures(self) -> Path:
        return self.root / "outputs" / "figures"

    @property
    def outputs_tables(self) -> Path:
        return self.root / "outputs" / "tables"

    @property
    def outputs_models(self) -> Path:
        return self.root / "outputs" / "models"


def get_project_paths(root: Path | None = None) -> ProjectPaths:
    """Return canonical project paths.

    If `root` is not provided, assume this file is at `<root>/src/config.py`.
    """

    resolved_root = root if root is not None else Path(__file__).resolve().parents[1]
    return ProjectPaths(root=resolved_root)
