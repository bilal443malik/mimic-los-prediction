"""Lightweight utility helpers for LOS analytics modules."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)


def ensure_directory(path: str | Path) -> Path:
    """Create directory if missing and return a `Path` object."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    df_name: str,
) -> None:
    """Raise a clear error if expected columns are missing."""

    required = list(required_columns)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    """Return numerator/denominator with zero-division guard."""

    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Save dataframe to CSV or parquet based on file extension."""

    output_path = Path(path)
    ensure_directory(output_path.parent)

    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported dataframe output suffix: {output_path.suffix}")

    return output_path


def save_joblib_artifact(obj: Any, path: str | Path) -> Path:
    """Serialize python object using joblib."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    joblib.dump(obj, output_path)
    return output_path


def load_joblib_artifact(path: str | Path) -> Any:
    """Load joblib artifact from disk."""

    return joblib.load(Path(path))
