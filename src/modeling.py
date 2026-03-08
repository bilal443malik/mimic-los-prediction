"""Training and decision-support utilities for LOS long-stay modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    FINAL_BALANCED_THRESHOLD,
    FINAL_LABEL_MODE,
    FINAL_MODEL_NAME,
    LONG_STAY_DAYS_THRESHOLD,
    LONG_STAY_PERCENTILE,
)


@dataclass
class ModelBundle:
    """Named model pipeline and status metadata."""

    name: str
    pipeline: Pipeline
    status: str = "trained"


def make_target_label(
    df: pd.DataFrame,
    label_mode: str = FINAL_LABEL_MODE,
    los_col: str = "los_days",
    fixed_day_threshold: float = LONG_STAY_DAYS_THRESHOLD,
    percentile_threshold: float = LONG_STAY_PERCENTILE,
    output_col: str = "target_long_stay",
) -> tuple[pd.DataFrame, float]:
    """Create modeling target label with selectable definition.

    label_mode:
    - "gt_7d": LOS > fixed threshold (default 7 days)
    - "gte_p90": LOS >= cohort percentile threshold (default p90)

    Submission default is "gt_7d" because operations teams can act on a fixed
    day-based definition more easily than percentile-relative labels.
    """

    out = df.copy()
    pctl_value = float(out[los_col].quantile(percentile_threshold))

    if label_mode == "gt_7d":
        out[output_col] = (out[los_col] > fixed_day_threshold).astype(int)
        threshold_value = fixed_day_threshold
    elif label_mode == "gte_p90":
        out[output_col] = (out[los_col] >= pctl_value).astype(int)
        threshold_value = pctl_value
    else:
        raise ValueError("label_mode must be one of: {'gt_7d', 'gte_p90'}")

    return out, threshold_value


def split_features_target(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "target_long_stay",
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target with stratification."""

    X = df.loc[:, feature_columns].copy()
    y = df[target_col].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for mixed tabular features."""

    # Mixed tabular handling keeps the workflow explainable and reproducible.
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Pipeline:
    """Train class-balanced logistic regression baseline."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    return pipeline.fit(X_train, y_train)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Pipeline:
    """Train class-balanced random forest baseline."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return pipeline.fit(X_train, y_train)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Pipeline:
    """Train optional XGBoost classifier.

    Raises ImportError if xgboost is unavailable.
    """

    from xgboost import XGBClassifier  # type: ignore

    positive_rate = float(np.mean(y_train)) if len(y_train) > 0 else 0.5
    neg_to_pos_ratio = (1 - positive_rate) / max(positive_rate, 1e-6)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    scale_pos_weight=neg_to_pos_ratio,
                    n_jobs=4,
                ),
            ),
        ]
    )

    return pipeline.fit(X_train, y_train)


def predict_classes_and_scores(model: Pipeline, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return class predictions and positive-class probabilities."""

    y_pred = pd.Series(model.predict(X), index=X.index, dtype="int64")

    if hasattr(model, "predict_proba"):
        y_score = pd.Series(model.predict_proba(X)[:, 1], index=X.index, dtype="float64")
    else:
        decision = model.decision_function(X)
        y_score = pd.Series(decision, index=X.index, dtype="float64")

    return y_pred, y_score


def run_baseline_model_suite(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    include_xgboost: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train baseline models and return prediction-ready artifacts."""

    results: dict[str, Any] = {
        "models": {},
        "predictions": {},
        "xgboost_status": "not_requested",
    }

    logistic = train_logistic_regression(X_train, y_train, random_state=random_state)
    rf = train_random_forest(X_train, y_train, random_state=random_state)

    for name, model in [("logistic_regression", logistic), ("random_forest", rf)]:
        y_pred, y_score = predict_classes_and_scores(model, X_test)
        results["models"][name] = model
        results["predictions"][name] = {
            "y_pred": y_pred,
            "y_score": y_score,
        }

    if include_xgboost:
        try:
            xgb_model = train_xgboost(X_train, y_train, random_state=random_state)
            y_pred, y_score = predict_classes_and_scores(xgb_model, X_test)
            results["models"]["xgboost"] = xgb_model
            results["predictions"]["xgboost"] = {
                "y_pred": y_pred,
                "y_score": y_score,
            }
            results["xgboost_status"] = "trained"
        except ImportError:
            results["xgboost_status"] = "unavailable"

    return results


def build_label_prevalence_table(
    df: pd.DataFrame,
    los_col: str = "los_days",
    fixed_day_threshold: float = LONG_STAY_DAYS_THRESHOLD,
    percentile_threshold: float = LONG_STAY_PERCENTILE,
) -> pd.DataFrame:
    """Build prevalence comparison for candidate long-stay labels."""

    total = int(df.shape[0])
    p90 = float(df[los_col].quantile(percentile_threshold))

    gt7_count = int((df[los_col] > fixed_day_threshold).sum())
    p90_count = int((df[los_col] >= p90).sum())

    return pd.DataFrame(
        [
            {
                "label_mode": "gt_7d",
                "definition": f"LOS > {fixed_day_threshold:.0f} days",
                "threshold_value": fixed_day_threshold,
                "positive_count": gt7_count,
                "prevalence": gt7_count / total if total else np.nan,
                "business_clarity": "High",
                "interpretability_note": "Simple fixed threshold for operations",
            },
            {
                "label_mode": "gte_p90",
                "definition": f"LOS >= p{int(percentile_threshold * 100)}",
                "threshold_value": p90,
                "positive_count": p90_count,
                "prevalence": p90_count / total if total else np.nan,
                "business_clarity": "Medium",
                "interpretability_note": "Distribution-relative threshold",
            },
        ]
    )


def build_final_recommendation_card(
    model_metrics_df: pd.DataFrame,
    label_prevalence_df: pd.DataFrame,
    final_label_mode: str = FINAL_LABEL_MODE,
    final_model_name: str = FINAL_MODEL_NAME,
    final_operating_threshold: float = FINAL_BALANCED_THRESHOLD,
) -> pd.DataFrame:
    """Build compact final recommendation card for notebook/dashboard packaging.

    Includes the submission-ready operating threshold for cross-surface consistency.
    """

    chosen_label = label_prevalence_df[label_prevalence_df["label_mode"] == final_label_mode].iloc[0]

    if final_model_name not in model_metrics_df["model"].values:
        fallback = model_metrics_df.sort_values("pr_auc", ascending=False).iloc[0]
        final_model_name = str(fallback["model"])

    # Final submission policy keeps Random Forest as primary model and Logistic as
    # interpretability reference; fallback only triggers if selected model is absent.
    chosen_model = model_metrics_df[model_metrics_df["model"] == final_model_name].iloc[0]

    return pd.DataFrame(
        [
            {
                "final_label_mode": final_label_mode,
                "final_label_definition": chosen_label["definition"],
                "final_label_prevalence": float(chosen_label["prevalence"]),
                "final_model_name": final_model_name,
                "final_operating_threshold": float(final_operating_threshold),
                "final_model_accuracy": float(chosen_model["accuracy"]),
                "final_model_recall": float(chosen_model["recall"]),
                "final_model_precision": float(chosen_model["precision"]),
                "final_model_f1": float(chosen_model["f1"]),
                "final_model_roc_auc": float(chosen_model["roc_auc"]),
                "final_model_pr_auc": float(chosen_model["pr_auc"]),
                "interpretability_tradeoff": "Random Forest improves predictive discrimination; Logistic retained for coefficient-level interpretation.",
                "business_recommendation": (
                    "Use gt_7d label and Random Forest for admission-time risk flagging; "
                    f"start with balanced threshold {float(final_operating_threshold):.2f}."
                ),
            }
        ]
    )
