"""Model evaluation, threshold tuning, and explainability helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def threshold_table(y_true: pd.Series, y_score: pd.Series) -> pd.DataFrame:
    """Build precision/recall table across decision thresholds."""

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    ).sort_values("threshold").reset_index(drop=True)


def evaluate_classifier(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series,
    model_name: str,
) -> dict[str, Any]:
    """Return a complete notebook-friendly evaluation bundle."""

    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_score))
    except ValueError:
        pr_auc = float("nan")

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": float(brier_score_loss(y_true, y_score)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )

    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    return {
        "metrics": metrics,
        "confusion_matrix": cm_df,
        "classification_report": report_df,
        "threshold_table": threshold_table(y_true, y_score),
    }


def compare_model_metrics(evaluations: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a comparison table across model evaluation bundles."""

    rows = [result["metrics"] for result in evaluations.values()]
    return pd.DataFrame(rows).sort_values(["pr_auc", "roc_auc"], ascending=False).reset_index(drop=True)


def build_threshold_tuning_table(
    y_true: pd.Series,
    y_score: pd.Series,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute threshold tradeoff metrics for operational policy selection."""

    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)

    rows: list[dict[str, float]] = []
    y_true_np = np.asarray(y_true).astype(int)

    for t in thresholds:
        y_pred = (np.asarray(y_score) >= float(t)).astype(int)

        tp = int(((y_true_np == 1) & (y_pred == 1)).sum())
        tn = int(((y_true_np == 0) & (y_pred == 0)).sum())
        fp = int(((y_true_np == 0) & (y_pred == 1)).sum())
        fn = int(((y_true_np == 1) & (y_pred == 0)).sum())

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        youden_j = recall + specificity - 1.0

        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "youden_j": youden_j,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_operating_thresholds(
    threshold_df: pd.DataFrame,
    recall_min: float = 0.80,
) -> dict[str, float]:
    """Select balanced and recall-priority operating thresholds."""

    # Balanced policy: maximize F1 (with Youden J tie-break) for overall utility.
    balanced_row = threshold_df.sort_values(["f1", "youden_j"], ascending=False).iloc[0]

    # Recall-priority policy: keep high sensitivity and choose the best precision.
    recall_candidates = threshold_df[threshold_df["recall"] >= recall_min]
    if recall_candidates.empty:
        recall_row = threshold_df.sort_values(["recall", "precision"], ascending=False).iloc[0]
    else:
        recall_row = recall_candidates.sort_values(["precision", "f1"], ascending=False).iloc[0]
    # On some cohorts these can legitimately select the same threshold (e.g., 0.40).

    return {
        "balanced_threshold": float(balanced_row["threshold"]),
        "balanced_f1": float(balanced_row["f1"]),
        "balanced_recall": float(balanced_row["recall"]),
        "balanced_precision": float(balanced_row["precision"]),
        "recall_priority_threshold": float(recall_row["threshold"]),
        "recall_priority_f1": float(recall_row["f1"]),
        "recall_priority_recall": float(recall_row["recall"]),
        "recall_priority_precision": float(recall_row["precision"]),
    }


def build_calibration_table(
    y_true: pd.Series,
    y_score: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build decile reliability table: predicted vs observed event rate."""

    cal = pd.DataFrame({"y_true": y_true.astype(int), "y_score": y_score.astype(float)})
    cal = cal.sort_values("y_score").reset_index(drop=True)

    cal["bin"] = pd.qcut(cal["y_score"], q=n_bins, labels=False, duplicates="drop")

    reliability = (
        cal.groupby("bin", as_index=False)
        .agg(
            n=("y_true", "size"),
            predicted_risk_mean=("y_score", "mean"),
            observed_event_rate=("y_true", "mean"),
        )
        .sort_values("bin")
        .reset_index(drop=True)
    )

    reliability["abs_calibration_gap"] = (
        reliability["predicted_risk_mean"] - reliability["observed_event_rate"]
    ).abs()

    return reliability


def _get_feature_names_from_pipeline(model: Pipeline) -> np.ndarray:
    """Extract transformed feature names from sklearn pipeline preprocessor."""

    preprocessor = model.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def extract_logistic_feature_importance(model: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """Return coefficient-based importance for logistic regression pipeline."""

    estimator = model.named_steps["model"]
    feature_names = _get_feature_names_from_pipeline(model)

    coefs = estimator.coef_.ravel()
    out = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }
    )

    return out.sort_values("abs_coefficient", ascending=False).head(top_n).reset_index(drop=True)


def extract_tree_feature_importance(model: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """Return tree feature importances from fitted pipeline."""

    estimator = model.named_steps["model"]
    feature_names = _get_feature_names_from_pipeline(model)
    importances = estimator.feature_importances_

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    return out.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def extract_permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 20,
    random_state: int = 42,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return permutation importance over transformed model pipeline."""

    # Use raw feature columns for permutation output to keep interpretation simple.
    feature_names = np.array(list(X_test.columns), dtype=object).reshape(-1)

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_mean = np.asarray(result.importances_mean).reshape(-1)
    importance_std = np.asarray(result.importances_std).reshape(-1)

    n = min(len(feature_names), len(importance_mean), len(importance_std))
    rows = list(zip(feature_names[:n], importance_mean[:n], importance_std[:n]))
    out = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_std"])
    return out.sort_values("importance_mean", ascending=False).head(top_n).reset_index(drop=True)
