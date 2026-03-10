"""Single-page Streamlit dashboard for LOS monitoring and admission-time risk scoring."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    FINAL_BALANCED_THRESHOLD,
    FINAL_LABEL_MODE,
    FINAL_MODEL_NAME,
    FINAL_RECALL_PRIORITY_THRESHOLD,
)
from src.features import engineer_demographic_features
from src.utils import load_joblib_artifact

st.set_page_config(page_title="Hospital LOS Dashboard", layout="wide")


def load_cohort_data() -> tuple[pd.DataFrame, Path]:
    """Load the project cohort file used as the dashboard's default real dataset."""

    cohort_path = PROJECT_ROOT / "data" / "interim" / "cohort_base.csv"
    if not cohort_path.exists():
        st.error(
            "Missing cohort file: `data/interim/cohort_base.csv`. "
            "Run the notebook through the final packaging steps to generate it."
        )
        st.stop()

    cohort_df = pd.read_csv(cohort_path)
    return cohort_df, cohort_path


def add_long_stay_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add both candidate long-stay definitions for dashboard charts."""

    out = df.copy()
    out["long_stay_gt_7d"] = (out["los_days"] > 7).astype(int)
    p90 = float(out["los_days"].quantile(0.90))
    out["long_stay_gte_p90"] = (out["los_days"] >= p90).astype(int)
    out["p90_threshold"] = p90
    return out


def simple_risk_score(age: int, admission_type: str, weekend: int, insurance: str) -> float:
    """Fallback heuristic score used when model artifacts are unavailable."""

    score = 0.08
    if age >= 75:
        score += 0.18
    elif age >= 65:
        score += 0.10

    if admission_type == "EMERGENCY":
        score += 0.22
    elif admission_type == "URGENT":
        score += 0.12

    if weekend == 1:
        score += 0.04

    if insurance in {"Medicaid", "Self Pay"}:
        score += 0.03

    return float(np.clip(score, 0.01, 0.95))


def load_artifacts() -> tuple[dict | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Load model bundle and supporting decision tables if available."""

    model_path = PROJECT_ROOT / "outputs" / "models" / "baseline_models.joblib"
    metrics_path = PROJECT_ROOT / "outputs" / "tables" / "model_metrics.csv"
    card_path = PROJECT_ROOT / "outputs" / "tables" / "final_model_card.csv"
    threshold_path = PROJECT_ROOT / "outputs" / "tables" / "threshold_tuning.csv"

    # Artifact-first flow: dashboard should use notebook-generated outputs when present.
    model_bundle = None
    if model_path.exists():
        try:
            model_bundle = load_joblib_artifact(model_path)
        except Exception:
            model_bundle = None

    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else None
    model_card_df = pd.read_csv(card_path) if card_path.exists() else None
    threshold_df = pd.read_csv(threshold_path) if threshold_path.exists() else None

    return model_bundle, metrics_df, model_card_df, threshold_df


def resolve_final_model_name(model_bundle: dict | None, model_card_df: pd.DataFrame | None) -> str:
    """Resolve final model name using model card first, then artifact fallback."""

    if model_card_df is not None and not model_card_df.empty and "final_model_name" in model_card_df.columns:
        return str(model_card_df.iloc[0]["final_model_name"])

    if model_bundle is not None and isinstance(model_bundle, dict):
        return str(model_bundle.get("primary_model_name", FINAL_MODEL_NAME))

    return FINAL_MODEL_NAME


def resolve_thresholds(model_bundle: dict | None, threshold_df: pd.DataFrame | None) -> tuple[float, float]:
    """Get balanced and recall-priority thresholds with fallback defaults.

    Fallback values are pinned to the final submission policy (0.40 / 0.40).
    """

    balanced = None
    recall_priority = None

    if threshold_df is not None and not threshold_df.empty:
        if "is_balanced" in threshold_df.columns:
            bal_rows = threshold_df[threshold_df["is_balanced"] == 1]
            if not bal_rows.empty:
                balanced = float(bal_rows.iloc[0]["threshold"])
        if "is_recall_priority" in threshold_df.columns:
            rec_rows = threshold_df[threshold_df["is_recall_priority"] == 1]
            if not rec_rows.empty:
                recall_priority = float(rec_rows.iloc[0]["threshold"])

    if model_bundle is not None and isinstance(model_bundle, dict):
        balanced = balanced if balanced is not None else float(model_bundle.get("balanced_threshold", FINAL_BALANCED_THRESHOLD))
        recall_priority = (
            recall_priority
            if recall_priority is not None
            else float(model_bundle.get("recall_priority_threshold", FINAL_RECALL_PRIORITY_THRESHOLD))
        )

    if balanced is None:
        balanced = FINAL_BALANCED_THRESHOLD
    if recall_priority is None:
        recall_priority = FINAL_RECALL_PRIORITY_THRESHOLD

    return balanced, recall_priority


def prepare_inference_row(
    age: int,
    gender: str,
    admission_type: str,
    admission_location: str,
    insurance: str,
    language: str,
    marital_status: str,
    race: str,
    admit_hour: int,
    admit_dayofweek: int,
    admit_month: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Construct single-row feature input aligned to trained feature columns."""

    row = pd.DataFrame(
        [
            {
                "age_at_admit": age,
                "gender": gender,
                "admission_type": admission_type,
                "admission_location": admission_location,
                "insurance": insurance,
                "language": language,
                "marital_status": marital_status,
                "race": race,
                "admit_hour": admit_hour,
                "admit_dayofweek": admit_dayofweek,
                "admit_month": admit_month,
                "is_weekend_admit": int(admit_dayofweek in [5, 6]),
            }
        ]
    )

    row = engineer_demographic_features(row)

    for col in feature_columns:
        if col not in row.columns:
            row[col] = np.nan

    return row[feature_columns]


def risk_tier(probability: float, balanced_threshold: float, recall_priority_threshold: float) -> str:
    """Map risk probability to plain-English tier."""

    # Tiering is anchored on tuned thresholds so interpretation stays policy-aligned.
    low_cut = min(recall_priority_threshold, balanced_threshold) * 0.6
    high_cut = max(recall_priority_threshold, balanced_threshold)

    if probability >= high_cut:
        return "High"
    if probability >= low_cut:
        return "Moderate"
    return "Low"


st.title("Hospital Length of Stay (LOS) Dashboard")
st.caption("Admission-time long-stay risk estimate for operations planning. Not a clinical decision tool.")
final_label_text = "LOS > 7 days" if FINAL_LABEL_MODE == "gt_7d" else "LOS >= p90"
st.caption(
    f"Final submission policy: label `{final_label_text}` ({FINAL_LABEL_MODE}), "
    f"model `{FINAL_MODEL_NAME}`, balanced threshold `{FINAL_BALANCED_THRESHOLD:.2f}`."
)
df, cohort_source_path = load_cohort_data()
st.success(f"Loaded cohort data from `{cohort_source_path.relative_to(PROJECT_ROOT)}`.")

if "los_days" not in df.columns:
    st.error("Dataset must include `los_days`.")
    st.stop()

df = add_long_stay_flags(df)
model_bundle, metrics_df, model_card_df, threshold_df = load_artifacts()

st.subheader("LOS Distribution")
st.caption("Distribution of observed length of stay in the selected dataset.")
fig_dist = px.histogram(df, x="los_days", nbins=50, title="Length of Stay (days)")
fig_dist.update_layout(bargap=0.05)
st.plotly_chart(fig_dist, use_container_width=True)

st.subheader("Long-Stay Share by Key Dimensions")
definition = st.radio(
    "Long-stay definition",
    options=["LOS > 7 days", "LOS >= p90"],
    horizontal=True,
)
target_col = "long_stay_gt_7d" if definition == "LOS > 7 days" else "long_stay_gte_p90"
st.caption("Modeling deployment policy uses `LOS > 7 days`; `LOS >= p90` is shown for sensitivity context.")

available_dimensions = [c for c in ["admission_type", "insurance", "is_weekend_admit", "race"] if c in df.columns]
dimension = st.selectbox("Dimension", options=available_dimensions)

share = (
    df.groupby(dimension, dropna=False)[target_col]
    .mean()
    .reset_index()
    .sort_values(target_col, ascending=False)
)
fig_share = px.bar(
    share,
    x=dimension,
    y=target_col,
    title=f"Long-stay share by {dimension}",
    labels={target_col: "Long-stay share"},
)
st.plotly_chart(fig_share, use_container_width=True)

st.subheader("Final Model Inference")
st.caption("Use admission-time inputs only. Output is probability + plain-English risk tier.")
if metrics_df is not None:
    st.caption("Latest model metrics")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

if model_card_df is not None and not model_card_df.empty:
    st.caption("Final model decision card")
    st.dataframe(model_card_df, use_container_width=True, hide_index=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    inp_age = st.slider("Age", min_value=18, max_value=95, value=65)
with col2:
    inp_gender = st.selectbox("Gender", options=["M", "F", "Unknown"])
with col3:
    inp_type = st.selectbox(
        "Admission type",
        options=sorted(df.get("admission_type", pd.Series(["EMERGENCY"])).dropna().astype(str).unique()),
    )
with col4:
    inp_location = st.selectbox(
        "Admission location",
        options=sorted(df.get("admission_location", pd.Series(["EMERGENCY ROOM"])).dropna().astype(str).unique()),
    )

col5, col6, col7, col8 = st.columns(4)
with col5:
    inp_ins = st.selectbox("Insurance", options=sorted(df.get("insurance", pd.Series(["Medicare"])).dropna().astype(str).unique()))
with col6:
    inp_lang = st.selectbox("Language", options=sorted(df.get("language", pd.Series(["ENGLISH"])).dropna().astype(str).unique()))
with col7:
    inp_marital = st.selectbox(
        "Marital status",
        options=sorted(df.get("marital_status", pd.Series(["Unknown"])).dropna().astype(str).unique()),
    )
with col8:
    inp_race = st.selectbox("Race", options=sorted(df.get("race", pd.Series(["Unknown"])).dropna().astype(str).unique()))

col9, col10, col11 = st.columns(3)
with col9:
    inp_hour = st.slider("Admission hour", min_value=0, max_value=23, value=12)
with col10:
    inp_day = st.slider("Admission day of week", min_value=0, max_value=6, value=2)
with col11:
    inp_month = st.slider("Admission month", min_value=1, max_value=12, value=6)

if model_bundle is not None and isinstance(model_bundle, dict) and "models" in model_bundle:
    final_model_name = resolve_final_model_name(model_bundle, model_card_df)
    model = model_bundle["models"].get(final_model_name)
    feature_columns = model_bundle.get("feature_columns", [])
    balanced_threshold, recall_priority_threshold = resolve_thresholds(model_bundle, threshold_df)

    if model is not None and feature_columns:
        X_new = prepare_inference_row(
            age=inp_age,
            gender=inp_gender,
            admission_type=inp_type,
            admission_location=inp_location,
            insurance=inp_ins,
            language=inp_lang,
            marital_status=inp_marital,
            race=inp_race,
            admit_hour=inp_hour,
            admit_dayofweek=inp_day,
            admit_month=inp_month,
            feature_columns=feature_columns,
        )

        prob = float(model.predict_proba(X_new)[:, 1][0])
        predicted_class = int(prob >= balanced_threshold)
        tier = risk_tier(prob, balanced_threshold, recall_priority_threshold)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Predicted class (balanced threshold)", "Long stay" if predicted_class == 1 else "Not long stay")
        with m2:
            st.metric("Risk probability", f"{prob * 100:.1f}%")
        with m3:
            st.metric("Risk tier", tier)

        st.caption(
            f"Model: {final_model_name} | Balanced threshold: {balanced_threshold:.2f} | "
            f"Recall-priority threshold: {recall_priority_threshold:.2f}"
        )
        st.info(
            "Interpretation: this is an admission-time operational estimate to support planning and escalation workflow design. "
            "It is not a clinical diagnosis or treatment recommendation."
        )
    else:
        fallback = simple_risk_score(inp_age, inp_type, int(inp_day in [5, 6]), inp_ins)
        st.metric("Estimated long-stay risk (fallback)", f"{fallback * 100:.1f}%")
        st.warning("Model artifacts were found but incomplete. Run notebook end-to-end to regenerate outputs.")
else:
    # Graceful fallback keeps the dashboard usable even before notebook packaging.
    fallback = simple_risk_score(inp_age, inp_type, int(inp_day in [5, 6]), inp_ins)
    st.metric("Estimated long-stay risk (fallback)", f"{fallback * 100:.1f}%")
    st.warning(
        "Final model artifacts are missing. Run the notebook through the final packaging section to create "
        "`baseline_models.joblib`, `final_model_card.csv`, and `threshold_tuning.csv`."
    )
