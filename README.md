# Hospital Length of Stay (LOS) Case Study - MIMIC-IV

End-to-end healthcare analytics case study focused on early identification of likely long-stay admissions for hospital operations planning.

## Repo Overview
This repository demonstrates a complete analytics delivery package:
- reproducible SQL cohort and LOS analysis
- modular Python pipeline for feature engineering and modeling
- story-driven notebook for business + technical walkthrough
- one-page Streamlit dashboard for decision support
- executive case summary for non-technical stakeholders

## Project Highlights (Recruiter / Client View)
- Solves a real operations problem: bed-capacity pressure from long-stay patients.
- Uses explainable admission-time features with explicit leakage controls.
- Compares label definitions and justifies a business-facing final choice.
- Delivers model + threshold policy, not just raw model metrics.
- Packages outputs for notebook, dashboard, and executive reporting.

## Business Problem
Hospitals face bed-capacity pressure because a small share of admissions consumes a disproportionate number of bed-days.  
Goal: predict long-stay admissions at admission time so operations teams can prioritize discharge planning and care coordination earlier.

## Dataset
- Target data design: **MIMIC-IV v3.1** (PhysioNet)
- Source link: https://physionet.org/content/mimiciv/3.1/
- Access requires PhysioNet credentialing and data-use compliance.
- Core tables used in this phase:
  - `admissions`
  - `patients`
- Current runnable local sample: **MIMIC-IV Demo 2.2** (for workflow demonstration)

## Final Decisions
- Final long-stay label: **`LOS > 7 days`** (`gt_7d`)
- Final prediction model: **`random_forest`**
- Final balanced operating threshold: **`0.40`**
- Interpretability reference model: **`logistic_regression`**

Why this setup:
- `LOS > 7 days` is easier for non-technical stakeholders to operationalize.
- Random Forest delivered the best discrimination in this run.
- Threshold `0.40` gave the strongest balanced policy point in tuning output.

## Method Summary
1. Build one-row-per-admission cohort (`hadm_id`) and compute LOS from `admittime`/`dischtime`.
2. Run data quality checks (missingness, dedup guard, LOS sanity).
3. Engineer admission-time-safe features and exclude leakage-prone columns.
4. Train baseline models (Logistic, Random Forest, optional XGBoost).
5. Evaluate with PR-AUC, ROC-AUC, recall, precision, F1, and confusion matrices.
6. Tune threshold policy for operations.
7. Save artifacts for dashboard inference and case packaging.

## Repository Structure
```text
hospital-los-case-study/
├── notebooks/los_case_study.ipynb
├── src/
├── sql/
├── dashboard/app.py
├── outputs/
├── case_summary.md
├── README.md
└── requirements.txt
```

## Key Outputs
- Cohort table: `data/interim/cohort_base.csv`
- Model artifact: `outputs/models/baseline_models.joblib`
- Metrics: `outputs/tables/model_metrics.csv`
- Threshold tuning: `outputs/tables/threshold_tuning.csv`
- Final decision card: `outputs/tables/final_model_card.csv`
- Feature importance tables: `outputs/tables/feature_importance_*.csv`

## Suggested Screenshots (Add Before Publishing)
Save screenshots under `assets/screenshots/` and update links/captions below.

1. **Notebook - LOS Distribution**
`assets/screenshots/01_notebook_los_distribution.png`  
Capture from: `notebooks/los_case_study.ipynb`, section **6) LOS Distribution and Summary** (histogram plot).

2. **Notebook - Label/Model Final Decision**
`assets/screenshots/02_notebook_final_decision.png`  
Capture from: notebook section **13) Final Label and Model Decision Support** (label prevalence + final model card output).

3. **Notebook - Threshold Policy**
`assets/screenshots/03_notebook_threshold_tradeoff.png`  
Capture from: notebook section **14) Threshold Policy** (tradeoff chart + selected thresholds table).

4. **Notebook - Feature Importance**
`assets/screenshots/04_notebook_feature_importance.png`  
Capture from: notebook section **15) Feature Importance and Probability Interpretation** (RF or logistic importance chart).

5. **Dashboard - LOS and Risk Prediction**
`assets/screenshots/05_dashboard_main.png`  
Capture from: `streamlit run dashboard/app.py` with:
- LOS distribution visible
- long-stay share-by-dimension chart visible
- risk prediction panel populated

Suggested caption format:
`Figure X. <what is shown> and why it matters operationally.`

## How to Run
1. Setup environment
```bash
cd /home/dell/projjLOS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Place required files
```text
data/raw/mimiciv/hosp/
├── admissions.csv.gz
└── patients.csv.gz
```

3. Run notebook
- Open `notebooks/los_case_study.ipynb`
- Run top-to-bottom through final packaging section

4. Run dashboard
```bash
streamlit run dashboard/app.py
```

## How to Demo This Project (5-7 Minutes)
1. Problem framing: explain bed-capacity impact of long-stay admissions.
2. Data foundation: show cohort logic and LOS calculation from timestamps.
3. Label decision: show `>7 days` vs `>=p90` comparison and why `>7d` was selected.
4. Model results: show model comparison and why Random Forest was chosen.
5. Threshold policy: explain why `0.40` was selected and what recall/precision tradeoff means.
6. Dashboard: run one example prediction and interpret risk tier in plain language.

## Interview Walkthrough (Short Script)
- **Context:** “This project predicts likely long-stay admissions at admission time for operations planning.”  
- **Data:** “I built a clean one-row-per-admission cohort from MIMIC admissions + patients.”  
- **Modeling:** “I used admission-time-safe features, compared baselines, and selected Random Forest.”  
- **Decisioning:** “I tuned thresholds and set a balanced policy threshold at 0.40.”  
- **Business translation:** “Output is packaged as a dashboard plus an executive summary, with clear limitations and non-clinical-use caveats.”

## Business Impact
- Earlier identification of likely long stays can prioritize case management and discharge planning.
- Threshold policy allows teams to balance sensitivity vs workflow burden.
- A standardized cohort + model artifact pipeline supports repeatable reporting over time.

## Findings (Current Demo Run)
1. LOS is right-skewed with a long-stay tail.
2. Label definition materially changes prevalence (`>7d` vs `>=p90`).
3. Random Forest outperformed baseline alternatives on current demo metrics.
4. Threshold choice strongly affects recall/precision and operational workload.

## Recommendations
1. Use `LOS > 7 days` as the primary operational label.
2. Use threshold `0.40` as the current balanced operating point.
3. Re-tune threshold as case mix changes.
4. Expand features (service-line/diagnosis context) before production use.
5. Validate on larger/full cohorts before deployment.

## Limitations
- Current results are demo-scale and may not generalize directly.
- Output is an operational risk estimate, not a clinical decision tool.
- Associations are not causal effects.

## Project Verification Checklist
- Run notebook end-to-end through final packaging (includes model training).
- Confirm notebook completes with no errors.
- Launch dashboard and confirm probability + risk tier render correctly.
- Confirm artifacts exist:
  - `outputs/models/baseline_models.joblib`
  - `outputs/tables/model_metrics.csv`
  - `outputs/tables/threshold_tuning.csv`
  - `outputs/tables/final_model_card.csv`
- Confirm final values are consistent:
  - Label: `LOS > 7 days`
  - Model: `random_forest`
  - Threshold: `0.40`

## Disclaimer
Educational/portfolio use with de-identified data only. Not for direct clinical decision-making.
