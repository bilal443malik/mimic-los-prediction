# Hospital Length of Stay (LOS) Case Study - MIMIC-IV

Portfolio-ready healthcare analytics project for identifying likely long-stay admissions and translating results into operations-focused actions.

## Business Problem
Hospitals face bed-capacity pressure because a small share of admissions consumes a disproportionate number of bed-days.  
Goal: predict long-stay admissions at admission time so operations teams can prioritize discharge planning and care coordination earlier.

## Dataset
- Target data design: **MIMIC-IV v3.1** (PhysioNet)
- Core tables used in this project phase:
  - `admissions`
  - `patients`
- Current runnable local sample: **MIMIC-IV Demo 2.2**

## Final Submission Decisions
- Final long-stay label: **`LOS > 7 days`** (`gt_7d`)
- Final prediction model: **`random_forest`**
- Final balanced operating threshold: **`0.40`**
- Interpretability reference model: **`logistic_regression`**

Why this setup:
- `LOS > 7 days` is easier for non-technical stakeholders to use operationally.
- Random Forest produced the strongest ranking/discrimination in current results.
- Threshold `0.40` gave the best balanced policy point in current tuning output.

## Method Overview
1. Build one-row-per-admission cohort (`hadm_id`) and compute LOS from `admittime`/`dischtime`.
2. Clean data (missingness, dedup guard, LOS sanity checks).
3. Engineer admission-time-safe features only.
4. Train baseline models (Logistic, Random Forest, optional XGBoost).
5. Evaluate with imbalance-aware metrics (PR-AUC, ROC-AUC, recall, precision, F1).
6. Tune threshold policy for operations use.
7. Package artifacts for notebook + dashboard use.

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
- Feature importance: `outputs/tables/feature_importance_*.csv`

## How to Run
1. Setup environment
```bash
cd /home/dell/projjLOS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Place data files
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

## Findings (Current Demo Run)
1. LOS distribution is right-skewed with a long-stay tail.
2. Label definition changes prevalence materially (`>7d` vs `>=p90`).
3. Random Forest outperformed baseline alternatives on current demo metrics.
4. Threshold choice strongly affects recall/precision and operational workload.

## Recommendations
1. Use `LOS > 7 days` as the primary operational label.
2. Use threshold `0.40` as the current balanced operating point.
3. Re-tune threshold as case mix changes.
4. Expand features (service line/diagnosis context) before production use.
5. Validate on a larger/full cohort before deployment.

## Limitations
- Current results are demo-scale and may not generalize directly.
- Output is an operational risk estimate, not a clinical decision tool.
- Associations are not causal effects.

## Final Verification Checklist
- Notebook runs with no errors through final packaging cell.
- Dashboard loads artifacts and shows model probability + risk tier.
- These files exist and are non-empty:
  - `outputs/models/baseline_models.joblib`
  - `outputs/tables/model_metrics.csv`
  - `outputs/tables/threshold_tuning.csv`
  - `outputs/tables/final_model_card.csv`
- Final values are consistent across notebook, dashboard, and docs:
  - Label: `LOS > 7 days`
  - Model: `random_forest`
  - Threshold: `0.40`

## Disclaimer
Educational/portfolio use with de-identified data only. Not for direct clinical decision-making.
