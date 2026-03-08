# Hospital LOS Case Summary

## Problem
Hospitals need earlier visibility into likely long-stay admissions because a relatively small patient segment can drive disproportionate bed utilization and throughput pressure.

## Approach
- Built a clean inpatient cohort from MIMIC-style `admissions` + `patients`.
- Calculated LOS from admission/discharge timestamps and filtered invalid stays.
- Engineered admission-time-safe features and excluded leakage-prone fields.
- Compared two long-stay definitions: `LOS > 7 days` and `LOS >= p90`.
- Trained baseline models and selected a final model using both performance and explainability.
- Tuned operating thresholds for workflow use.

## Final Definition and Model
- Final label: **`LOS > 7 days` (`gt_7d`)**
- Final model: **`Random Forest` (`random_forest`)**
- Final balanced threshold: **`0.40`**
- Interpretability reference: **Logistic Regression**

## Key Findings
1. LOS is strongly right-skewed with a long-stay tail.
2. Label choice materially changes prevalence and triage workload.
3. Random Forest produced the strongest predictive discrimination on the current run.
4. Threshold selection meaningfully changes precision/recall tradeoff.
5. Admission-time features provide useful early operational signal.

## Actionable Recommendations
1. Operationalize `LOS > 7 days` as the primary long-stay flag.
2. Start with threshold `0.40` for balanced operations monitoring.
3. Re-tune threshold regularly as case mix shifts.
4. Add richer context features (service line/diagnosis) in next iteration.
5. Validate externally on larger cohorts before production deployment.

## Limitations
- Current evidence is based on demo-scale data.
- Probabilities are operational estimates, not clinical directives.
- Calibration and generalization require ongoing validation.

## Quick Verification
- Notebook completes through final artifact packaging with no errors.
- Dashboard loads model artifacts and displays prediction + risk tier.
- Final decision values remain consistent:
  - Label `LOS > 7 days`
  - Model `random_forest`
  - Threshold `0.40`
