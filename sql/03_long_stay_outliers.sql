-- 03_long_stay_outliers.sql
-- Purpose:
--   Identify longest-stay admissions for operations and data-quality review.
--
-- Long-stay candidates included in output:
--   A) LOS > 7 days
--   B) LOS >= p90
-- Modeling note:
--   Final classifier policy uses A) LOS > 7 days; B is retained for sensitivity context.
--
-- Tuning knobs (in `params` CTE):
--   - `top_n_by_los`: retain top-N longest stays
--   - `min_percent_rank`: retain very high percentile stays (e.g., top 1%)
-- Standalone usage:
--   Execute directly in PostgreSQL for operational outlier review.

WITH params AS (
    SELECT
        100::INT AS top_n_by_los,
        0.99::NUMERIC AS min_percent_rank
),
admissions_base AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.admission_type,
        a.admission_location,
        a.discharge_location,
        a.insurance,
        a.language,
        a.marital_status,
        a.race,
        a.hospital_expire_flag,
        ROW_NUMBER() OVER (
            PARTITION BY a.hadm_id
            ORDER BY a.admittime NULLS LAST, a.dischtime NULLS LAST, a.subject_id
        ) AS hadm_row_num
    FROM mimiciv_hosp.admissions AS a
),
cohort AS (
    SELECT
        ab.subject_id,
        ab.hadm_id,
        ab.admittime,
        ab.dischtime,
        ab.admission_type,
        ab.admission_location,
        ab.discharge_location,
        ab.insurance,
        ab.language,
        ab.marital_status,
        ab.race,
        ab.hospital_expire_flag,
        p.gender,
        p.anchor_age,
        p.anchor_year,
        CASE
            WHEN p.anchor_age IS NOT NULL
             AND p.anchor_year IS NOT NULL
             AND ab.admittime IS NOT NULL
            THEN p.anchor_age + (EXTRACT(YEAR FROM ab.admittime)::INT - p.anchor_year)
            ELSE NULL
        END AS age_at_admit,
        EXTRACT(EPOCH FROM (ab.dischtime - ab.admittime)) / 86400.0 AS los_days
    FROM admissions_base AS ab
    LEFT JOIN mimiciv_hosp.patients AS p
        ON ab.subject_id = p.subject_id
    WHERE ab.hadm_row_num = 1
      AND ab.admittime IS NOT NULL
      AND ab.dischtime IS NOT NULL
      AND ab.dischtime > ab.admittime
),
thresholds AS (
    SELECT
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY c.los_days) AS p90_los_days
    FROM cohort AS c
),
ranked AS (
    SELECT
        c.*,
        DENSE_RANK() OVER (ORDER BY c.los_days DESC) AS los_dense_rank,
        PERCENT_RANK() OVER (ORDER BY c.los_days) AS los_percent_rank,
        t.p90_los_days,
        CASE WHEN c.los_days > 7.0 THEN 1 ELSE 0 END AS is_long_stay_gt_7d,
        CASE WHEN c.los_days >= t.p90_los_days THEN 1 ELSE 0 END AS is_long_stay_gte_p90
    FROM cohort AS c
    CROSS JOIN thresholds AS t
)
SELECT
    r.subject_id,
    r.hadm_id,
    r.admittime,
    r.dischtime,
    r.admission_type,
    r.admission_location,
    r.discharge_location,
    r.insurance,
    r.language,
    r.marital_status,
    r.race,
    r.hospital_expire_flag,
    r.gender,
    r.anchor_age,
    r.anchor_year,
    r.age_at_admit,
    r.los_days,
    r.p90_los_days,
    r.is_long_stay_gt_7d,
    r.is_long_stay_gte_p90,
    r.los_dense_rank,
    r.los_percent_rank
FROM ranked AS r
CROSS JOIN params AS p
WHERE r.los_dense_rank <= p.top_n_by_los
   OR r.los_percent_rank >= p.min_percent_rank
ORDER BY r.los_days DESC, r.hadm_id;
