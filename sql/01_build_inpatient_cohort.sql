-- 01_build_inpatient_cohort.sql
-- Purpose:
--   Build a clean base hospital-admission cohort for LOS analytics from MIMIC-IV.
--
-- MIMIC assumptions used here:
--   1) `mimiciv_hosp.admissions` is one row per hospital admission (`hadm_id`) in normal cases.
--      We still apply a defensive deduplication guard to ensure one-row-per-hadm_id.
--   2) LOS is calculated as the interval between `admittime` and `dischtime` in days.
--   3) Approximate age-at-admission uses anchor metadata:
--      age_at_admit ~= anchor_age + (year(admittime) - anchor_year)
--      This is de-identification-safe and useful for analytics, not exact birth-date age.
--   4) Both long-stay labels are analyzed downstream; final modeling label is LOS > 7 days.
-- Standalone usage:
--   Run directly in PostgreSQL; this query does not depend on temp tables.

WITH admissions_base AS (
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
admissions_dedup AS (
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
        ab.hospital_expire_flag
    FROM admissions_base AS ab
    WHERE ab.hadm_row_num = 1
),
joined_patients AS (
    SELECT
        ad.subject_id,
        ad.hadm_id,
        ad.admittime,
        ad.dischtime,
        ad.admission_type,
        ad.admission_location,
        ad.discharge_location,
        ad.insurance,
        ad.language,
        ad.marital_status,
        ad.race,
        ad.hospital_expire_flag,
        p.gender,
        p.anchor_age,
        p.anchor_year
    FROM admissions_dedup AS ad
    LEFT JOIN mimiciv_hosp.patients AS p
        ON ad.subject_id = p.subject_id
),
cohort_derived AS (
    SELECT
        jp.subject_id,
        jp.hadm_id,
        jp.admittime,
        jp.dischtime,
        jp.admission_type,
        jp.admission_location,
        jp.discharge_location,
        jp.insurance,
        jp.language,
        jp.marital_status,
        jp.race,
        jp.hospital_expire_flag,
        jp.gender,
        jp.anchor_age,
        jp.anchor_year,
        CASE
            WHEN jp.anchor_age IS NOT NULL
             AND jp.anchor_year IS NOT NULL
             AND jp.admittime IS NOT NULL
            THEN jp.anchor_age + (EXTRACT(YEAR FROM jp.admittime)::INT - jp.anchor_year)
            ELSE NULL
        END AS age_at_admit,
        EXTRACT(EPOCH FROM (jp.dischtime - jp.admittime)) / 86400.0 AS los_days
    FROM joined_patients AS jp
),
cohort_clean AS (
    SELECT
        cd.subject_id,
        cd.hadm_id,
        cd.admittime,
        cd.dischtime,
        cd.admission_type,
        cd.admission_location,
        cd.discharge_location,
        cd.insurance,
        cd.language,
        cd.marital_status,
        cd.race,
        cd.hospital_expire_flag,
        cd.gender,
        cd.anchor_age,
        cd.anchor_year,
        cd.age_at_admit,
        cd.los_days
    FROM cohort_derived AS cd
    WHERE cd.admittime IS NOT NULL
      AND cd.dischtime IS NOT NULL
      AND cd.dischtime > cd.admittime
      AND cd.los_days > 0
)
SELECT
    cc.subject_id,
    cc.hadm_id,
    cc.admittime,
    cc.dischtime,
    cc.admission_type,
    cc.admission_location,
    cc.discharge_location,
    cc.insurance,
    cc.language,
    cc.marital_status,
    cc.race,
    cc.hospital_expire_flag,
    cc.gender,
    cc.anchor_age,
    cc.anchor_year,
    cc.age_at_admit,
    cc.los_days
FROM cohort_clean AS cc
ORDER BY cc.admittime;
