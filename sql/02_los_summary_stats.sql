-- 02_los_summary_stats.sql
-- Purpose:
--   Summarize LOS distribution and compare long-stay candidate thresholds.
--   Both candidate labels are reported; final modeling policy uses LOS > 7 days.
--
-- Output blocks:
--   Query A: overall LOS stats + thresholds + prevalence
--   Query B: subgroup long-stay rates by key operational dimensions
-- Standalone usage:
--   Execute file in PostgreSQL; returns two result sets in order.

-- Query A: overall cohort metrics and long-stay candidate prevalence.
WITH admissions_base AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.admission_type,
        a.insurance,
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
        ab.admission_type,
        ab.insurance,
        ab.hospital_expire_flag,
        EXTRACT(EPOCH FROM (ab.dischtime - ab.admittime)) / 86400.0 AS los_days
    FROM admissions_base AS ab
    WHERE ab.hadm_row_num = 1
      AND ab.admittime IS NOT NULL
      AND ab.dischtime IS NOT NULL
      AND ab.dischtime > ab.admittime
),
thresholds AS (
    SELECT
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY c.los_days) AS p90_los_days
    FROM cohort AS c
)
SELECT
    COUNT(*) AS n_admissions,
    AVG(c.los_days) AS mean_los_days,
    STDDEV_SAMP(c.los_days) AS std_los_days,
    MIN(c.los_days) AS min_los_days,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY c.los_days) AS p25_los_days,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY c.los_days) AS median_los_days,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY c.los_days) AS p75_los_days,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY c.los_days) AS p90_los_days,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY c.los_days) AS p95_los_days,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY c.los_days) AS p99_los_days,
    MAX(c.los_days) AS max_los_days,
    7.0 AS long_stay_days_threshold,
    t.p90_los_days AS long_stay_p90_threshold,
    AVG(CASE WHEN c.los_days > 7.0 THEN 1.0 ELSE 0.0 END) AS long_stay_share_gt_7d,
    AVG(CASE WHEN c.los_days >= t.p90_los_days THEN 1.0 ELSE 0.0 END) AS long_stay_share_gte_p90
FROM cohort AS c
CROSS JOIN thresholds AS t;

-- Query B: subgroup summary for operational targeting.
WITH admissions_base AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.admission_type,
        a.insurance,
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
        COALESCE(ab.admission_type, 'Unknown') AS admission_type,
        COALESCE(ab.insurance, 'Unknown') AS insurance,
        COALESCE(ab.hospital_expire_flag::text, 'Unknown') AS hospital_expire_flag,
        EXTRACT(EPOCH FROM (ab.dischtime - ab.admittime)) / 86400.0 AS los_days
    FROM admissions_base AS ab
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
grouped AS (
    SELECT
        'admission_type' AS dimension,
        c.admission_type AS dimension_value,
        COUNT(*) AS n_admissions,
        AVG(c.los_days) AS mean_los_days,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY c.los_days) AS median_los_days,
        AVG(CASE WHEN c.los_days > 7.0 THEN 1.0 ELSE 0.0 END) AS long_stay_share_gt_7d,
        AVG(CASE WHEN c.los_days >= t.p90_los_days THEN 1.0 ELSE 0.0 END) AS long_stay_share_gte_p90
    FROM cohort AS c
    CROSS JOIN thresholds AS t
    GROUP BY c.admission_type

    UNION ALL

    SELECT
        'insurance' AS dimension,
        c.insurance AS dimension_value,
        COUNT(*) AS n_admissions,
        AVG(c.los_days) AS mean_los_days,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY c.los_days) AS median_los_days,
        AVG(CASE WHEN c.los_days > 7.0 THEN 1.0 ELSE 0.0 END) AS long_stay_share_gt_7d,
        AVG(CASE WHEN c.los_days >= t.p90_los_days THEN 1.0 ELSE 0.0 END) AS long_stay_share_gte_p90
    FROM cohort AS c
    CROSS JOIN thresholds AS t
    GROUP BY c.insurance

    UNION ALL

    SELECT
        'hospital_expire_flag' AS dimension,
        c.hospital_expire_flag AS dimension_value,
        COUNT(*) AS n_admissions,
        AVG(c.los_days) AS mean_los_days,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY c.los_days) AS median_los_days,
        AVG(CASE WHEN c.los_days > 7.0 THEN 1.0 ELSE 0.0 END) AS long_stay_share_gt_7d,
        AVG(CASE WHEN c.los_days >= t.p90_los_days THEN 1.0 ELSE 0.0 END) AS long_stay_share_gte_p90
    FROM cohort AS c
    CROSS JOIN thresholds AS t
    GROUP BY c.hospital_expire_flag
)
SELECT
    g.dimension,
    g.dimension_value,
    g.n_admissions,
    g.mean_los_days,
    g.median_los_days,
    g.long_stay_share_gt_7d,
    g.long_stay_share_gte_p90
FROM grouped AS g
ORDER BY g.dimension, g.long_stay_share_gte_p90 DESC, g.n_admissions DESC;
