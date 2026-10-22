-- ===========================================================================
-- 03_data_quality_and_leakage.sql
-- ===========================================================================
-- Data-quality and leakage-control checks documented as SQL so the
-- assumptions are auditable and can be re-run on any warehouse.
-- ===========================================================================

-- (a) Row counts and the outcome funnel ------------------------------------
SELECT
    count(*)                                            AS n_rows_total,
    sum(resolved_flag)                                  AS n_resolved,
    sum(default_flag)                                   AS n_defaults,
    round(avg(default_flag) FILTER (WHERE resolved_flag = 1), 4) AS default_rate_resolved
FROM accepted;

-- (b) Missing-value profile for the modeling features ---------------------
-- (share of NULLs per candidate feature, descending)
SELECT 'loan_amnt' AS variable, avg(CASE WHEN loan_amnt IS NULL THEN 1.0 ELSE 0 END) AS pct_missing FROM accepted
UNION ALL SELECT 'annual_inc',   avg(CASE WHEN annual_inc   IS NULL THEN 1.0 ELSE 0 END) FROM accepted
UNION ALL SELECT 'dti',          avg(CASE WHEN dti          IS NULL THEN 1.0 ELSE 0 END) FROM accepted
UNION ALL SELECT 'revol_util',   avg(CASE WHEN revol_util   IS NULL THEN 1.0 ELSE 0 END) FROM accepted
UNION ALL SELECT 'emp_length',   avg(CASE WHEN emp_length   IS NULL THEN 1.0 ELSE 0 END) FROM accepted
UNION ALL SELECT 'mths_since_last_delinq', avg(CASE WHEN mths_since_last_delinq IS NULL THEN 1.0 ELSE 0 END) FROM accepted
ORDER BY pct_missing DESC;

-- (c) Leakage control -------------------------------------------------------
-- These columns are populated *after* origination and/or encode the outcome.
-- They are excluded from every model; the query proves they are non-null for
-- the vast majority of loans and would therefore trivially leak the target.
SELECT
    count(*)                                                        AS n_resolved_loans,
    avg(CASE WHEN total_pymnt   IS NOT NULL THEN 1.0 ELSE 0 END)    AS pct_has_total_pymnt,
    avg(CASE WHEN recoveries    IS NOT NULL THEN 1.0 ELSE 0 END)    AS pct_has_recoveries,
    avg(CASE WHEN last_pymnt_d  IS NOT NULL THEN 1.0 ELSE 0 END)    AS pct_has_last_pymnt_d,
    avg(CASE WHEN out_prncp     IS NOT NULL THEN 1.0 ELSE 0 END)    AS pct_has_out_prncp
FROM accepted
WHERE resolved_flag = 1;

-- (d) Outlier scan: extreme values in key exposure / income fields ---------
SELECT
    min(loan_amnt) AS min_loan,  max(loan_amnt) AS max_loan,
    min(annual_inc) AS min_income, max(annual_inc) AS max_income,
    min(dti) AS min_dti, max(dti) AS max_dti,
    min(revol_util) AS min_revol_util, max(revol_util) AS max_revol_util
FROM accepted
WHERE resolved_flag = 1;
