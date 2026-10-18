-- ===========================================================================
-- 01_vintage_analysis.sql
-- ===========================================================================
-- Vintage analysis: default rate by origination cohort.
--
-- Stored procedure-compatible with HiveQL / Spark SQL (ANSI functions only,
-- -- comments, CTEs). It runs unchanged on a Hive/Spark cluster; locally DuckDB
-- executes the identical text over the parquet extract, acting as a stand-in
-- for a distributed warehouse.
--
-- Input table : accepted  (Lending Club loans with a resolved outcome)
-- Output grain: origination year x origination month
-- ===========================================================================

WITH resolved AS (
    SELECT
        issue_year,
        issue_ym,
        default_flag,
        loan_amnt,
        CASE WHEN default_flag = 1 THEN loan_amnt ELSE 0 END AS defaulted_amount
    FROM accepted
    WHERE resolved_flag = 1
      AND issue_year IS NOT NULL
)

SELECT
    issue_year                                                    AS origination_year,
    issue_ym                                                      AS origination_month,
    COUNT(*)                                                      AS n_loans,
    SUM(default_flag)                                             AS n_defaults,
    ROUND(AVG(CAST(default_flag AS DOUBLE)), 4)                   AS default_rate,
    ROUND(SUM(loan_amnt), 2)                                      AS exposure,
    ROUND(SUM(defaulted_amount), 2)                               AS defaulted_exposure,
    ROUND(SUM(defaulted_amount) / NULLIF(SUM(loan_amnt), 0), 4)   AS exposure_weighted_default_rate
FROM resolved
GROUP BY issue_year, issue_ym
ORDER BY issue_year, issue_ym;
