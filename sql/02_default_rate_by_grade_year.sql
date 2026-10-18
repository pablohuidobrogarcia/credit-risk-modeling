-- ===========================================================================
-- 02_default_rate_by_grade_year.sql
-- ===========================================================================
-- Portfolio heat map: default rate by risk grade, globally and per
-- origination year. HiveQL / Spark SQL compatible; executed locally by DuckDB.
-- ===========================================================================

-- (a) Lifetime default rate per grade -------------------------------------
WITH resolved AS (
    SELECT issue_year, grade, loan_amnt, default_flag
    FROM accepted
    WHERE resolved_flag = 1 AND issue_year IS NOT NULL
)
SELECT
    'by_grade'                                      AS view_name,
    grade,
    count(*)                                        AS n_loans,
    round(avg(cast(default_flag AS double)), 4)     AS default_rate,
    round(sum(loan_amnt), 2)                        AS exposure
FROM resolved
GROUP BY grade
ORDER BY grade;

-- (b) Grade x origination-year default rate (drill-down) ------------------
SELECT
    grade,
    issue_year,
    count(*)                                        AS n_loans,
    round(avg(cast(default_flag AS double)), 4)     AS default_rate
FROM accepted
WHERE resolved_flag = 1 AND issue_year IS NOT NULL
GROUP BY grade, issue_year
ORDER BY grade, issue_year;
