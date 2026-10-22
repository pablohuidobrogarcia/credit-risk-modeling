from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import duckdb
import pandas as pd

from .config import Config
from .utils import set_seeds


def detect_raw_files(root_dir: Path | None = None) -> Tuple[Path | None, Path | None]:
    """Locate the accepted and rejected Lending Club CSV files in ``data/raw``.

    Detection is keyword based (``accepted``/``rejected`` in the file name) so
    that slight differences in capitalization or suffixes still work.
    """
    if root_dir is None:
        c = Config.load()
        root_dir = c.path("raw_dir")
    acc = None
    rej = None
    for p in root_dir.glob("*"):
        name = p.name.lower()
        if "accepted" in name:
            acc = p
        if "rejected" in name:
            rej = p
    return acc, rej


def build_accepted_parquet(force: bool = False) -> Path:
    """Convert the raw accepted-loan CSV into a typed parquet file.

    Only columns used downstream are typed/cast; the rest are kept as text so
    that unexpected values never break the ingestion. Footer rows
    (``Total amount funded ...``) are dropped.
    """
    c = Config.load()
    set_seeds(c.seed)
    acc_csv, _ = detect_raw_files()
    if acc_csv is None:
        acc_csv = c.path("accepted_csv")
    out_parquet = c.path("accepted_parquet")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    if out_parquet.exists() and not force:
        return out_parquet

    con = duckdb.connect()
    con.execute("SET threads TO 4")
    con.execute(
        f"CREATE OR REPLACE TABLE accepted_raw AS "
        f"SELECT * FROM read_csv_auto('{acc_csv.as_posix()}', all_varchar=TRUE, ignore_errors=TRUE)"
    )
    # Keep only genuine loan rows (numeric id); the file ends with aggregate footers.
    con.execute(
        """
        CREATE OR REPLACE TABLE accepted_clean AS
        SELECT * FROM accepted_raw
        WHERE id IS NOT NULL AND regexp_matches(id, '^[0-9]+$')
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE accepted_derived AS
        SELECT
               TRY_CAST(id AS BIGINT)            AS id,
               TRY_CAST(member_id AS BIGINT)     AS member_id,
               TRY_CAST(loan_amnt AS DOUBLE)     AS loan_amnt,
               TRY_CAST(funded_amnt AS DOUBLE)   AS funded_amnt,
               TRY_CAST(funded_amnt_inv AS DOUBLE) AS funded_amnt_inv,
               term,
               TRY_CAST(replace(int_rate, '%', '') AS DOUBLE) AS int_rate,
               TRY_CAST(installment AS DOUBLE)   AS installment,
               grade,
               sub_grade,
               emp_title,
               emp_length,
               home_ownership,
               TRY_CAST(annual_inc AS DOUBLE)    AS annual_inc,
               verification_status,
               issue_d,
               loan_status,
               "desc" AS loan_desc,
               purpose,
               title,
               zip_code,
               addr_state,
               TRY_CAST(dti AS DOUBLE)           AS dti,
               TRY_CAST(delinq_2yrs AS INT)      AS delinq_2yrs,
               earliest_cr_line,
               TRY_CAST(inq_last_6mths AS INT)   AS inq_last_6mths,
               TRY_CAST(mths_since_last_delinq AS INT) AS mths_since_last_delinq,
               TRY_CAST(mths_since_last_record AS INT) AS mths_since_last_record,
               TRY_CAST(open_acc AS INT)         AS open_acc,
               TRY_CAST(pub_rec AS INT)          AS pub_rec,
               TRY_CAST(revol_bal AS DOUBLE)     AS revol_bal,
               TRY_CAST(replace(revol_util, '%', '') AS DOUBLE) AS revol_util,
               TRY_CAST(total_acc AS INT)        AS total_acc,
               initial_list_status,
               TRY_CAST(out_prncp AS DOUBLE)     AS out_prncp,
               TRY_CAST(total_pymnt AS DOUBLE)   AS total_pymnt,
               TRY_CAST(total_rec_prncp AS DOUBLE) AS total_rec_prncp,
               TRY_CAST(total_rec_int AS DOUBLE) AS total_rec_int,
               TRY_CAST(total_rec_late_fee AS DOUBLE) AS total_rec_late_fee,
               TRY_CAST(recoveries AS DOUBLE)    AS recoveries,
               TRY_CAST(collection_recovery_fee AS DOUBLE) AS collection_recovery_fee,
               last_pymnt_d,
               TRY_CAST(last_pymnt_amnt AS DOUBLE) AS last_pymnt_amnt,
               next_pymnt_d,
               last_credit_pull_d,
               TRY_CAST(collections_12_mths_ex_med AS INT) AS collections_12_mths_ex_med,
               TRY_CAST(mths_since_last_major_derog AS INT) AS mths_since_last_major_derog,
               TRY_CAST(policy_code AS INT)      AS policy_code,
               application_type,
               TRY_CAST(tot_coll_amt AS DOUBLE)  AS tot_coll_amt,
               TRY_CAST(tot_cur_bal AS DOUBLE)   AS tot_cur_bal,
               TRY_CAST(total_rev_hi_lim AS DOUBLE) AS total_rev_hi_lim,
               CASE
                 WHEN loan_status IN ('Charged Off', 'Default') THEN 1
                 WHEN loan_status = 'Fully Paid' THEN 0
                 ELSE NULL
               END AS default_flag,
               CASE
                 WHEN loan_status IN ('Charged Off', 'Default', 'Fully Paid') THEN 1
                 ELSE 0
               END AS resolved_flag,
               strftime(strptime(issue_d, '%b-%Y'), '%Y-%m') AS issue_ym,
               TRY_CAST(strftime(strptime(issue_d, '%b-%Y'), '%Y') AS INT) AS issue_year
        FROM accepted_clean
        """
    )
    con.execute(
        f"COPY (SELECT * FROM accepted_derived) TO '{out_parquet.as_posix()}' (FORMAT PARQUET, COMPRESSION 'zstd')"
    )
    con.close()
    return out_parquet


class _AcceptedConnection:
    """Thin wrapper exposing a DuckDB cursor plus the resolved parquet path."""

    def __init__(self, con: duckdb.DuckDBPyConnection, acc_parquet: Path):
        self._con = con
        self._acc_parquet = acc_parquet

    def execute(self, *args, **kwargs):
        return self._con.execute(*args, **kwargs)

    def close(self) -> None:
        self._con.close()


def _accepted_connection() -> _AcceptedConnection:
    """Open a DuckDB connection to the accepted-loan parquet (building it if needed)."""
    c = Config.load()
    acc_parquet = c.path("accepted_parquet")
    if not acc_parquet.exists():
        acc_parquet = build_accepted_parquet(force=False)
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    return _AcceptedConnection(con, acc_parquet)


def population_totals() -> Dict[str, int]:
    """Population counts of the accepted file (documents the sampling fraction)."""
    con = _accepted_connection()
    try:
        row = con.execute(
            f"""
            SELECT COUNT(*)                                    AS n_accepted,
                   SUM(resolved_flag)                          AS n_resolved,
                   SUM(CASE WHEN loan_status = 'Fully Paid'   THEN 1 ELSE 0 END) AS n_fully_paid,
                   SUM(CASE WHEN loan_status = 'Charged Off'  THEN 1 ELSE 0 END) AS n_charged_off,
                   SUM(CASE WHEN loan_status = 'Default'      THEN 1 ELSE 0 END) AS n_default_status
            FROM '{con._acc_parquet.as_posix()}'
            """
        ).df()
    finally:
        con.close()
    return {k: int(v) for k, v in row.iloc[0].items()}


def resolution_by_year() -> pd.DataFrame:
    """Right-censoring diagnostic: how many loans of each vintage are resolved.

    The modeling sample only contains loans with a final outcome, so a loan
    issued in 2017-2018 only enters the sample if it defaulted or was prepaid
    early enough to mature before the data cut-off (2018-12). This table makes
    that selection explicit: ``resolved_share`` is the fraction of the
    *population* vintage that had a known outcome by the cut-off.
    """
    con = _accepted_connection()
    try:
        tab = con.execute(
            f"""
            SELECT issue_year,
                   COUNT(*)                                    AS n_loans,
                   SUM(resolved_flag)                          AS n_resolved,
                   SUM(COALESCE(default_flag, 0))              AS n_defaulted,
                   ROUND(SUM(resolved_flag) * 1.0 / COUNT(*), 4) AS resolved_share,
                   ROUND(SUM(COALESCE(default_flag, 0)) * 1.0
                         / NULLIF(SUM(resolved_flag), 0), 4)     AS default_rate_resolved
            FROM '{con._acc_parquet.as_posix()}'
            WHERE issue_year IS NOT NULL
            GROUP BY issue_year
            ORDER BY issue_year
            """
        ).df()
    finally:
        con.close()
    return tab


def resolution_by_year_term() -> pd.DataFrame:
    """Resolved share of every (origination year x term) cell of the population.

    This is the table the validation split is built from. ``term`` is the
    contractual tenor (36 or 60 months). A 60-month loan issued in 2014 reaches
    its scheduled maturity in 2019 - past the 2018-12 data cut-off - so its
    resolved share is much lower than a 36-month loan of the same vintage.
    Splitting by *both* dimensions reveals exactly which cells are mature
    enough for an unbiased out-of-time test.
    """
    con = _accepted_connection()
    try:
        tab = con.execute(
            f"""
            SELECT issue_year,
                   regexp_replace(term, '[^0-9]', '', 'g')     AS term_months,
                   COUNT(*)                                     AS n_loans,
                   SUM(resolved_flag)                           AS n_resolved,
                   ROUND(SUM(resolved_flag) * 1.0 / COUNT(*), 4) AS resolved_share,
                   ROUND(SUM(COALESCE(default_flag, 0)) * 1.0
                         / NULLIF(SUM(resolved_flag), 0), 4)      AS default_rate_resolved
            FROM '{con._acc_parquet.as_posix()}'
            WHERE issue_year IS NOT NULL AND term IS NOT NULL
            GROUP BY issue_year, term_months
            ORDER BY issue_year, term_months
            """
        ).df()
    finally:
        con.close()
    return tab


def mature_split_rule(
    table: "pd.DataFrame | None" = None,
    min_resolved_share: float = 0.90,
) -> Dict[str, object]:
    """Derive the primary validation split from the resolved-share table.

    Rule
    ----
    A ``(year, term)`` cell is usable for the primary out-of-time test when at
    least ``min_resolved_share`` (default 90%) of its population loans had a
    *final* outcome by the data cut-off. Cells below the threshold are excluded
    from the primary split and analyzed separately as the "immature" window,
    so their censoring bias cannot leak into the headline metrics.

    The returned object records the rule, the threshold, every immature cell and
    the per-term verdict, so the decision is fully auditable from
    ``summary.json``.
    """
    tab = resolution_by_year_term() if table is None else table.copy()
    tab["usable"] = tab["resolved_share"] >= float(min_resolved_share)
    # A term is usable when *every* year it appears in is usable (conservative).
    by_term = tab.groupby("term_months")["usable"].all().to_dict()
    immature = tab[~tab["usable"]].copy()
    mature = tab[tab["usable"]].copy()
    return {
        "min_resolved_share": float(min_resolved_share),
        "rule": (
            "A (issue_year, term) cell enters the primary split when at least "
            f"{min_resolved_share:.0%} of its population loans are resolved."
        ),
        "terms_fully_mature": sorted(str(t) for t, ok in by_term.items() if ok),
        "terms_with_immature_cells": sorted(str(t) for t, ok in by_term.items() if not ok),
        "mature_cells": mature[["issue_year", "term_months", "resolved_share"]].to_dict("records"),
        "immature_cells": immature[["issue_year", "term_months", "resolved_share"]].to_dict("records"),
        "mature_years": sorted(int(y) for y in mature["issue_year"].unique()),
    }


def build_modeling_sample(
    force: bool = False,
    sample_size: Optional[int] = None,
) -> Path:
    """Create a reproducible, year-stratified sample of resolved loans.

    Resolved loans are those with a final outcome (Fully Paid / Charged Off /
    Default). Sampling is deterministic: rows are ranked within each
    origination year by a stable hash, so the same seed always yields the same
    sample and no RNG dependency is introduced.
    """
    c = Config.load()
    set_seeds(c.seed)
    acc_parquet = c.path("accepted_parquet")
    if not acc_parquet.exists():
        acc_parquet = build_accepted_parquet(force=False)
    out_model = c.path("modeling_parquet")
    out_model.parent.mkdir(parents=True, exist_ok=True)
    if out_model.exists() and not force:
        return out_model
    if sample_size is None:
        sample_size = c.data.get("sample_size", 500000)

    con = duckdb.connect()
    con.execute("SET threads TO 2")
    try:
        con.execute(f"CREATE OR REPLACE VIEW resolved AS SELECT * FROM '{acc_parquet.as_posix()}' WHERE resolved_flag = 1 AND issue_year IS NOT NULL")
        counts = con.execute(
            "SELECT issue_year, COUNT(*) AS cnt FROM resolved GROUP BY issue_year ORDER BY issue_year"
        ).df()
        total = int(counts["cnt"].sum()) if len(counts) else 0
        if total == 0:
            raise RuntimeError("No resolved loans with a valid origination year were found.")

        takes = {}
        assigned = 0
        for _, row in counts.iterrows():
            y = int(row["issue_year"])
            take = int(round(int(row["cnt"]) * sample_size / total))
            takes[y] = take
            assigned += take
        # Fix rounding drift by adding/removing rows from the largest stratum.
        largest = max(takes, key=lambda k: takes[k])
        takes[largest] += sample_size - assigned
        # Clamp to available counts.
        takes = {y: max(0, min(t, int(counts.loc[counts.issue_year == y, "cnt"].iloc[0]))) for y, t in takes.items()}

        con.execute(
            """
            CREATE OR REPLACE TABLE ranked AS
            SELECT *, row_number() OVER (PARTITION BY issue_year ORDER BY hash(id)) AS rn
            FROM resolved
            """
        )
        condition = " OR ".join(
            f"(issue_year = {y} AND rn <= {t})" for y, t in sorted(takes.items()) if t > 0
        )
        con.execute(
            f"COPY (SELECT * EXCLUDE rn FROM ranked WHERE {condition}) "
            f"TO '{out_model.as_posix()}' (FORMAT PARQUET, COMPRESSION 'zstd')"
        )
    finally:
        con.close()
    return out_model
