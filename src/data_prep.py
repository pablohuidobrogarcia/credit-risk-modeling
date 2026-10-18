from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import duckdb

from .config import Config
from .utils import set_seeds


def detect_raw_files(root_dir: Path | None = None) -> Tuple[Path | None, Path | None]:
    """Locate the accepted and rejected Lending Club CSV files in ``data/raw``.

    Detection is keyword based (``accepted``/``rejected`` in the file name) so
    that slight differences in capitalisation or suffixes still work.
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
