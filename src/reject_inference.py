"""Exploratory reject-inference analysis (clearly labeled as such).

LendingClub publishes the *rejected* applications alongside the accepted ones.
Because only accepted loans have a known outcome, the PD model is trained on a
sample selected by the lender's own approval decision - the classic
"reject inference" problem. This module does **not** claim to fix it; it builds
the evidence that the problem is real:

1. **Common-variable comparison.** The rejected file only shares a handful of
   columns with the accepted file (requested amount, DTI, employment length,
   state, and a lender risk score that post-dates 2012). Those common variables
   are compared between the two populations; a large shift means the accepted
   sample is not representative of the applicant flow.
2. **Extrapolation gap (illustrative).** A simple logistic model on the common
   variables is fitted on the accepted loans and scored on a sample of rejected
   applications. The difference in mean predicted PD between the two populations
   is an *illustrative* lower bound on the selection bias - it is not a
   calibrated reject-inference correction and is not used anywhere else in the
   pipeline.

Everything here is exploratory and clearly labeled; no headline number depends
on it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import duckdb
import numpy as np
import pandas as pd

from .config import Config


def _rejected_glob(cfg: Config) -> Optional[Path]:
    rej = cfg.path("rejected_csv")
    if rej.exists():
        return rej
    raw = cfg.path("raw_dir")
    for p in raw.glob("*"):
        if "rejected" in p.name.lower():
            return p
    return None


def rejected_sample(n: int = 200_000, seed: int = 42) -> pd.DataFrame:
    """Deterministic sample of the rejected applications (common variables only)."""
    cfg = Config.load()
    path = _rejected_glob(cfg)
    if path is None:
        raise FileNotFoundError("Rejected-applications CSV not found under data/raw")
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    try:
        df = con.execute(
            f"""
            SELECT "Amount Requested"                AS amount_requested,
                   TRY_CAST(REPLACE("Debt-To-Income Ratio", '%', '') AS DOUBLE) AS dti,
                   "Employment Length"               AS emp_length,
                   "State"                           AS addr_state,
                   TRY_CAST("Risk_Score" AS DOUBLE)  AS risk_score,
                   "Application Date"                AS application_date
            FROM read_csv_auto('{path.as_posix()}', all_varchar=TRUE, ignore_errors=TRUE)
            USING SAMPLE {int(n)} ROWS (reservoir, {int(seed)})
            """
        ).df()
    finally:
        con.close()
    return df


def _accepted_common(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the accepted modeling frame to the variables shared with the rejected file."""
    out = pd.DataFrame(
        {
            "amount_requested": pd.to_numeric(df.get("loan_amnt"), errors="coerce"),
            "dti": pd.to_numeric(df.get("dti"), errors="coerce"),
            "emp_length": df.get("emp_length"),
            "addr_state": df.get("addr_state"),
            "default_flag": pd.to_numeric(df.get("default_flag"), errors="coerce"),
        }
    )
    return out


def common_variable_shift(accepted: pd.DataFrame, rejected: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side summary of the common variables in both populations.

    DTI is reported both raw and clipped to the plausible ``[0, 100]`` band used
    for the accepted population: the rejected file contains a handful of junk
    values that would otherwise dominate the mean, so the median is the robust
    headline and the clipped mean is shown alongside it.
    """
    acc = _accepted_common(accepted)
    rej = rejected
    rows = []
    for col in ("amount_requested", "dti"):
        a = pd.to_numeric(acc[col], errors="coerce").dropna()
        r = pd.to_numeric(rej[col], errors="coerce").dropna()
        if len(a) == 0 or len(r) == 0:
            continue
        r_clip = r.clip(lower=0, upper=100) if col == "dti" else r
        rows.append(
            {
                "variable": col,
                "accepted_mean": round(float(a.mean()), 4),
                "rejected_mean_clipped": round(float(r_clip.mean()), 4),
                "accepted_median": round(float(a.median()), 4),
                "rejected_median": round(float(r.median()), 4),
                "shift_median (rejected - accepted)": round(float(r.median() - a.median()), 4),
                "n_accepted": int(len(a)),
                "n_rejected": int(len(r)),
            }
        )
    return pd.DataFrame(rows)


def extrapolation_gap(accepted: pd.DataFrame, rejected: pd.DataFrame, seed: int = 42) -> Dict[str, float]:
    """Illustrative PD gap between accepted and rejected applicants.

    Fits a logistic model for ``default_flag`` on the common variables using the
    accepted loans, then scores a sample of the rejected applications. The mean
    predicted PD on the rejected flow minus that on the accepted flow is an
    *illustrative* selection-bias gap (the rejected applicants never had an
    observed outcome, so this cannot be validated).
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    acc = _accepted_common(accepted).dropna(subset=["default_flag"])
    rej = rejected.copy()

    def _design(d: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "amount_requested": pd.to_numeric(d["amount_requested"], errors="coerce"),
                "dti": pd.to_numeric(d["dti"], errors="coerce"),
            }
        )

    Xa, ya = _design(acc), acc["default_flag"].astype(int)
    Xr = _design(rej)
    if Xa.dropna().empty or Xr.dropna().empty:
        return {"note": "insufficient common-variable overlap"}

    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        ]
    )
    pipe.fit(Xa, ya)
    pd_acc = float(np.mean(pipe.predict_proba(Xa)[:, 1]))
    pd_rej = float(np.mean(pipe.predict_proba(Xr)[:, 1]))
    return {
        "mean_pd_accepted": round(pd_acc, 4),
        "mean_pd_rejected": round(pd_rej, 4),
        "illustrative_gap": round(pd_rej - pd_acc, 4),
        "ratio_rejected_over_accepted": round(pd_rej / pd_acc, 3) if pd_acc else None,
        "note": (
            "Illustrative only: the rejected applicants have no observed outcome, so the gap "
            "cannot be validated. Common variables are requested amount and DTI."
        ),
    }


def run_reject_inference(df: pd.DataFrame, n_rejected: int = 200_000, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """End-to-end exploratory reject-inference summary (labeled exploratory)."""
    rej = rejected_sample(n=n_rejected, seed=seed)
    shift = common_variable_shift(df, rej)
    gap = extrapolation_gap(df, rej, seed=seed)
    return {"common_variable_shift": shift, "extrapolation_gap": gap, "rejected_sample": rej}
