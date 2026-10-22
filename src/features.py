"""Feature engineering and leakage control.

Only information known **at origination** is used. Every post-origination
field (payments, recoveries, outstanding principal, status dates, ...) is
dropped explicitly so that the model cannot peek into the future.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Leakage / post-origination columns that must never enter the model.
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS: List[str] = [
    "loan_status",          # target source
    "default_flag",
    "resolved_flag",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    # identifiers / free text / non-generalisable columns
    "id",
    "member_id",
    "url",
    "loan_desc",
    "title",
    "zip_code",
    "issue_d",
    "issue_ym",
    "issue_year",           # used for the time split, not as a feature
]

# Lender's own risk assessment -> excluded so the scorecard is built from raw
# borrower / credit-bureau data rather than re-learning LendingClub's model.
LENDER_MODEL_COLUMNS: List[str] = ["grade", "sub_grade", "int_rate"]

NUMERIC_FEATURES: List[str] = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "mths_since_last_major_derog",
    "collections_12_mths_ex_med",
    "credit_history_months",
    "loan_amnt_to_income",
    "installment_to_income",
    "revol_bal_to_income",
    "limit_utilisation",
]

CATEGORICAL_FEATURES: List[str] = [
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "initial_list_status",
    "application_type",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create origination-time derived features and clean raw columns.

    The function is side-effect free and returns a new DataFrame.
    """
    out = df.copy()

    # --- origination date helpers (used only for the chronological split) ---
    if "issue_ym" in out.columns:
        out["issue_date"] = pd.to_datetime(out["issue_ym"] + "-01", format="%Y-%m-%d", errors="coerce")
    elif "issue_d" in out.columns:
        out["issue_d_dt"] = pd.to_datetime(out["issue_d"], format="%b-%Y", errors="coerce")
        out["issue_ym"] = out["orig_issue_ym"] = out["issue_d"]
    # Normalize issue_year
    if "issue_year" not in out.columns and "issue_ym" in out.columns:
        out["issue_year"] = pd.to_datetime(out["issue_ym"]).dt.year

    # earliest credit line -> credit history length in months at origination
    if "earliest_cr_line" in out.columns:
        ecl = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y", errors="coerce")
        if "issue_ym" in out.columns and out["issue_ym"].notna().any():
            issue = pd.to_datetime(out["issue_ym"], format="%Y-%m", errors="coerce")
        else:
            issue = pd.to_datetime(out["issue_d"], format="%b-%Y", errors="coerce")
        out["credit_history_months"] = (issue.dt.year - ecl.dt.year) * 12 + (issue.dt.month - ecl.dt.month)
        out.loc[out["credit_history_months"] < 0, "credit_history_months"] = np.nan

    # --- affordability / leverage ratios ---
    inc = out["annual_inc"].replace(0, np.nan)
    out["loan_amnt_to_income"] = out["loan_amnt"] / inc
    out["installment_to_income"] = out["installment"] * 12.0 / inc
    out["revol_bal_to_income"] = out["revol_bal"] / inc
    out["limit_utilisation"] = out["revol_bal"] / out["total_rev_hi_lim"].replace(0, np.nan)

    # --- term: normalize "36 months" -> "36" (kept as a categorical level) ---
    out["term"] = out["term"].astype(str).str.replace("months", "", regex=False).str.strip()

    # --- ordinal employment length ---
    out["emp_length"] = out["emp_length"].astype(str).str.strip()

    # --- tidy categoricals ---
    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip().fillna("missing")

    return out


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize a numeric series at the given quantiles (outlier control)."""
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lo, hi)


def fit_clip_bounds(X: pd.DataFrame, num_features: List[str], lower: float = 0.01, upper: float = 0.99) -> dict:
    """Learn winsorization bounds on the *training* sample only."""
    bounds = {}
    for col in num_features:
        if col in X.columns:
            s = X[col].astype(float)
            bounds[col] = (float(s.quantile(lower)), float(s.quantile(upper)))
    return bounds


def apply_clip_bounds(X: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Apply previously learned winsorization bounds (no re-fitting -> no leakage)."""
    out = X.copy()
    for col, (lo, hi) in bounds.items():
        if col in out.columns:
            out[col] = out[col].astype(float).clip(lo, hi)
    return out


def build_model_matrix(
    df: pd.DataFrame,
    include_lender_grade: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Return (X, y, numeric_features, categorical_features) ready for modeling.

    Parameters
    ----------
    include_lender_grade:
        When ``True`` the lender's own ``grade``/``sub_grade``/``int_rate`` are
        re-introduced (used only for the "benchmark" sensitivity comparison).
    """
    num = list(NUMERIC_FEATURES)
    cat = list(CATEGORICAL_FEATURES)

    if include_lender_grade:
        num = num + ["int_rate"]
        cat = cat + ["grade"]

    missing = [c for c in num + cat if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected feature columns: {missing}")

    X = df[num + cat].copy()
    for col in num:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in cat:
        X[col] = X[col].astype(str)

    y = df["default_flag"].astype(int)
    return X, y, num, cat
