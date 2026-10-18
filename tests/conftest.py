"""Shared pytest fixtures.

The unit-test suite is deliberately independent of the raw Lending Club data:
every test runs on small synthetic frames so that CI never needs the dataset.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

# Make ``import src` work without installing the package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _months(start_year: int, n: int) -> list[str]:
    out = []
    y, m = 2012, 1
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return out


@pytest.fixture(scope="session")
def synthetic_loans() -> "pd.DataFrame":
    """A small, fully synthetic Lending-Club-shaped frame."""
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 6000

    annual_inc = np.clip(rng.lognormal(10.9, 0.5, n), 12_000, 500_000)
    loan_amnt = rng.choice([5_000, 10_000, 15_000, 20_000, 25_000, 35_000], size=n).astype(float)
    dti = np.clip(rng.normal(18, 8, n), 0, 45)
    installment = loan_amnt / rng.choice([36, 60], size=n) * 1.05
    revol_bal = np.clip(rng.normal(15_000, 12_000, n), 0, None)
    rev_hi = np.clip(rng.normal(30_000, 15_000, n), 500, None)
    open_acc = rng.poisson(11, n)
    total_acc = open_acc + rng.poisson(13, n)

    logit = (
        -2.6
        + 0.045 * (dti - 18)
        + 6.0 * (loan_amnt / annual_inc)
        + 0.9 * (revol_bal / rev_hi)
        - 0.10 * (open_acc - 11)
        + 0.35 * (rng.random(n) < 0.5)
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    default_flag = (rng.random(n) < p).astype(int)

    months = _months(2012, n)
    month_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    earliest = [
        f"{month_abbr[int(m) - 1]}-{int(y)}"
        for m, y in zip(rng.integers(1, 13, n), rng.integers(1985, 2015, n))
    ]

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "member_id": np.arange(1, n + 1),
            "loan_amnt": loan_amnt,
            "funded_amnt": loan_amnt,
            "funded_amnt_inv": loan_amnt,
            "term": rng.choice(["36 months", "60 months"], n),
            "int_rate": np.round(np.clip(6 + 12 * p, 5, 30), 2),
            "installment": installment,
            "grade": rng.choice(list("ABCDEFG"), n),
            "sub_grade": rng.choice([f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)], n),
            "emp_title": "engineer",
            "emp_length": rng.choice(["< 1 year", "2 years", "5 years", "10+ years", "missing"], n),
            "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.45, 0.10]),
            "annual_inc": annual_inc,
            "verification_status": rng.choice(["Verified", "Not Verified", "Source Verified"], n),
            "issue_d": months,
            "issue_ym": months,
            "loan_status": np.where(default_flag == 1, "Charged Off", "Fully Paid"),
            "loan_desc": "",
            "purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement", "other"], n),
            "title": "loan",
            "zip_code": "100xx",
            "addr_state": rng.choice(["CA", "NY", "TX", "FL", "IL"], n),
            "dti": dti,
            "delinq_2yrs": rng.poisson(0.3, n),
            "earliest_cr_line": earliest,
            "inq_last_6mths": rng.poisson(0.7, n),
            "mths_since_last_delinq": rng.integers(0, 120, n).astype(float),
            "mths_since_last_record": rng.integers(0, 200, n).astype(float),
            "open_acc": open_acc,
            "pub_rec": rng.poisson(0.2, n),
            "revol_bal": revol_bal,
            "revol_util": np.round(np.clip(revol_bal / rev_hi * 100, 0, 100), 1),
            "total_acc": total_acc,
            "initial_list_status": rng.choice(["w", "f"], n),
            "out_prncp": 0.0,
            "total_pymnt": 0.0,
            "total_rec_prncp": np.where(default_flag == 1, loan_amnt * rng.uniform(0.3, 0.9, n), loan_amnt),
            "total_rec_int": 0.0,
            "total_rec_late_fee": 0.0,
            "recoveries": np.where(default_flag == 1, loan_amnt * rng.uniform(0.0, 0.25, n), 0.0),
            "collection_recovery_fee": 0.0,
            "last_pymnt_d": months,
            "last_pymnt_amnt": 0.0,
            "next_pymnt_d": months,
            "last_credit_pull_d": months,
            "collections_12_mths_ex_med": rng.poisson(0.05, n),
            "mths_since_last_major_derog": rng.integers(0, 200, n).astype(float),
            "policy_code": 1,
            "application_type": rng.choice(["Individual", "Joint App"], n, p=[0.94, 0.06]),
            "tot_coll_amt": np.clip(rng.normal(1500, 4000, n), 0, None),
            "tot_cur_bal": np.clip(rng.normal(120_000, 90_000, n), 0, None),
            "total_rev_hi_lim": rev_hi,
            "default_flag": default_flag,
            "resolved_flag": 1,
            "issue_year": [int(m[:4]) for m in months],
        }
    )
    return df
