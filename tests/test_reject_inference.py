"""Exploratory reject-inference helpers (data-free tests).

The real rejected file is not available in CI, so these tests exercise the pure
pandas logic on small synthetic frames shaped like the two populations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import reject_inference as ri


def _accepted(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dti = rng.uniform(5, 35, n)
    income = rng.lognormal(11, 0.4, n)
    logit = -2.0 + 0.05 * dti - 1e-5 * income
    default = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame(
        {
            "loan_amnt": rng.choice([5000, 10000, 15000], n).astype(float),
            "dti": dti,
            "default_flag": default,
        }
    )


def _rejected(n: int = 500, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amount_requested": rng.choice([3000, 8000, 12000], n).astype(float),
            "dti": rng.uniform(10, 45, n),
            "emp_length": rng.choice(["1 year", "5 years", "10+ years"], n),
            "addr_state": rng.choice(["CA", "TX", "NY"], n),
            "risk_score": rng.uniform(500, 750, n),
            "application_date": "2015-01-01",
        }
    )


def test_common_variable_shift_has_both_variables_and_a_median_shift():
    tab = ri.common_variable_shift(_accepted(), _rejected())
    assert set(tab["variable"]) == {"amount_requested", "dti"}
    assert "shift_median (rejected - accepted)" in tab.columns
    assert (tab["n_accepted"] > 0).all()
    assert (tab["n_rejected"] > 0).all()


def test_common_variable_shift_clips_dti_outliers():
    rej = _rejected()
    rej.loc[:5, "dti"] = 1e6  # junk values that would swamp the mean
    tab = ri.common_variable_shift(_accepted(), rej)
    dti_row = tab.loc[tab["variable"] == "dti"].iloc[0]
    assert dti_row["rejected_mean_clipped"] <= 100.0


def test_extrapolation_gap_is_produced_and_labelled_illustrative():
    gap = ri.extrapolation_gap(_accepted(), _rejected())
    assert "mean_pd_accepted" in gap
    assert "mean_pd_rejected" in gap
    assert "illustrative_gap" in gap
    assert "Illustrative only" in gap["note"]


def test_extrapolation_gap_reports_insufficient_overlap():
    empty = pd.DataFrame({"amount_requested": [], "dti": []})
    gap = ri.extrapolation_gap(_accepted(), empty)
    assert "insufficient common-variable overlap" in gap.get("note", "")


def test_higher_risk_rejected_population_shows_a_positive_gap():
    """A rejected population with worse DTI should be predicted riskier."""
    acc = _accepted()
    rej = _rejected()
    rej["dti"] = rej["dti"] + 30.0  # materially worse than the accepted flow
    gap = ri.extrapolation_gap(acc, rej)
    assert gap["illustrative_gap"] > 0
