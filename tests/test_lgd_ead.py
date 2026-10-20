"""LGD / EAD definitions and expected-loss aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import lgd_ead


def test_lgd_only_defined_for_defaults():
    df = pd.DataFrame(
        {
            "funded_amnt": [10_000.0, 10_000.0, 10_000.0],
            "total_rec_prncp": [4_000.0, 10_000.0, 6_000.0],
            "recoveries": [1_500.0, 0.0, 500.0],
            "default_flag": [1, 0, 1],
        }
    )
    out = lgd_ead.compute_lgd_ead(df)
    assert np.isnan(out.loc[1, "lgd"])
    assert out.loc[0, "ead"] == 6_000.0
    assert abs(out.loc[0, "lgd"] - (1 - 1500 / 6000)) < 1e-12
    assert abs(out.loc[2, "lgd"] - (1 - 500 / 4000)) < 1e-12
    assert out["lgd"].dropna().between(0, 1).all()


def test_lgd_eda_table_groups():
    df = pd.DataFrame(
        {
            "funded_amnt": [10_000.0] * 4,
            "total_rec_prncp": [5_000.0] * 4,
            "recoveries": [1_000.0, 500.0, 0.0, 2_000.0],
            "default_flag": [1, 1, 1, 1],
            "grade": ["A", "A", "B", "B"],
        }
    )
    out = lgd_ead.compute_lgd_ead(df)
    tab = lgd_ead.lgd_eda_table(out)
    assert "portfolio" in set(tab["group"])
    assert {"A", "B"} <= set(tab["group"])
    assert {"n", "mean", "median"} <= set(tab.columns)


def test_expected_loss_is_pd_times_lgd_times_ead():
    df = pd.DataFrame({"grade": ["A", "A", "B"], "loan_amnt": [1000.0] * 3, "funded_amnt": [1000.0] * 3, "default_flag": [0, 1, 0]})
    el = lgd_ead.expected_loss_table(df, pd_hat=[0.1, 0.2, 0.3], lgd_hat=[0.5, 0.4, 0.9])
    assert np.allclose(el["el"].to_numpy(), [0.1 * 0.5 * 1000, 0.2 * 0.4 * 1000, 0.3 * 0.9 * 1000])
    agg = lgd_ead.aggregate_el(el)
    assert agg["portfolio"]["expected_loss"].iloc[0] == el["el"].sum()
    assert len(agg["by_grade"]) == 2


def test_regression_metrics_perfect():
    y = np.array([0.0, 0.5, 1.0])
    m = lgd_ead.regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert abs(m["r2"] - 1.0) < 1e-12


def _two_part_frame(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    dti = rng.uniform(0, 40, n)
    inc = rng.lognormal(11, 0.5, n)
    # Boundary (LGD == 1) more likely at high DTI / low income.
    boundary_logit = -2.0 + 0.08 * dti - 0.00002 * inc
    boundary = rng.binomial(1, 1 / (1 + np.exp(-boundary_logit)))
    interior = np.clip(rng.beta(6, 2, n), 0.01, 0.99)  # severity on (0, 1)
    lgd = np.where(boundary == 1, 1.0, interior)
    x = pd.DataFrame({"dti": dti, "annual_inc": inc, "purpose": rng.choice(["a", "b"], n)})
    return x, pd.Series(lgd), pd.Series(boundary)


def test_lgd_boundary_flags():
    y = pd.Series([0.0, 0.5, 1.0, 1.0])
    flags = lgd_ead.lgd_boundary_flags(y)
    assert flags.tolist() == [0, 0, 1, 1]


def test_two_part_lgd_shapes_and_range():
    X, y, _ = _two_part_frame()
    model = lgd_ead.fit_lgd_two_part(X, y, ["dti", "annual_inc"], ["purpose"], seed=1)
    pred = lgd_ead.predict_lgd_two_part(model, X, ["dti", "annual_inc"], ["purpose"])
    assert pred.shape == (len(X),)
    assert np.all(np.isfinite(pred))
    assert np.all((pred >= 0) & (pred <= 1))
    assert 0.0 < model["boundary_rate"] < 1.0


def test_two_part_captures_boundary_mass():
    """Loans the model expects at the boundary should on average get a higher
    predicted LGD than loans it expects to recover something on."""
    X, y, boundary = _two_part_frame(seed=3)
    model = lgd_ead.fit_lgd_two_part(X, y, ["dti", "annual_inc"], ["purpose"], seed=1)
    pred = lgd_ead.predict_lgd_two_part(model, X, ["dti", "annual_inc"], ["purpose"])
    assert pred[boundary.to_numpy() == 1].mean() > pred[boundary.to_numpy() == 0].mean()


def test_grade_calibration_table_shape():
    df = pd.DataFrame({"grade": ["A", "A", "B", "B", "C"]})
    y = np.array([0.2, 0.3, 0.5, 0.6, 0.9])
    pred = np.array([0.25, 0.28, 0.55, 0.58, 0.8])
    tab = lgd_ead.lgd_by_grade_calibration(df, y, pred)
    assert {"grade", "n", "mean_observed", "mean_predicted", "gap"} <= set(tab.columns)
    assert "portfolio" in set(tab["grade"])
    assert len(tab) == 4  # A, B, C + portfolio
