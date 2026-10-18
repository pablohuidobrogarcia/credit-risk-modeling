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
