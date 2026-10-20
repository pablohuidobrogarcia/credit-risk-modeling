"""Time-series module: series construction, stationarity tests, SARIMAX and the
block-5 forecast benchmarks (seasonal naive, last-value naive, ETS)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import timeseries as ts


def _monthly_frame(n_months: int = 72, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.period_range("2010-01", periods=n_months, freq="M")
    rows = []
    for i, m in enumerate(months):
        n = 200
        # A mild trend plus a seasonal component so the series is non-trivial.
        rate = 0.12 + 0.0015 * i + 0.02 * np.sin(2 * np.pi * i / 12)
        defaults = rng.binomial(1, np.clip(rate, 0.01, 0.6), n)
        for d in defaults:
            rows.append(
                {
                    "issue_d": m.strftime("%Y-%m"),
                    "issue_ym": m.strftime("%Y-%m"),
                    "issue_year": m.year,
                    "default_flag": int(d),
                    "resolved_flag": 1,
                }
            )
    return pd.DataFrame(rows)


def test_monthly_default_series_is_monthly_and_bounded():
    df = _monthly_frame()
    s = ts.monthly_default_series(df)
    assert isinstance(s, pd.Series)
    assert len(s) > 0
    assert s.between(0.0, 1.0).all()


def test_stationarity_tests_return_adf_and_kpss_rows():
    s = ts.monthly_default_series(_monthly_frame())
    out = ts.stationarity_tests(s)
    assert len(out) >= 2
    assert {"test", "statistic", "p_value"} <= set(out.columns)
    assert {"ADF", "KPSS"} <= set(out["test"])


def test_mase_matches_the_definition():
    rng = np.random.default_rng(0)
    y_train = 0.10 + 0.02 * rng.random(24)
    y_test = 0.11 + 0.01 * rng.random(6)
    pred = np.full(6, 0.12)
    scale = np.mean(np.abs(y_train[12:] - y_train[:-12]))
    expected = np.mean(np.abs(y_test - pred)) / scale
    assert abs(ts._mase(y_train, y_test, pred, period=12) - expected) < 1e-12


def test_mase_is_nan_when_the_scale_is_degenerate():
    flat = np.full(24, 0.10)
    assert np.isnan(ts._mase(flat, np.array([0.1, 0.2]), np.array([0.1, 0.2]), period=12))


def test_benchmark_forecasts_returns_all_methods_on_the_same_split():
    s = ts.monthly_default_series(_monthly_frame(n_months=60))
    table = ts.benchmark_forecasts(s, train_frac=0.8)
    assert set(table["method"]) == {"seasonal_naive", "last_value_naive", "ets", "sarimax"}
    # Every method must be scored on the identical holdout length.
    assert table["n_test"].nunique() == 1
    finite = table["rmse"].dropna()
    assert (finite >= 0).all()
    assert finite.is_monotonic_increasing


def test_benchmark_table_is_sorted_by_rmse():
    s = ts.monthly_default_series(_monthly_frame(n_months=60, seed=2))
    table = ts.benchmark_forecasts(s, train_frac=0.75)
    finite = table[table["rmse"].notna()]
    assert finite["rmse"].is_monotonic_increasing
    assert table["method"].iloc[0] in set(table["method"])


def test_contiguous_monthly_drops_interior_holes():
    """The vintage series can have interior holes (months below the volume
    threshold). Every forecast arm needs an unbroken monthly grid, so the helper
    must return the longest gap-free block with no NaN."""
    idx = pd.to_datetime(
        ["2008-01", "2008-02", "2008-03", "2008-05", "2008-06", "2008-07", "2008-08"]
    )
    s = pd.Series([0.1, 0.11, 0.12, 0.14, 0.13, 0.12, 0.11], index=idx)
    # Force a hole at 2008-04 (a month with almost no data).
    s = s.asfreq("MS")
    out = ts._contiguous_monthly(s)
    assert not out.isna().any()
    assert len(out) == 4  # 2008-05 .. 2008-08 is the longest run
    assert out.index.min() == pd.Timestamp("2008-05-01")


def test_benchmark_forecasts_reports_mase_for_every_finite_arm():
    s = ts.monthly_default_series(_monthly_frame(n_months=72))
    table = ts.benchmark_forecasts(s, train_frac=0.8)
    for _, row in table.iterrows():
        if pd.notna(row["rmse"]):
            assert pd.notna(row["mase"]), row["method"]
