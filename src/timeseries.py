"""Vintage trend analysis and forecasting of the monthly default rate.

Steps
-----
1. Aggregate the resolved loans into a monthly default-rate series by
   origination vintage (weighted by the number of loans).
2. Time-series decomposition (trend / seasonal / residual).
3. Stationarity tests: ADF and KPSS.
4. ACF / PACF inspection.
5. SARIMAX fit with a chronological train/test split, Ljung-Box test on the
   residuals, and an out-of-sample forecast with confidence bands.
"""
from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

from .utils import save_fig, save_json, save_table


def monthly_default_series(df: pd.DataFrame) -> pd.Series:
    """Monthly default rate by origination vintage (a proper time series).

    Notes
    -----
    Only loans with a resolved outcome enter the series, so recent vintages are
    mechanically under-represented (immaturity bias). ``run_timeseries``
    truncates the immature tail before modelling.
    """
    d = df.dropna(subset=["issue_ym", "default_flag"]).copy()
    d["issue_ym"] = d["issue_ym"].astype(str)
    s = d.groupby("issue_ym")["default_flag"].mean()
    s.index = pd.to_datetime(s.index, format="%Y-%m")
    return s.sort_index()


def vintage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly default rate plus volumes for the vintage table."""
    d = df.dropna(subset=["issue_ym", "default_flag"]).copy()
    tab = (
        d.groupby("issue_ym")
        .agg(n_loans=("default_flag", "size"), default_rate=("default_flag", "mean"), exposure=("loan_amnt", "sum"))
        .reset_index()
        .sort_values("issue_ym")
    )
    tab["issue_ts"] = pd.to_datetime(tab["issue_ym"], format="%Y-%m")
    tab["default_rate"] = tab["default_rate"].round(4)
    return tab


def decompose_series(series: pd.Series, period: int = 12, name: str = "ts_decomposition.png"):
    """Additive trend/seasonality/residual decomposition plot."""
    dec = seasonal_decompose(series, model="additive", period=period)
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(series.index, series.values)
    axes[0].set_ylabel("Observed")
    axes[1].plot(series.index, dec.trend)
    axes[1].set_ylabel("Trend")
    axes[2].plot(series.index, dec.seasonal)
    axes[2].set_ylabel("Seasonal")
    axes[3].plot(series.index, dec.resid, "o", ms=3)
    axes[3].set_ylabel("Residual")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Monthly default rate by vintage - additive decomposition", fontweight="bold")
    save_fig(fig, name)
    return dec


def stationarity_tests(series: pd.Series) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationarity) tests."""
    rows = []
    adf_stat, adf_p, adf_used, _, adf_crit, _ = adfuller(series.dropna(), autolag="AIC")
    rows.append(
        {
            "test": "ADF",
            "statistic": float(adf_stat),
            "p_value": float(adf_p),
            "crit_5pct": float(adf_crit["5%"]),
            "n_lags": int(adf_used),
            "H0": "unit root (non-stationary)",
            "conclusion": "stationary" if adf_p < 0.05 else "fail to reject unit root",
        }
    )
    kpss_stat, kpss_p, _, kpss_crit = kpss(series.dropna(), regression="c", nlags="auto")
    rows.append(
        {
            "test": "KPSS",
            "statistic": float(kpss_stat),
            "p_value": float(kpss_p),
            "crit_5pct": float(kpss_crit["5%"]),
            "n_lags": np.nan,
            "H0": "trend-stationary",
            "conclusion": "stationary" if kpss_p > 0.05 else "reject stationarity",
        }
    )
    return pd.DataFrame(rows)


def acf_pacf_plot(series: pd.Series, lags: int = 30, name: str = "acf_pacf.png"):
    """ACF and PACF plots of the (possibly differenced) series."""
    s = series.dropna()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_acf(s, lags=lags, ax=axes[0])
    plot_pacf(s, lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title("Autocorrelation (ACF)")
    axes[1].set_title("Partial autocorrelation (PACF)")
    save_fig(fig, name)


def fit_sarimax(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 12),
    train_frac: float = 0.8,
) -> Dict[str, object]:
    """Fit SARIMAX on the first ``train_frac`` of the series and forecast the rest."""
    s = series.dropna().asfreq("MS")
    n_train = int(len(s) * train_frac)
    train, test = s.iloc[:n_train], s.iloc[n_train:]

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    forecast = res.get_forecast(steps=len(test))
    pred_mean = forecast.predicted_mean
    ci = forecast.conf_int(alpha=0.05)

    lb = acorr_ljungbox(res.resid, lags=[10, 20], return_df=True)

    metrics = {
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_rmse": float(np.sqrt(np.mean((train - res.fittedvalues) ** 2))),
        "test_rmse": float(np.sqrt(np.mean((test - pred_mean) ** 2))),
        "test_mae": float(np.mean(np.abs(test - pred_mean))),
        "ljung_box_lag10_pvalue": float(lb.loc[10, "lb_pvalue"]),
        "ljung_box_lag20_pvalue": float(lb.loc[20, "lb_pvalue"]),
    }
    return {
        "result": res,
        "train": train,
        "test": test,
        "forecast": pred_mean,
        "conf_int": ci,
        "metrics": metrics,
        "ljung_box": lb,
    }


def forecast_plot(out: Dict[str, object], name: str = "ts_forecast.png"):
    """Plot the realised series, the fitted values and the out-of-sample forecast."""
    train, test = out["train"], out["test"]
    fc, ci = out["forecast"], out["conf_int"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train.index, train.values, label="Train (actual)", color="steelblue")
    ax.plot(test.index, test.values, label="Test (actual)", color="black")
    ax.plot(fc.index, fc.values, label="SARIMAX forecast", color="crimson", ls="--")
    ax.fill_between(fc.index, ci.iloc[:, 0], ci.iloc[:, 1], color="crimson", alpha=0.2, label="95% CI")
    ax.set_ylabel("Monthly default rate")
    ax.set_title("SARIMAX forecast of the monthly vintage default rate")
    ax.legend(loc="upper left", fontsize=8)
    return save_fig(fig, name)


def censoring_sensitivity(
    series: pd.Series, mature_max_year: int = 2015, train_frac: float = 0.8
) -> Dict[str, object]:
    """Re-estimate the model on vintages old enough to be mostly resolved.

    Right-censoring: with the data cut-off at 2018-12, a 36-month loan issued in
    2017-2018 can only be in the sample if it defaulted or prepaid early. The
    late vintages of the default-rate series are therefore biased towards fast
    defaults. Truncating the series at ``mature_max_year`` removes that bias and
    shows how much the conclusions depend on it.
    """
    mature = series[series.index.year <= int(mature_max_year)].dropna()
    if len(mature) < 36:
        return {"mature_max_year": int(mature_max_year), "n_months_mature": int(len(mature)), "note": "series too short to re-estimate"}
    tests = stationarity_tests(mature)
    out = fit_sarimax(mature, train_frac=train_frac)
    return {
        "mature_max_year": int(mature_max_year),
        "n_months_mature": int(len(mature)),
        "series_mature": mature,
        "stationarity_mature": tests,
        "sarimax_mature": out["metrics"],
        "ljung_box_mature": out["ljung_box"],
    }


def censoring_plot(
    series: pd.Series, mature_max_year: int = 2015, name: str = "ts_censoring_sensitivity.png"
) -> None:
    """Full vintage series with the right-censored window shaded."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series.index, series.values, color="steelblue", label="Monthly default rate (resolved loans)")
    if series.index.max().year > mature_max_year:
        cens_start = pd.Timestamp(f"{int(mature_max_year) + 1}-01-01")
        ax.axvspan(cens_start, series.index.max(), color="crimson", alpha=0.15,
                   label=f"Right-censored window (vintages > {mature_max_year})")
    ax.axvline(pd.Timestamp(f"{int(mature_max_year) + 1}-01-01"), color="crimson", ls="--", lw=1)
    ax.set_ylabel("Monthly default rate")
    ax.set_title("Right-censoring of the vintage series (data cut-off 2018-12)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    return save_fig(fig, name)


def run_timeseries(
    df: pd.DataFrame, train_frac: float = 0.8, mature_max_year: int = 2015
) -> Dict[str, object]:
    """End-to-end vintage/trend pipeline used by the notebook and the report."""
    series_all = monthly_default_series(df)
    # Truncate the immature tail: keep months with at least 200 resolved loans.
    counts = df.assign(m=df["issue_ym"].astype(str)).groupby("m", observed=True).size()
    counts.index = pd.to_datetime(counts.index, format="%Y-%m")
    keep = counts[counts >= 200].index
    series = series_all[series_all.index.isin(keep)]
    save_table(series.rename("default_rate").reset_index(name="month"), "ts_monthly_default_rate.csv")

    dec = decompose_series(series)
    tests = stationarity_tests(series)
    save_table(tests, "ts_stationarity_tests.csv")
    acf_pacf_plot(series)

    out = fit_sarimax(series, train_frac=train_frac)
    save_json(out["metrics"], "ts_sarimax_metrics.json")
    save_table(out["ljung_box"].reset_index(), "ts_ljung_box.csv")
    forecast_plot(out)

    censoring_plot(series, mature_max_year=mature_max_year)
    cens = censoring_sensitivity(series, mature_max_year=mature_max_year, train_frac=train_frac)
    if "sarimax_mature" in cens:
        save_json(
            {
                "mature_max_year": cens["mature_max_year"],
                "n_months_mature": cens["n_months_mature"],
                "full_sample": out["metrics"],
                "mature_vintages_only": cens["sarimax_mature"],
                "stationarity_full": _stationarity_records(tests),
                "stationarity_mature": _stationarity_records(cens["stationarity_mature"]),
            },
            "ts_censoring_sensitivity.json",
        )
        save_table(
            cens["stationarity_mature"], "ts_stationarity_tests_mature.csv"
        )

    # Values used in the README/report tables.
    out["series"] = series
    out["stationarity"] = tests
    out["decomposition"] = dec
    out["censoring"] = cens
    return out


def _stationarity_records(tests: pd.DataFrame) -> list:
    """Stationarity table reduced to JSON-safe scalars."""
    return [
        {
            "test": str(r.test),
            "statistic": round(float(r.statistic), 4),
            "p_value": round(float(r.p_value), 4),
            "conclusion": str(r.conclusion),
        }
        for r in tests.itertuples()
    ]
