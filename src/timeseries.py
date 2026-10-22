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


def _contiguous_monthly(series: pd.Series) -> pd.Series:
    """Return the longest gap-free monthly block of ``series``.

    The vintage series is built with a volume threshold, so a handful of very
    early months (fewer than the minimum number of resolved loans) can leave
    *interior* holes once the series is put on a monthly frequency. A hole is not
    a zero - it is a month with almost no data - and every forecast method here
    (seasonal naive, last-value naive, ETS, SARIMAX) needs an unbroken monthly
    grid to be compared fairly. This keeps only the longest contiguous run and
    drops the leading/trailing incomplete stubs.
    """
    s = pd.Series(series).asfreq("MS")
    if s.empty:
        return s
    # A month is present when it has data (not NaN); a run is contiguous when the
    # month ordinals step by exactly 1. Keep the longest present+contiguous run.
    ordinals = s.index.to_period("M").astype("int64")
    present = s.notna().to_numpy()
    runs: list[list[int]] = []
    current: list[int] = []
    for i in range(len(s)):
        if present[i] and (not current or ordinals[i] - ordinals[i - 1] == 1):
            current.append(i)
        elif present[i]:
            runs.append(current)
            current = [i]
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:  # pragma: no cover - defensive
        return s.dropna()
    best = max(runs, key=len)
    return s.iloc[best]


def monthly_default_series(df: pd.DataFrame) -> pd.Series:
    """Monthly default rate by origination vintage (a proper time series).

    Notes
    -----
    Only loans with a resolved outcome enter the series, so recent vintages are
    mechanically under-represented (immaturity bias). ``run_timeseries``
    truncates the immature tail before modeling.
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
    res = model.fit(disp=False, maxiter=200)

    forecast = res.get_forecast(steps=len(test))
    pred_mean = forecast.predicted_mean
    ci = forecast.conf_int(alpha=0.05)

    resid = pd.Series(np.asarray(res.resid, dtype=float)).dropna()
    lb = acorr_ljungbox(resid.to_numpy(), lags=[10, 20], return_df=True)

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


def fit_ets(series: pd.Series, train_frac: float = 0.8) -> Dict[str, object]:
    """Holt-Winters (ETS) benchmark on the same chronological split.

    Additive trend and seasonality, the standard automatic ETS recipe for a
    monthly series with a seasonal shape. Used purely as a benchmark against
    which SARIMAX is judged on the temporal holdout.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    s = series.dropna().asfreq("MS")
    n_train = int(len(s) * train_frac)
    train, test = s.iloc[:n_train], s.iloc[n_train:]
    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated"
    )
    res = model.fit()
    fc = res.forecast(len(test))
    fc_vals = np.asarray(fc.values, dtype=float)
    # Holt-Winters can fail to converge on a short, noisy series, in which case
    # the forecast comes back as NaN. Report that honestly as NaN rather than
    # dropping the arm, so the benchmark table still shows ETS was attempted.
    if not np.isfinite(fc_vals).all():
        test_rmse = float("nan")
        test_mae = float("nan")
    else:
        test_rmse = float(np.sqrt(np.mean((test.values - fc_vals) ** 2)))
        test_mae = float(np.mean(np.abs(test.values - fc_vals)))
    return {
        "train": train,
        "test": test,
        "forecast": fc,
        "metrics": {
            "method": "ETS(add,add,12)",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_rmse": float(np.sqrt(np.mean((train - res.fittedvalues) ** 2))),
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        },
    }


def _mase(y_train: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, period: int = 12) -> float:
    """Mean absolute scaled error: test MAE divided by the in-sample seasonal-naive MAE."""
    y_train = np.asarray(y_train, dtype=float)
    scale = np.mean(np.abs(y_train[period:] - y_train[:-period])) if len(y_train) > period else np.mean(np.abs(np.diff(y_train)))
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y_test) - np.asarray(y_pred))) / scale)


def benchmark_forecasts(series: pd.Series, train_frac: float = 0.8) -> pd.DataFrame:
    """Temporal-holdout comparison: SARIMAX vs seasonal naive vs ETS.

    All three are fitted on the *same* chronological split (first ``train_frac``
    of the mature series) and scored on the same holdout, so the RMSE and MASE
    columns are directly comparable. The honest question - does the ARIMA
    machinery earn its keep over a naive seasonal benchmark? - is answered here.
    """
    s = series.dropna().asfreq("MS")
    n_train = int(len(s) * train_frac)
    train, test = s.iloc[:n_train], s.iloc[n_train:]

    rows: list[Dict[str, object]] = []

    # 1. Seasonal naive: y_hat(t) = y(t - 12).
    if len(train) >= 12:
        naive_vals = [train.iloc[len(train) - 12 + (i % 12)] for i in range(len(test))]
    else:  # pragma: no cover - defensive
        naive_vals = [float(train.iloc[-1])] * len(test)
    naive = np.asarray(naive_vals, dtype=float)
    # 2. Random-walk / last-value naive (the non-seasonal benchmark).
    lastval = np.full(len(test), float(train.iloc[-1]))
    # 3. SARIMAX.
    sar = fit_sarimax(series, train_frac=train_frac)
    sar_fc = np.asarray(sar["forecast"].values, dtype=float)
    # 4. ETS.
    ets = fit_ets(series, train_frac=train_frac)
    ets_fc = np.asarray(ets["forecast"].values, dtype=float)

    for name, pred in (
        ("seasonal_naive", naive),
        ("last_value_naive", lastval),
        ("ets", ets_fc),
        ("sarimax", sar_fc),
    ):
        if not np.isfinite(pred).all():
            # ETS can fail to converge on a short series; keep the row so the
            # benchmark shows the attempt, but leave its scores as NaN.
            rows.append(
                {"method": name, "n_train": int(len(train)), "n_test": int(len(test)),
                 "rmse": float("nan"), "mae": float("nan"), "mase": float("nan")}
            )
            continue
        rmse = float(np.sqrt(np.mean((test.values - pred) ** 2)))
        mae = float(np.mean(np.abs(test.values - pred)))
        rows.append(
            {
                "method": name,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "rmse": round(rmse, 6),
                "mae": round(mae, 6),
                "mase": round(_mase(train.values, test.values, pred), 4),
            }
        )
    table = pd.DataFrame(rows).sort_values("rmse", na_position="last").reset_index(drop=True)
    return table


def forecast_benchmark_plot(table: pd.DataFrame, name: str = "ts_forecast_benchmark.png") -> None:
    """Bar chart of holdout RMSE for SARIMAX and the naive/ETS benchmarks."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["crimson" if m == "sarimax" else "steelblue" for m in table["method"]]
    ax.bar(table["method"], table["rmse"], color=colors)
    ax.set_ylabel("Holdout RMSE (monthly default rate)")
    ax.set_title("Forecast benchmark: SARIMAX vs seasonal naive and ETS")
    for i, v in enumerate(table["rmse"]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    return save_fig(fig, name)


def run_timeseries(
    df: pd.DataFrame, train_frac: float = 0.8, mature_max_year: int = 2015
) -> Dict[str, object]:
    """End-to-end vintage/trend pipeline used by the notebook and the report.

    The **primary** series is restricted to the mature vintages
    (``issue_year <= mature_max_year``): only those loans are old enough for
    their outcome to be observed before the 2018-12 cut-off, so the resulting
    default rates are not biased towards fast defaults. The full-sample series
    (including the right-censored tail) is estimated in parallel as the
    secondary, censoring-exposed analysis.
    """
    series_all = monthly_default_series(df)
    # Truncate the immature tail: keep months with at least 200 resolved loans.
    counts = df.assign(m=df["issue_ym"].astype(str)).groupby("m", observed=True).size()
    counts.index = pd.to_datetime(counts.index, format="%Y-%m")
    keep = counts[counts >= 200].index
    series_full = series_all[series_all.index.isin(keep)]
    save_table(series_full.rename("default_rate").reset_index(name="month"), "ts_monthly_default_rate_full.csv")

    # Primary series = mature vintages only, made gap-free so every forecast arm
    # (seasonal naive / last-value / ETS / SARIMAX) is compared on the same
    # unbroken monthly grid. Interior holes - months below the volume threshold -
    # are dropped rather than treated as zeros.
    series = _contiguous_monthly(series_full[series_full.index.year <= int(mature_max_year)])
    save_table(series.rename("default_rate").reset_index(name="month"), "ts_monthly_default_rate.csv")

    dec = decompose_series(series)
    tests = stationarity_tests(series)
    save_table(tests, "ts_stationarity_tests.csv")
    acf_pacf_plot(series)

    out = fit_sarimax(series, train_frac=train_frac)
    save_json(out["metrics"], "ts_sarimax_metrics.json")
    save_table(out["ljung_box"].reset_index(), "ts_ljung_box.csv")
    forecast_plot(out)

    # Benchmark comparison (block 5): does SARIMAX beat seasonal naive / ETS?
    bench = benchmark_forecasts(series, train_frac=train_frac)
    save_table(bench, "ts_forecast_benchmark.csv")
    forecast_benchmark_plot(bench)
    out["forecast_benchmark"] = bench

    # Secondary: the same model on the full series (immature tail included), plus
    # the resolved-share diagnostic that motivates the choice.
    censoring_plot(series_full, mature_max_year=mature_max_year)
    full_out = fit_sarimax(series_full, train_frac=train_frac)
    full_tests = stationarity_tests(series_full)
    cens: Dict[str, object] = {
        "mature_max_year": int(mature_max_year),
        "n_months_mature": int(len(series)),
        "n_months_full": int(len(series_full)),
        "sarimax_full_sample": full_out["metrics"],
        "stationarity_full": full_tests,
        "sarimax_mature": out["metrics"],
        "stationarity_mature": tests,
    }
    save_json(
        {
            "mature_max_year": cens["mature_max_year"],
            "n_months_mature": cens["n_months_mature"],
            "n_months_full": cens["n_months_full"],
            "mature_vintages_only": cens["sarimax_mature"],
            "full_sample": cens["sarimax_full_sample"],
            "stationarity_mature": _stationarity_records(tests),
            "stationarity_full": _stationarity_records(full_tests),
        },
        "ts_censoring_sensitivity.json",
    )
    save_table(full_tests, "ts_stationarity_tests_full.csv")

    # Values used in the README/report tables.
    out["series"] = series
    out["series_full"] = series_full
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
