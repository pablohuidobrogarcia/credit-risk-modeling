"""End-to-end pipeline: data -> quality -> EDA -> PD -> LGD/EAD -> clustering -> simulation.

Usage
-----
    python -m src.pipeline all
    python -m src.pipeline data quality sql eda pd

Every stage writes small artifacts (CSV/JSON) under ``reports/results`` and
figures under ``reports/figures``. The final ``reports/results/summary.json``
collects every headline number quoted in the README and the model validation
report, so those documents never state an un-computed figure.
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from . import (
    clustering,
    data_prep,
    features,
    interpret,
    lgd_ead,
    pd_model,
    quality,
    scorecard,
    selection,
    sql_runner,
    timeseries,
    validation,
)
from . import metrics as M
from . import simulation as sim
from .config import Config
from .utils import ensure_dirs, save_fig, save_json, save_table, set_seeds
from .woe_iv import fit_encoders, iv_summary, transform_frame

STAGES = ["data", "quality", "sql", "eda", "pd", "lgd", "clustering", "simulation"]

# Borrower / bureau variables used for the behavioural segmentation. They are
# all known at origination, so a segment can be assigned to a new applicant.
CLUSTER_FEATURES: List[str] = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "revol_util",
    "open_acc",
    "total_acc",
    "credit_history_months",
    "installment_to_income",
    "loan_amnt_to_income",
    "tot_cur_bal",
    "revol_bal",
    "total_rev_hi_lim",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """Recursively replace NaN/inf and numpy scalars so JSON is always valid."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _records(obj)
    if isinstance(obj, pd.Series):
        return _jsonable(obj.tolist())
    return obj


def _records(df: pd.DataFrame, ndigits: int = 4) -> List[Dict[str, Any]]:
    """DataFrame -> list of JSON-safe records (NaN becomes null)."""
    if df is None or df.empty:
        return []
    return _jsonable(df.round(ndigits).to_dict(orient="records"))


def load_modeling_frame(cfg: Config) -> pd.DataFrame:
    """Load the stratified modeling sample and engineer origination-time features."""
    parquet = cfg.path("modeling_parquet")
    if not parquet.exists():
        data_prep.build_modeling_sample(force=False)
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    df = con.execute(f"SELECT * FROM '{parquet.as_posix()}'").df()
    con.close()
    return features.engineer_features(df)


def train_oot_split(
    df: pd.DataFrame,
    cfg: Config,
    kind: str = "mature",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: fit on early vintages, validate on later ones.

    ``kind="mature"`` (primary) uses only the fully-resolved ``(year, term)``
    cells configured under ``modeling.mature_*``; ``kind="immature"``
    (secondary) reproduces the original late-vintage window. See
    :func:`split_design` for the rule and the resolved-share evidence.
    """
    if kind == "immature":
        split_year = int(cfg.modeling.get("oot_split_year", 2017))
        train = df[df["issue_year"] < split_year].copy()
        oot = df[df["issue_year"] >= split_year].copy()
        return train, oot

    train_years = {int(y) for y in cfg.modeling.get("mature_train_years", [2010, 2011, 2012, 2013])}
    test_years = {int(y) for y in cfg.modeling.get("mature_test_years", [2014, 2015])}
    terms = [str(t) for t in cfg.modeling.get("mature_terms", ["36"])]

    def _restrict(frame: pd.DataFrame, years: set) -> pd.DataFrame:
        m = frame["issue_year"].isin(years)
        if terms and "term" in frame.columns:
            m &= frame["term"].astype(str).isin(terms)
        return frame[m].copy()

    return _restrict(df, train_years), _restrict(df, test_years)


def split_design(cfg: Config) -> Dict[str, Any]:
    """Document the primary/secondary validation split from the resolved shares.

    Combines the configured years/terms with the empirical ``(year, term)``
    resolved-share table so every claim about "maturity" is backed by a number.
    """
    table = data_prep.resolution_by_year_term()
    rule = data_prep.mature_split_rule(
        table, min_resolved_share=float(cfg.modeling.get("min_resolved_share", 0.90))
    )
    train_years = [int(y) for y in cfg.modeling.get("mature_train_years", [2010, 2011, 2012, 2013])]
    test_years = [int(y) for y in cfg.modeling.get("mature_test_years", [2014, 2015])]
    terms = [str(t) for t in cfg.modeling.get("mature_terms", ["36"])]
    immature_year = int(cfg.modeling.get("oot_split_year", 2017))

    def _shares(years, terms_):
        m = table["issue_year"].isin(years) & table["term_months"].astype(str).isin(terms_)
        sub = table[m]
        if sub.empty:
            return None
        n = float(sub["n_loans"].sum())
        r = float(sub["n_resolved"].sum())
        return {
            "resolved_share": round(r / n, 4) if n else None,
            "n_loans": int(n),
            "n_resolved": int(r),
        }

    return {
        "primary": {
            "label": "mature",
            "train_years": train_years,
            "test_years": test_years,
            "terms": terms,
            "rule": rule["rule"],
            "min_resolved_share": rule["min_resolved_share"],
            "train_resolution": _shares(train_years, terms),
            "test_resolution": _shares(test_years, terms),
        },
        "secondary": {
            "label": "immature",
            "train_years_before": immature_year,
            "test_years": [immature_year, int(cfg.modeling.get("test_years", [2017, 2018])[-1])],
            "rule": f"Issue year >= {immature_year} (the original split, heavily right-censored).",
            "test_resolution_all_terms": _shares([immature_year, immature_year + 1], ["36", "60"]),
        },
        "resolved_share_by_year_term": _records(table),
        "mature_cells": rule["mature_cells"],
        "immature_cells": rule["immature_cells"],
        "terms_with_immature_cells": rule["terms_with_immature_cells"],
    }


# ---------------------------------------------------------------------------
# Stage 1: data and quality
# ---------------------------------------------------------------------------
def stage_data() -> None:
    """Build the parquet extract and the stratified modeling sample."""
    print("[data] building parquet extract and modeling sample ...")
    data_prep.build_accepted_parquet(force=False)
    data_prep.build_modeling_sample(force=False)


def stage_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Missingness, outlier and leakage-control reports."""
    print("[quality] data-quality and leakage reports ...")
    quality.build_reports(df)
    return quality.quality_summary(df)


def stage_sql() -> Dict[str, pd.DataFrame]:
    """Execute the documented HiveQL-style query suite with DuckDB."""
    print("[sql] executing the documented HiveQL-style queries ...")
    return sql_runner.run_all_sql()


# ---------------------------------------------------------------------------
# Stage 2: EDA and time-series modelling
# ---------------------------------------------------------------------------
def stage_eda(df: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """Vintage analysis, censoring diagnostics and SARIMAX forecast."""
    print("[eda] vintage analysis, censoring diagnostic and SARIMAX forecast ...")
    save_table(timeseries.vintage_table(df), "vintage_table.csv")

    # Right-censoring diagnostic: the sample only holds resolved loans, so late
    # vintages are systematically biased towards fast defaults.
    maturity = data_prep.resolution_by_year()
    save_table(maturity, "censoring_resolved_share_by_year.csv")
    _resolved_share_plot(maturity)

    mature_max_year = int(cfg.modeling.get("mature_max_year", 2015))
    out = timeseries.run_timeseries(df, mature_max_year=mature_max_year)
    metrics = _jsonable(out["metrics"])
    lb = out["ljung_box"].reset_index()

    result: Dict[str, Any] = {
        "n_months": int(len(out["series"])),
        "series_first": str(out["series"].index.min().date()),
        "series_last": str(out["series"].index.max().date()),
        "mean_default_rate": round(float(out["series"].mean()), 4),
        "min_default_rate": round(float(out["series"].min()), 4),
        "max_default_rate": round(float(out["series"].max()), 4),
        "stationarity": _records(out["stationarity"]),
        "sarimax": metrics,
        "sarimax_basis": f"mature vintages only (issue_year <= {mature_max_year})",
        "ljung_box": _records(lb),
        "resolution_by_year": _records(maturity),
        "resolution_by_year_term": _records(data_prep.resolution_by_year_term()),
    }
    cens = out.get("censoring", {})
    if "sarimax_full_sample" in cens:
        result["censoring_sensitivity"] = {
            "mature_max_year": cens["mature_max_year"],
            "n_months_mature": int(cens["n_months_mature"]),
            "n_months_full": int(cens.get("n_months_full", len(out.get("series_full", [])))),
            "sarimax_mature_vintages": _jsonable(cens["sarimax_mature"]),
            "sarimax_full_sample": _jsonable(cens["sarimax_full_sample"]),
            "stationarity_mature_vintages": _records(cens["stationarity_mature"]),
            "stationarity_full_sample": _records(cens["stationarity_full"]),
        }
    return result


def _resolved_share_plot(maturity: pd.DataFrame, name: str = "censoring_resolved_share.png") -> None:
    """Resolved share and vintage default rate, showing why late years are biased."""
    fig, ax = plt.subplots(figsize=(9, 5))
    years = maturity["issue_year"].astype(int)
    ax.bar(years, maturity["resolved_share"] * 100.0, color="steelblue", label="Share of vintage resolved")
    ax.set_xlabel("Origination year")
    ax.set_ylabel("Resolved share (% of the vintage)", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax.twinx()
    ax2.plot(years, maturity["default_rate_resolved"] * 100.0, "o--", color="crimson",
             label="Default rate among resolved")
    ax2.set_ylabel("Default rate among resolved (%)", color="crimson")
    ax2.grid(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")
    ax.set_title("Right-censoring: only resolved loans enter the sample")
    save_fig(fig, name)


# ---------------------------------------------------------------------------
# Stage 2b: PD scorecard and challengers
# ---------------------------------------------------------------------------
def _ks_optimal_cutoff(y: pd.Series, p: np.ndarray) -> float:
    """Threshold maximising the KS statistic (selected on the training sample)."""
    fpr, tpr, thr = roc_curve(np.asarray(y), np.asarray(p))
    best = int(np.argmax(tpr - fpr))
    return float(thr[min(best, len(thr) - 1)])


def _score_histogram(score_train: np.ndarray, score_oot: np.ndarray, name: str = "scorecard_distribution.png") -> None:
    """Points-score distribution on the train window vs out of time."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(score_train, bins=60, alpha=0.55, label="Train score", color="steelblue", density=True)
    ax.hist(score_oot, bins=60, alpha=0.55, label="Out-of-time score", color="orange", density=True)
    ax.set_xlabel("Scorecard points (higher = lower risk)")
    ax.set_ylabel("Density")
    ax.set_title("Scorecard distribution: training vs out-of-time")
    ax.legend()
    save_fig(fig, name)


def _fit_all_pd_models(
    train: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    num_feats: List[str],
    cat_feats: List[str],
    encoders: Dict[str, Any],
    cfg: Config,
) -> Tuple[List[pd_model.FittedModel], pd.DataFrame]:
    """Fit the scorecard logit/probit, the regularised logits and the tree challengers."""
    models: List[pd_model.FittedModel] = []

    logit, coef_table = pd_model.fit_statsmodels_woe("logit", "logit", X, y, encoders)
    save_table(coef_table, "pd_logit_coefficients.csv")
    models.append(logit)
    print(f"    logit    pseudo-R2={logit.params['pseudo_r2']:.4f}")

    probit, coef_probit = pd_model.fit_statsmodels_woe("probit", "probit", X, y, encoders)
    save_table(coef_probit, "pd_probit_coefficients.csv")
    models.append(probit)
    print(f"    probit   pseudo-R2={probit.params['pseudo_r2']:.4f}")

    for penalty in ("l2", "l1"):
        reg, coef_reg, grid = pd_model.fit_regularised_logit(
            X, y, encoders, penalty=penalty, seed=cfg.seed, cv_sample_size=150_000
        )
        save_table(coef_reg, f"pd_logit_{penalty}_coefficients.csv")
        save_table(grid, f"pd_logit_{penalty}_cv_grid.csv")
        models.append(reg)
        print(f"    logit_{penalty}: C={reg.params['C']:.4g}  cv_auc={reg.cv_auc:.4f}")

    cv_sub = train.sample(n=min(100_000, len(train)), random_state=cfg.seed)
    for name in ("decision_tree", "random_forest", "lightgbm"):
        columns = num_feats + cat_feats
        cv_auc, cv_std = pd_model.cross_validate_pipeline(
            pd_model.make_tree_pipeline(name, num_feats, cat_feats, cfg.seed),
            cv_sub[columns],
            cv_sub["default_flag"],
            seed=cfg.seed,
        )
        model = pd_model.fit_tree_model(
            name, X[columns], y, num_feats, cat_feats, cfg.seed, run_cv=False
        )
        model.cv_auc, model.cv_auc_std = cv_auc, cv_std
        models.append(model)
        print(f"    {name}: cv_auc={cv_auc:.4f}")
    return models, coef_table


def stage_pd(df: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """PD scorecard on the primary (mature) window plus the secondary comparison.

    The primary result drives every downstream stage (LGD/EAD, EL, Monte Carlo)
    and the headline metrics. The secondary (immature) window is re-run with the
    identical code path so the two are directly comparable; only the comparison
    table and a couple of headline scalars are kept for the summary.
    """
    print("[pd] binning, variable selection, PD models and OOT validation ...")
    design = split_design(cfg)
    save_json(_jsonable(design), "pd_split_design.json")

    primary = _run_pd_window(df, cfg, kind="mature", tag="")
    print(f"    primary (mature) champion: {primary['champion']}  AUC={primary['champion_auc']:.4f}")

    secondary = _run_pd_window(df, cfg, kind="immature", tag="secondary")
    print(f"    secondary (immature) champion: {secondary['champion']}  AUC={secondary['champion_auc']:.4f}")

    comparison = _window_comparison(primary, secondary)
    save_table(comparison, "pd_window_comparison.csv")
    print(comparison.to_string(index=False))

    primary["split_design"] = design
    primary["window_comparison"] = comparison
    primary["secondary_summary"] = _secondary_summary(secondary)
    return primary


def _run_pd_window(
    df: pd.DataFrame, cfg: Config, kind: str, tag: str
) -> Dict[str, Any]:
    """Fit and validate the full PD stack on one validation window.

    ``tag`` is a filename suffix (``""`` for the primary window, ``"_secondary"``
    for the immature one) so the two runs never overwrite each other's artifacts.
    Every fit (winsorisation bounds, WoE bins, variable selection, the
    calibrator, the KS cut-off) uses the training window only.
    """
    train, oot = train_oot_split(df, cfg, kind=kind)

    X_all, y_all, num_all, cat_all = features.build_model_matrix(train)
    X_oot_all, y_oot, _, _ = features.build_model_matrix(oot)

    # Winsorisation bounds learned on the training window only (no leakage).
    bounds = features.fit_clip_bounds(X_all, num_all)
    X_all = features.apply_clip_bounds(X_all, bounds)
    X_oot_all = features.apply_clip_bounds(X_oot_all, bounds)

    encoders = fit_encoders(X_all, y_all, num_all, cat_all, n_bins=20, min_bin_pct=0.05, max_bins=8)
    W = transform_frame(X_all, encoders)
    iv_tab = iv_summary(encoders)
    save_table(iv_tab, f"pd_iv_summary{tag}.csv")

    sel = selection.select_variables(encoders, W, iv_min=0.02, corr_max=0.7, vif_max=5.0)
    selected = list(sel["selected"])
    save_table(sel["iv_table"], f"pd_variable_selection{tag}.csv")
    save_table(sel["correlation_dropped"], f"pd_correlation_dropped{tag}.csv")
    save_table(sel["vif_final"], f"pd_vif_final{tag}.csv")
    save_json({"selected": selected, "iv_pass": sel["iv_pass"]}, f"pd_selected_variables{tag}.json")
    print(f"    [{kind}] binned {len(encoders)} variables, kept {len(selected)} after IV/corr/VIF")

    sel_num = [c for c in num_all if c in selected]
    sel_cat = [c for c in cat_all if c in selected]
    cols = sel_num + sel_cat
    encoders_sel = {k: encoders[k] for k in cols}

    models, coef_table = _fit_all_pd_models(train, X_all[cols], y_all, sel_num, sel_cat, encoders_sel, cfg)

    X_oot = X_oot_all[cols]
    X_tr = X_all[cols]
    comparison = validation.comparison_table(models, X_oot, y_oot, X_tr, y_all, label="oot")
    save_table(comparison, f"pd_model_comparison{tag}.csv")
    print(comparison[["model", "auc", "gini", "ks"]].to_string(index=False))

    champion = max(models, key=lambda m: M.auc(y_oot, m.predict_proba(X_oot)))
    pd_oot = champion.predict_proba(X_oot)
    print(f"    [{kind}] champion: {champion.name}  AUC={M.auc(y_oot, pd_oot):.4f}")

    validation.roc_plot(models, X_oot, y_oot, name=f"roc_curves{tag}.png")
    validation.ks_plot(champion, X_oot, y_oot, name=f"ks_curve{tag}.png")
    validation.score_distribution_plot(champion, X_oot, y_oot, name=f"score_distribution{tag}.png")

    calib = M.calibration_table(y_oot, pd_oot, n_bins=10)
    save_table(calib, f"pd_calibration_champion{tag}.csv")
    validation.calibration_plot(champion, X_oot, y_oot, name=f"calibration{tag}.png")

    # Probability calibration. An isotonic (monotone) map is fitted on the
    # TRAINING window and applied out of time: it cannot change the ranking, so
    # AUC/Gini/KS stay identical, but the probability scale - and therefore the
    # expected loss - becomes usable. Both raw and calibrated figures are kept.
    pd_tr_raw = champion.predict_proba(X_tr)
    calibrator = M.fit_calibrator(y_all, pd_tr_raw)
    pd_tr_cal = M.apply_calibrator(calibrator, pd_tr_raw)
    pd_oot_cal = M.apply_calibrator(calibrator, pd_oot)
    calib_metrics = {
        "method": "isotonic_regression",
        "fit_on": "training window (in-sample scores)",
        "raw_oot": M.calibration_intercept_slope(y_oot, pd_oot),
        "calibrated_oot": M.calibration_intercept_slope(y_oot, pd_oot_cal),
        "calibrated_train": M.calibration_intercept_slope(y_all, pd_tr_cal),
    }
    save_json(_jsonable(calib_metrics), f"pd_calibration_metrics{tag}.json")
    save_table(M.calibration_table(y_oot, pd_oot_cal, n_bins=10), f"pd_calibration_champion_calibrated{tag}.csv")
    raw_brier = calib_metrics["raw_oot"]["brier"]
    cal_brier = calib_metrics["calibrated_oot"]["brier"]
    print(
        f"    [{kind}] calibration (OOT) Brier raw={raw_brier:.5f} calibrated={cal_brier:.5f} "
        f"| mean p calibrated={calib_metrics['calibrated_oot']['mean_predicted']:.4f} "
        f"observed={calib_metrics['calibrated_oot']['observed_rate']:.4f}"
    )

    psi = validation.stability_report(champion, X_tr, X_oot, sel_num, sel_cat)
    save_table(psi, f"pd_psi_champion{tag}.csv")

    # Cut-off chosen on the TRAIN sample (KS-optimal) and applied out of time.
    cutoff = _ks_optimal_cutoff(y_all, champion.predict_proba(X_tr))
    confusion = M.confusion_at_cutoff(y_oot, pd_oot, cutoff)
    save_json(_jsonable(confusion), f"pd_confusion_oot{tag}.json")

    # Points-based scorecard from the interpretable logit.
    scaler = scorecard.ScorecardScaler(base_score=600, base_odds=50, pdo=20)
    betas = dict(zip(coef_table["variable"], coef_table["coefficient"]))
    intercept = float(betas.pop("const", 0.0))
    points = scorecard.build_points_table(encoders_sel, betas, intercept, scaler)
    save_table(points, f"pd_scorecard_points{tag}.csv")
    save_json(_jsonable(scaler.to_dict()), f"pd_scorecard_scaler{tag}.json")

    woe_tr = transform_frame(X_tr, encoders_sel)
    woe_oot = transform_frame(X_oot, encoders_sel)
    score_tr = scorecard.score_from_logit(woe_tr, betas, intercept, scaler)
    score_oot = scorecard.score_from_logit(woe_oot, betas, intercept, scaler)
    _score_histogram(score_tr, score_oot, name=f"scorecard_distribution{tag}.png")

    # Benchmark sensitivity: what does the lender's own grade/rate add on top?
    benchmark = None
    benchmark_gini = None
    try:
        Xb, yb, numb, catb = features.build_model_matrix(train, include_lender_grade=True)
        Xb = features.apply_clip_bounds(Xb, features.fit_clip_bounds(Xb, numb))
        enc_b = fit_encoders(Xb, yb, numb, catb, n_bins=20, min_bin_pct=0.05, max_bins=8)
        bmodel, _ = pd_model.fit_statsmodels_woe("logit_lender_grade", "logit", Xb, yb, enc_b)
        Xb_oot, yb_oot, _, _ = features.build_model_matrix(oot, include_lender_grade=True)
        pb = np.asarray(bmodel.predict_proba(Xb_oot), dtype=float)
        benchmark = round(float(M.auc(yb_oot, pb)), 4)
        benchmark_gini = round(float(M.gini(yb_oot, pb)), 4)
        print(f"    [{kind}] benchmark (with lender grade/int_rate): AUC={benchmark:.4f} Gini={benchmark_gini:.4f}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"    benchmark sensitivity skipped: {exc}")

    shap_importance = None
    lgbm_model = next((m for m in models if m.name == "lightgbm"), None)
    if kind == "mature" and lgbm_model is not None:
        try:
            shap_importance = interpret.shap_summary(lgbm_model, X_oot, sample_size=5000, seed=cfg.seed)
            save_table(shap_importance, f"pd_shap_importance{tag}.csv")
            save_table(interpret.logistic_vs_shap(shap_importance, coef_table), f"pd_shap_vs_logit{tag}.csv")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"    SHAP summary skipped: {exc}")

    return {
        "window": kind,
        "n_train": int(len(train)),
        "n_oot": int(len(oot)),
        "train_years": [int(train["issue_year"].min()), int(train["issue_year"].max())],
        "oot_years": [int(oot["issue_year"].min()), int(oot["issue_year"].max())],
        "default_rate_train": round(float(y_all.mean()), 4),
        "default_rate_oot": round(float(y_oot.mean()), 4),
        "n_variables_binned": len(encoders),
        "n_features_used": len(cols),
        "selected": selected,
        "iv_table": iv_tab,
        "selection": sel,
        "comparison": comparison,
        "champion": champion.name,
        "champion_auc": round(float(M.auc(y_oot, pd_oot)), 4),
        "champion_gini": round(float(M.gini(y_oot, pd_oot)), 4),
        "champion_ks": round(float(M.ks(y_oot, pd_oot)), 4),
        "cutoff": round(float(cutoff), 6),
        "confusion": confusion,
        "psi": psi,
        "calibration": calib,
        "scorecard_points": points,
        "scaler": scaler.to_dict(),
        "score_psi": round(float(M.psi(score_tr, score_oot)), 4),
        "benchmark_auc_lender_grade": benchmark,
        "benchmark_gini_lender_grade": benchmark_gini,
        "calibration_metrics": calib_metrics,
        "shap": shap_importance,
        "coef_table": coef_table,
        "pd_oot": np.asarray(pd_oot, dtype=float),
        "X_oot": X_oot,
        "y_oot": y_oot,
        "oot_frame": oot,
        "models": models,
        "pd_oot_calibrated": np.asarray(pd_oot_cal, dtype=float),
        "pd_train_calibrated": np.asarray(pd_tr_cal, dtype=float),
    }


def _window_comparison(primary: Dict[str, Any], secondary: Dict[str, Any]) -> pd.DataFrame:
    """Side-by-side headline metrics: mature (primary) vs immature (secondary)."""

    def _row(res: Dict[str, Any], label: str) -> Dict[str, Any]:
        cal = res["calibration_metrics"]["calibrated_oot"]
        return {
            "window": label,
            "test_years": f"{res['oot_years'][0]}-{res['oot_years'][1]}",
            "n_test": res["n_oot"],
            "default_rate_test": res["default_rate_oot"],
            "champion": res["champion"],
            "auc": res["champion_auc"],
            "gini": res["champion_gini"],
            "ks": res["champion_ks"],
            "calibration_gap": round(float(cal["mean_predicted"] - cal["observed_rate"]), 4),
            "brier_calibrated": round(float(cal["brier"]), 5),
            "benchmark_auc_lender_grade": res["benchmark_auc_lender_grade"],
            "score_psi": res["score_psi"],
        }

    return pd.DataFrame([_row(primary, "mature (primary)"), _row(secondary, "immature (secondary)")])


def _secondary_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce the secondary-window run to the scalars quoted in the report."""
    cal = res["calibration_metrics"]["calibrated_oot"]
    return {
        "window": res["window"],
        "train_years": res["train_years"],
        "oot_years": res["oot_years"],
        "n_train": res["n_train"],
        "n_oot": res["n_oot"],
        "default_rate_train": res["default_rate_train"],
        "default_rate_oot": res["default_rate_oot"],
        "champion": res["champion"],
        "champion_auc": res["champion_auc"],
        "champion_gini": res["champion_gini"],
        "champion_ks": res["champion_ks"],
        "benchmark_auc_lender_grade": res["benchmark_auc_lender_grade"],
        "calibration_gap": round(float(cal["mean_predicted"] - cal["observed_rate"]), 4),
        "score_psi": res["score_psi"],
    }


def _summarise_pd(res: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce the PD stage result to JSON-safe headline numbers."""
    return {
        "n_train": res["n_train"],
        "n_oot": res["n_oot"],
        "train_years": res["train_years"],
        "oot_years": res["oot_years"],
        "default_rate_train": res["default_rate_train"],
        "default_rate_oot": res["default_rate_oot"],
        "n_variables_binned": res["n_variables_binned"],
        "n_features_used": res["n_features_used"],
        "selected_variables": res["selected"],
        "models": _records(res["comparison"]),
        "champion": res["champion"],
        "champion_auc": res["champion_auc"],
        "champion_gini": res["champion_gini"],
        "champion_ks": res["champion_ks"],
        "benchmark_auc_lender_grade": res["benchmark_auc_lender_grade"],
        "benchmark_gini_lender_grade": res["benchmark_gini_lender_grade"],
        "calibration_metrics": _jsonable(res["calibration_metrics"]),
        "cutoff": res["cutoff"],
        "confusion_oot": _jsonable(res["confusion"]),
        "calibration_oot": _records(res["calibration"]),
        "psi": _records(res["psi"]),
        "score_psi_train_vs_oot": res["score_psi"],
        "iv_top": _records(res["iv_table"].head(15)),
        "shap_top": None if res["shap"] is None else _records(res["shap"].head(15)),
        "split_design": _jsonable(res.get("split_design", {})),
        "window_comparison": _records(res["window_comparison"]) if "window_comparison" in res else [],
        "secondary_window": _jsonable(res.get("secondary_summary", {})),
    }


# ---------------------------------------------------------------------------
# Stage 3: LGD, EAD and expected loss
# ---------------------------------------------------------------------------
def _lgd_feature_frame(df: pd.DataFrame, num: List[str], cat: List[str]) -> pd.DataFrame:
    """Origination features for LGD scoring on an arbitrary loan sample."""
    X = df[num + cat].copy()
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cat:
        X[c] = X[c].astype(str)
    return X


def _el_by_grade_plot(el_by_grade: pd.DataFrame, name: str = "el_by_grade.png") -> None:
    """Expected-loss rate and mean PD side by side for each lender grade."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(el_by_grade["grade"].astype(str), el_by_grade["el_rate"] * 100.0, color="steelblue", label="Expected loss / EAD")
    ax.set_xlabel("Lender grade (A = lowest risk)")
    ax.set_ylabel("Expected loss (% of EAD)")
    ax2 = ax.twinx()
    ax2.plot(el_by_grade["grade"].astype(str), el_by_grade["mean_pd"] * 100.0, "o--", color="crimson", label="Mean PD")
    ax2.set_ylabel("Mean PD (%)", color="crimson")
    ax2.grid(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.set_title("Expected loss and PD by lender grade (out-of-time portfolio)")
    save_fig(fig, name)


def stage_lgd(df: pd.DataFrame, pd_result: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """LGD model (Gamma GLM vs boosting), EAD and expected loss per grade."""
    print("[lgd] loss given default, exposure at default and expected loss ...")
    d = lgd_ead.compute_lgd_ead(df)
    save_table(lgd_ead.lgd_eda_table(d), "lgd_distribution.csv")
    lgd_ead.lgd_distribution_plot(d)
    lgd_ead.lgd_distribution_plot_by_grade(d)

    num, cat = list(features.NUMERIC_FEATURES), list(features.CATEGORICAL_FEATURES)
    # Same primary (mature) window as the PD stage: fit on 2010-2013, test 2014-2015,
    # 36-month loans only.
    train, oot = train_oot_split(df, cfg, kind="mature")
    defaults = d[d["lgd"].notna()]
    train_def = defaults.loc[defaults.index.intersection(train.index)]
    oot_def = defaults.loc[defaults.index.intersection(oot.index)]

    X_tr, y_tr = lgd_ead.prepare_lgd_matrix(train_def, num, cat)
    X_te, y_te = lgd_ead.prepare_lgd_matrix(oot_def, num, cat)
    save_json(
        {
            "n_defaults_train": int(len(y_tr)),
            "n_defaults_oot": int(len(y_te)),
            "mean_lgd_train": round(float(y_tr.mean()), 4),
            "mean_lgd_oot": round(float(y_te.mean()), 4),
            "median_lgd_oot": round(float(y_te.median()), 4),
        },
        "lgd_sample.json",
    )

    gamma_res, gamma_coef = lgd_ead.fit_lgd_gamma(X_tr, y_tr, num, cat)
    save_table(gamma_coef, "lgd_gamma_coefficients.csv")
    pred_gamma = lgd_ead.predict_lgd_gamma(gamma_res, X_te, num, cat)

    tree = lgd_ead.fit_lgd_tree(X_tr, y_tr, num, cat, seed=cfg.seed)
    pred_tree = np.clip(np.asarray(tree.predict(X_te), dtype=float), 0.0, 1.0)

    m_gamma = lgd_ead.regression_metrics(y_te.to_numpy(), pred_gamma)
    m_tree = lgd_ead.regression_metrics(y_te.to_numpy(), pred_tree)
    comparison = pd.DataFrame(
        [{"model": "gamma_glm", **m_gamma}, {"model": "lightgbm", **m_tree}]
    )
    save_table(comparison, "lgd_model_comparison.csv")
    lgd_ead.lgd_scatter_plot(y_te.to_numpy(), pred_gamma, pred_tree)
    print(comparison.to_string(index=False))

    best = "gamma_glm" if m_gamma["mae"] <= m_tree["mae"] else "lightgbm"

    # Score LGD for the whole out-of-time portfolio (EL = PD x LGD x EAD).
    # The PD basis is the calibrated champion score; the same portfolio is also
    # scored with the raw score so the effect of calibration on EL is visible.
    oot_frame = pd_result["oot_frame"]
    X_all = _lgd_feature_frame(oot_frame, num, cat)
    if best == "gamma_glm":
        lgd_hat = lgd_ead.predict_lgd_gamma(gamma_res, X_all, num, cat)
    else:
        lgd_hat = np.clip(np.asarray(tree.predict(X_all), dtype=float), 0.0, 1.0)

    pd_raw = np.asarray(pd_result["pd_oot"], dtype=float)
    pd_cal = np.asarray(pd_result.get("pd_oot_calibrated", pd_raw), dtype=float)
    el_loan = lgd_ead.expected_loss_table(oot_frame, pd_cal, lgd_hat)
    el = lgd_ead.aggregate_el(el_loan)
    save_table(el["by_grade"], "el_by_grade.csv")
    save_table(el["portfolio"], "el_portfolio.csv")
    _el_by_grade_plot(el["by_grade"])

    portfolio = el["portfolio"].iloc[0]
    portfolio_raw_pd = lgd_ead.aggregate_el(
        lgd_ead.expected_loss_table(oot_frame, pd_raw, lgd_hat)
    )["portfolio"].iloc[0]
    return {
        "pd_basis": "champion score, isotonic-calibrated on the training window",
        "n_defaults_train": int(len(y_tr)),
        "n_defaults_oot": int(len(y_te)),
        "realised_lgd_train": round(float(y_tr.mean()), 4),
        "realised_lgd_oot": round(float(y_te.mean()), 4),
        "median_lgd_oot": round(float(y_te.median()), 4),
        "model_comparison": comparison,
        "best_lgd_model": best,
        "ead_total_oot": round(float(portfolio["ead"]), 2),
        "mean_pd_oot": round(float(portfolio["mean_pd"]), 4),
        "mean_pd_oot_raw": round(float(portfolio_raw_pd["mean_pd"]), 4),
        "mean_lgd_oot": round(float(portfolio["mean_lgd"]), 4),
        "expected_loss_oot": round(float(portfolio["expected_loss"]), 2),
        "expected_loss_oot_raw_pd": round(float(portfolio_raw_pd["expected_loss"]), 2),
        "el_rate_oot": round(float(portfolio["el_rate"]), 4),
        "el_rate_oot_raw_pd": round(float(portfolio_raw_pd["el_rate"]), 4),
        "el_by_grade": el["by_grade"],
        "lgd_hat_oot": np.asarray(lgd_hat, dtype=float),
    }


def _summarise_lgd(res: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _jsonable(v) for k, v in res.items() if k in {
        "pd_basis", "n_defaults_train", "n_defaults_oot", "realised_lgd_train", "realised_lgd_oot",
        "median_lgd_oot", "best_lgd_model", "ead_total_oot", "mean_pd_oot", "mean_pd_oot_raw",
        "mean_lgd_oot", "expected_loss_oot", "expected_loss_oot_raw_pd",
        "el_rate_oot", "el_rate_oot_raw_pd",
    }} | {
        "model_comparison": _records(res["model_comparison"]),
        "el_by_grade": _records(res["el_by_grade"]),
    }


# ---------------------------------------------------------------------------
# Stage 4: behavioural segmentation
# ---------------------------------------------------------------------------
def stage_clustering(df: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """k-means (auto-k with silhouette + stability), hierarchical clustering, PCA."""
    print("[clustering] k-means + hierarchical segmentation ...")
    out = clustering.run_clustering(
        df, CLUSTER_FEATURES, k=None, k_range=range(2, 9), seed=cfg.seed, max_rows=120_000
    )
    print(out["kmeans_profile"].to_string(index=False))
    return out


def _summarise_clustering(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "k": res["k"],
        "features": CLUSTER_FEATURES,
        "n_degenerate_segments": res["n_degenerate_segments"],
        "degenerate_share_threshold": res["degenerate_share_threshold"],
        "k_selection": _records(res["k_selection"]),
        "kmeans_profile": _records(res["kmeans_profile"]),
        "hierarchical_profile": _records(res["hierarchical_profile"]),
        "default_rate_spread": _jsonable(res.get("default_rate_spread", {})),
    }


# ---------------------------------------------------------------------------
# Stage 5: Monte Carlo portfolio simulation
# ---------------------------------------------------------------------------
def stage_simulation(df: pd.DataFrame, ctx: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Single-factor Vasicek loss simulation, convergence, variance reduction, stress."""
    print("[simulation] Monte Carlo portfolio loss distribution ...")
    pd_res, lgd_res = ctx["pd"], ctx["lgd"]
    oot = pd_res["oot_frame"]
    pd_base = np.asarray(pd_res.get("pd_oot_calibrated", pd_res["pd_oot"]), dtype=float)

    portfolio = sim.build_portfolio(
        oot, pd_base, lgd_res["lgd_hat_oot"], ead_col="funded_amnt", n_detile=10
    )
    save_table(portfolio.summary().round(6), "sim_portfolio_buckets.csv")
    ead_w_pd = float(np.average(portfolio.pd, weights=portfolio.ead))
    ead_w_lgd = float(np.average(portfolio.lgd, weights=portfolio.ead))
    print(f"    buckets={len(portfolio.labels)}  total EAD={portfolio.total_ead:,.0f}"
          f"  EAD-weighted PD={ead_w_pd:.4f}  LGD={ead_w_lgd:.4f}")

    rho = float(cfg.simulation.get("asset_correlation", 0.12))
    n_sim = int(cfg.simulation.get("n_simulations", 100_000))
    run = sim.simulate_losses(portfolio, rho=rho, n_simulations=n_sim, seed=cfg.seed, antithetic=True)
    stats = run["stats"]
    save_json(_jsonable(stats), "sim_loss_statistics.json")
    save_table(pd.DataFrame([stats]), "sim_loss_statistics.csv")
    sim.loss_distribution_plot(run["losses"], stats)

    vrr = sim.variance_reduction_report(portfolio, rho=rho, n_max=n_sim, seed=cfg.seed)
    save_table(vrr.round(4), "sim_variance_reduction.csv")
    plain = sim.convergence_study(portfolio, rho=rho, n_max=n_sim, seed=cfg.seed, antithetic=False)
    anti = sim.convergence_study(portfolio, rho=rho, n_max=n_sim, seed=cfg.seed, antithetic=True)
    sim.convergence_plot({"plain": plain, "antithetic": anti})
    save_table(plain.round(4), "sim_convergence.csv")
    save_table(anti.round(4), "sim_convergence_antithetic.csv")

    # Importance sampling on the systematic factor: an independent, stronger
    # variance-reduction experiment aimed at the 99.9% VaR estimate.
    is_cfg = cfg.simulation.get("importance_sampling", {}) or {}
    is_out = sim.importance_sampling_var(
        portfolio,
        rho=rho,
        level=0.999,
        n_paths=int(is_cfg.get("n_tail_paths", 20_000)),
        shift=float(is_cfg.get("shift", 2.5)),
        seed=cfg.seed,
    )
    save_json(_jsonable(is_out), "sim_importance_sampling.json")
    is_sens = sim.importance_sampling_sensitivity(
        portfolio,
        rhos_shift=tuple(is_cfg.get("shift_grid", (0.5, 1.0, 1.5, 2.0, 2.5))),
        level=0.999,
        n_paths=int(is_cfg.get("n_tail_paths", 20_000)),
        rho=rho,
        seed=cfg.seed,
    )
    save_table(is_sens.round(4), "sim_importance_sampling_sensitivity.csv")
    print(
        "    importance sampling (99.9%): VarIS={:,} vs naive={:,}  ESS={:,.0f}".format(
            int(is_out["var_importance_sampling"]),
            int(is_out["var_naive_same_paths"]),
            is_out["effective_sample_size"],
        )
    )

    stress = sim.stress_test(
        portfolio,
        pd_shocks=tuple(cfg.simulation.get("pd_shocks", (0.30, 0.50))),
        lgd_shocks=tuple(cfg.simulation.get("lgd_shocks", (0.10,))),
        rho=rho,
        seed=cfg.seed,
    )
    save_table(stress.round(6), "sim_stress_test.csv")
    sim.stress_plot(stress)
    print(
        stress[["scenario", "expected_loss", "var_0.999", "es_0.99"]].round(0).to_string(index=False)
    )

    # Asset-correlation sensitivity. The loss distribution is driven by rho, so
    # it is swept over the Basel retail range and anchored on the PD-dependent
    # correlation implied by the Basel formula for the observed portfolio PD.
    rho_grid = [float(r) for r in cfg.simulation.get("rho_sensitivity", (0.02, 0.05, 0.10, 0.15))]
    rho_basel = sim.basel_retail_correlation(ead_w_pd)
    labels_map: Dict[float, str] = {round(r, 6): f"rho={r:.2f}" for r in rho_grid}
    labels_map[round(rho, 6)] = f"rho={rho:.3f} (base)"
    labels_map[round(rho_basel, 6)] = f"rho={rho_basel:.3f} (Basel implied)"
    ordered = sorted(labels_map.items())
    sens = sim.rho_sensitivity(
        portfolio,
        [r for r, _ in ordered],
        names=[n for _, n in ordered],
        n_simulations=n_sim,
        seed=cfg.seed,
    )
    save_table(sens.round(6), "sim_rho_sensitivity.csv")
    sim.rho_sensitivity_plot(sens, base_rho=rho)
    print(
        sens[["label", "expected_loss", "var_0.99", "var_0.999", "es_0.99"]]
        .round(0)
        .to_string(index=False)
    )

    return {
        "n_buckets": int(len(portfolio.labels)),
        "total_ead": float(portfolio.total_ead),
        "ead_weighted_pd": round(ead_w_pd, 4),
        "ead_weighted_lgd": round(ead_w_lgd, 4),
        "n_simulations": int(run["n_simulations"]),
        "asset_correlation": rho,
        "rho_basel_implied": round(float(rho_basel), 4),
        "rho_note": (
            "Base case rho=0.12 (mid-range of the Basel IRB retail band); the grid spans "
            "0.02-0.15 and includes the PD-dependent Basel formula value."
        ),
        "antithetic": bool(run["antithetic"]),
        "statistics": {k: _jsonable(v) for k, v in stats.items()},
        "convergence": _records(plain),
        "rho_sensitivity": _records(sens),
        "variance_reduction": _records(vrr),
        "importance_sampling": _jsonable(is_out),
        "importance_sampling_sensitivity": _records(is_sens),
        "stress": _records(stress),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Credit-risk modeling pipeline")
    parser.add_argument("stages", nargs="*", default=["all"], help=f"stages: {', '.join(STAGES)} or all")
    args = parser.parse_args(argv)

    cfg = Config.load()
    set_seeds(cfg.seed)
    ensure_dirs()

    requested = STAGES if ("all" in args.stages or not args.stages) else args.stages
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {unknown}. Valid: {STAGES}")

    t0 = time.time()
    summary: Dict[str, Any] = {"seed": cfg.seed, "stages": requested}
    ctx: Dict[str, Any] = {}

    if "data" in requested:
        stage_data()

    need_frame = any(s in requested for s in ("quality", "eda", "pd", "lgd", "clustering", "simulation"))
    df: Optional[pd.DataFrame] = None
    if need_frame:
        print("[load] reading modeling sample and engineering features ...")
        df = load_modeling_frame(cfg)
        summary["data"] = {
            "n_rows": int(len(df)),
            "default_rate": round(float(df["default_flag"].mean()), 4),
            "years": [int(df["issue_year"].min()), int(df["issue_year"].max())],
            "population": data_prep.population_totals(),
        }

    if "quality" in requested and df is not None:
        summary["quality"] = _jsonable(stage_quality(df))
    if "sql" in requested:
        summary["sql"] = {k: list(v.shape) for k, v in stage_sql().items()}
    if "eda" in requested and df is not None:
        summary["eda"] = stage_eda(df, cfg)

    if "pd" in requested and df is not None:
        ctx["pd"] = stage_pd(df, cfg)
        summary["pd"] = _summarise_pd(ctx["pd"])
    if "lgd" in requested and df is not None:
        if "pd" not in ctx:
            ctx["pd"] = stage_pd(df, cfg)
        ctx["lgd"] = stage_lgd(df, ctx["pd"], cfg)
        summary["lgd"] = _summarise_lgd(ctx["lgd"])
    if "clustering" in requested and df is not None:
        ctx["clustering"] = stage_clustering(df, cfg)
        summary["clustering"] = _summarise_clustering(ctx["clustering"])
    if "simulation" in requested and df is not None:
        if "pd" not in ctx:
            ctx["pd"] = stage_pd(df, cfg)
        if "lgd" not in ctx:
            ctx["lgd"] = stage_lgd(df, ctx["pd"], cfg)
        ctx["simulation"] = stage_simulation(df, ctx, cfg)
        summary["simulation"] = ctx["simulation"]

    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    save_json(summary, "summary.json")
    print(f"\n[done] pipeline finished in {summary['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
