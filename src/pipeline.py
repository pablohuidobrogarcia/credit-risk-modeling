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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from . import (
    calibration,
    clustering,
    data_prep,
    features,
    interpret,
    lgd_ead,
    pd_model,
    quality,
    reject_inference,
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

STAGES = ["data", "quality", "sql", "eda", "pd", "lgd", "clustering", "simulation", "reject"]

# Borrower / bureau variables used for the behavioral segmentation. They are
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
# Stage 2: EDA and time-series modeling
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
        "forecast_benchmark": _records(out["forecast_benchmark"]),
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
    X_chal: pd.DataFrame | None = None,
    num_chal: List[str] | None = None,
    cat_chal: List[str] | None = None,
    lgbm_params: Dict[str, Any] | None = None,
    monotone: Dict[str, int] | None = None,
    years: pd.Series | None = None,
) -> Tuple[List[pd_model.FittedModel], pd.DataFrame]:
    """Fit the scorecard logit/probit, the regularised logits and the tree challengers.

    Linear/GLM models consume the WoE matrix built from the IV-selected variables
    (the interpretable scorecard). The tree/boosting challengers consume the
    *wider* leakage-free origination feature set (``X_chal``) so they are not
    handicapped by the scorecard's parsimonious selection.
    """
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

    # Challenger design matrix: the wider set by default, falling back to the
    # scorecard columns when the wider set is not supplied.
    Xc = X_chal if X_chal is not None else X
    num_c = num_chal if num_chal is not None else num_feats
    cat_c = cat_chal if cat_chal is not None else cat_feats
    columns = num_c + cat_c
    print(f"    challenger feature set: {len(columns)} variables ({len(num_c)} numeric, {len(cat_c)} categorical)")

    cv_sub = train.sample(n=min(100_000, len(train)), random_state=cfg.seed)
    for name in ("decision_tree", "random_forest", "lightgbm"):
        auc_cv, std_cv = pd_model.cross_validate_pipeline(
            pd_model.make_tree_pipeline(
                name, num_c, cat_c, cfg.seed, lgbm_params=lgbm_params if name == "lightgbm" else None,
                monotone_constraints=monotone if name == "lightgbm" else None,
            ),
            cv_sub[columns],
            cv_sub["default_flag"],
            seed=cfg.seed,
        )
        model = pd_model.fit_tree_model(
            name,
            Xc[columns],
            y,
            num_c,
            cat_c,
            cfg.seed,
            run_cv=False,
            lgbm_params=lgbm_params if name == "lightgbm" else None,
            monotone_constraints=monotone if name == "lightgbm" else None,
        )
        model.cv_auc, model.cv_auc_std = auc_cv, std_cv
        models.append(model)
        print(f"    {name}: cv_auc={auc_cv:.4f}")
    _ = years
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


def _temporal_oos_scores(
    train: pd.DataFrame,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    champion: pd_model.FittedModel,
    model_fit_years: Sequence[int],
    score_years: Sequence[int],
    num_feats: List[str],
    cat_feats: List[str],
    encoders: Dict[str, Any],
    cfg: Config,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Out-of-sample scores for ``score_years`` from a model trained on ``model_fit_years``.

    Both slices live *inside* the training window, so this produces genuinely
    out-of-sample scores for the calibration/selection slices without ever
    touching the primary test window.
    """
    m_fit = train["issue_year"].isin(list(model_fit_years)).to_numpy()
    m_score = train["issue_year"].isin(list(score_years)).to_numpy()
    if m_fit.sum() < 100 or m_score.sum() < 100:
        return None
    Xf_raw, yf = X_all.loc[m_fit], y_all.loc[m_fit]
    Xs_raw = X_all.loc[m_score]
    if champion.kind == "raw":
        cols = champion.features
        tmp = pd_model.fit_tree_model(
            champion.name,
            Xf_raw[cols],
            yf,
            num_feats,
            cat_feats,
            cfg.seed,
            run_cv=False,
            lgbm_params=champion.params.get("lgbm_params") or None,
            monotone_constraints=champion.params.get("monotone_constraints") or None,
        )
        return np.asarray(tmp.predict_proba(Xs_raw[cols]), dtype=float), np.asarray(y_all.loc[m_score], dtype=int)
    # WoE / linear champion: refit the encoders on the fit slice, then the logit.
    b = features.fit_clip_bounds(Xf_raw, num_feats)
    Xf = features.apply_clip_bounds(Xf_raw, b)
    Xs = features.apply_clip_bounds(Xs_raw, b)
    enc = fit_encoders(Xf, yf, num_feats, cat_feats, n_bins=20, min_bin_pct=0.05, max_bins=8)
    sel = [c for c in champion.features if c in enc]
    enc = {c: enc[c] for c in sel}
    link = "probit" if champion.name == "probit" else "logit"
    tmp, _ = pd_model.fit_statsmodels_woe(champion.name, link, Xf[sel], yf, enc)
    return np.asarray(tmp.predict_proba(Xs[sel]), dtype=float), np.asarray(y_all.loc[m_score], dtype=int)


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

    # --- Challenger feature set -------------------------------------------------
    # The scorecard above is deliberately parsimonious (IV + corr + VIF). The
    # challengers instead receive every leakage-free origination variable, so
    # the comparison measures modeling capacity rather than the scorecard's
    # variable selection. This is the single biggest change of block 2.
    use_full = bool(cfg.modeling.get("challenger_uses_full_features", True))
    if use_full:
        chal_num, chal_cat = list(num_all), list(cat_all)
    else:
        chal_num, chal_cat = sel_num, sel_cat
    # Drop any candidate variables that are entirely missing inside this window:
    # a constant/all-NaN column cannot be imputed or split on, and keeping it only
    # emits a sklearn warning. The drop is recorded in the run log.
    all_nan = [c for c in (chal_num + chal_cat) if X_all[c].isna().all()]
    if all_nan:
        chal_num = [c for c in chal_num if c not in all_nan]
        chal_cat = [c for c in chal_cat if c not in all_nan]
        print(f"    [{kind}] dropped {len(all_nan)} all-missing challenger variable(s): {', '.join(all_nan)}")
    chal_cols = chal_num + chal_cat
    X_chal = X_all[chal_cols]

    # --- LightGBM tuning by temporal CV *inside* the train window only ----------
    # Fit on 2010-2012, validate on 2013. The primary test window (2014-2015) is
    # never touched, so the golden rule (no tuning on test) is preserved.
    lgbm_params: Dict[str, Any] = {}
    monotone: Dict[str, int] = {}
    tuning_table = None
    tune_cfg = cfg.modeling.get("lgbm_tuning", {}) or {}
    if kind == "mature" and bool(tune_cfg.get("enabled", False)):
        monotone = dict(tune_cfg.get("monotone_constraints", {}) or {})
        try:
            lgbm_params, tuning_table = pd_model.tune_lightgbm_temporal(
                X_chal,
                y_all,
                train["issue_year"],
                chal_num,
                chal_cat,
                tune_years=tune_cfg.get("tuning_train_years", [2010, 2011, 2012]),
                valid_years=tune_cfg.get("tuning_valid_years", [2013]),
                monotone_constraints=monotone or None,
                n_trials=int(tune_cfg.get("n_trials", 12)),
                seed=cfg.seed,
            )
            save_table(tuning_table, f"pd_lgbm_tuning{tag}.csv")
            print(f"    [{kind}] LightGBM temporal tuning best valid AUC={tuning_table['valid_auc'].max():.4f}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"    LightGBM tuning skipped: {exc}")

    models, coef_table = _fit_all_pd_models(
        train,
        X_all[cols],
        y_all,
        sel_num,
        sel_cat,
        encoders_sel,
        cfg,
        X_chal=X_chal,
        num_chal=chal_num,
        cat_chal=chal_cat,
        lgbm_params=lgbm_params,
        monotone=monotone,
    )

    # Each model is evaluated on the columns it was actually trained on.
    X_oot = X_oot_all[cols]
    X_tr = X_all[cols]

    def _model_matrix(model: pd_model.FittedModel) -> pd.DataFrame:
        return X_oot_all[model.features] if model.kind == "raw" else X_oot

    def _train_matrix(model: pd_model.FittedModel) -> pd.DataFrame:
        return X_all[model.features] if model.kind == "raw" else X_tr

    comparison = validation.comparison_table_multi(
        models, X_oot_all, y_oot, X_all, y_all, label="oot"
    )
    save_table(comparison, f"pd_model_comparison{tag}.csv")
    print(comparison[["model", "auc", "gini", "ks"]].to_string(index=False))

    champion = max(models, key=lambda m: M.auc(y_oot, m.predict_proba(_model_matrix(m))))
    pd_oot = champion.predict_proba(_model_matrix(champion))
    print(f"    [{kind}] champion: {champion.name}  AUC={M.auc(y_oot, pd_oot):.4f}")

    validation.roc_plot_multi(models, X_oot_all, y_oot, name=f"roc_curves{tag}.png")
    validation.ks_plot(champion, _model_matrix(champion), y_oot, name=f"ks_curve{tag}.png")
    validation.score_distribution_plot(champion, _model_matrix(champion), y_oot, name=f"score_distribution{tag}.png")

    calib = M.calibration_table(y_oot, pd_oot, n_bins=10)
    save_table(calib, f"pd_calibration_champion{tag}.csv")
    validation.calibration_plot(champion, _model_matrix(champion), y_oot, name=f"calibration{tag}.png")

    # ---------------------------------------------------------------------
    # Probability calibration (block 3).
    # Every calibrator is fitted on OUT-OF-SAMPLE scores from a later temporal
    # slice of the training window (a model trained on 2010-2012 scores 2013),
    # and the method is chosen on a separate inner slice (2020-2011 model scores
    # 2012). The primary 2014-2015 test window is never used to fit or choose.
    # ---------------------------------------------------------------------
    X_champ_tr = _train_matrix(champion)
    X_champ_oot = _model_matrix(champion)
    pd_tr_raw = champion.predict_proba(X_champ_tr)
    pd_oot = np.asarray(pd_oot, dtype=float)

    cal_cfg = cfg.modeling.get("calibration", {}) or {}
    fit_years = [int(y) for y in cal_cfg.get("fit_years", [2013])]
    fit_model_years = [int(y) for y in cal_cfg.get("model_fit_years", [2010, 2011, 2012])]
    sel_years = [int(y) for y in cal_cfg.get("select_years", [2012])]
    sel_model_years = [int(y) for y in cal_cfg.get("select_model_fit_years", [2010, 2011])]
    methods = list(cal_cfg.get("methods", ["none", "intercept", "platt", "isotonic"]))

    fit_scores = _temporal_oos_scores(
        train, X_all, y_all, champion, fit_model_years, fit_years, num_all, cat_all, encoders, cfg
    )
    sel_scores = _temporal_oos_scores(
        train, X_all, y_all, champion, sel_model_years, sel_years, num_all, cat_all, encoders, cfg
    )

    calibration_comparison = None
    selected_method = "none"
    pd_oot_cal = pd_oot
    pd_tr_cal = pd_tr_raw
    if fit_scores is not None and sel_scores is not None:
        p_fit, y_fit = fit_scores
        p_sel, y_sel = sel_scores
        cal_table, cal_out, selected_method = calibration.compare_calibrators(
            p_fit, y_fit, p_sel, y_sel, pd_oot, np.asarray(y_oot), methods=methods
        )
        calibration_comparison = cal_table
        save_table(cal_table, f"pd_calibration_comparison{tag}.csv")
        print(f"    [{kind}] calibration selected on the {sel_years} slice: {selected_method}")

        chosen = calibration.CALIBRATORS[selected_method]().fit(p_fit, y_fit)
        pd_oot_cal = chosen.predict(pd_oot)
        pd_tr_cal = chosen.predict(pd_tr_raw)
        save_table(M.calibration_table(y_oot, pd_oot_cal, n_bins=10), f"pd_calibration_champion_calibrated{tag}.csv")
    else:  # pragma: no cover - fallback if slices are too small
        chosen = M.fit_calibrator(y_all, pd_tr_raw)
        pd_tr_cal = M.apply_calibrator(chosen, pd_tr_raw)
        pd_oot_cal = M.apply_calibrator(chosen, pd_oot)
        selected_method = "isotonic"

    calib_metrics = {
        "method": selected_method,
        "fit_on": f"out-of-sample scores from the {fit_years} slice (model fitted on {fit_model_years})",
        "selected_on": f"out-of-sample scores from the {sel_years} slice (model fitted on {sel_model_years})",
        "raw_oot": M.calibration_intercept_slope(y_oot, pd_oot),
        "calibrated_oot": M.calibration_intercept_slope(y_oot, pd_oot_cal),
        "calibrated_train": M.calibration_intercept_slope(y_all, pd_tr_cal),
    }
    save_json(_jsonable(calib_metrics), f"pd_calibration_metrics{tag}.json")
    raw_brier = calib_metrics["raw_oot"]["brier"]
    cal_brier = calib_metrics["calibrated_oot"]["brier"]
    print(
        f"    [{kind}] calibration (OOT) method={selected_method} Brier raw={raw_brier:.5f} "
        f"calibrated={cal_brier:.5f} "
        f"| mean p calibrated={calib_metrics['calibrated_oot']['mean_predicted']:.4f} "
        f"observed={calib_metrics['calibrated_oot']['observed_rate']:.4f}"
    )

    if champion.kind == "raw":
        psi = validation.stability_report(champion, X_champ_tr, X_champ_oot, chal_num, chal_cat)
    else:
        psi = validation.stability_report(champion, X_champ_tr, X_champ_oot, sel_num, sel_cat)
    save_table(psi, f"pd_psi_champion{tag}.csv")

    # Cut-off chosen on the TRAIN sample (KS-optimal) and applied out of time.
    cutoff = _ks_optimal_cutoff(y_all, champion.predict_proba(X_champ_tr))
    confusion = M.confusion_at_cutoff(y_oot, pd_oot, cutoff)
    save_json(_jsonable(confusion), f"pd_confusion_oot{tag}.json")

    # Points-based scorecard from the interpretable logit (always the scorecard
    # columns, independent of whichever challenger won).
    scaler = scorecard.ScorecardScaler(base_score=600, base_odds=50, pdo=20)
    betas = dict(zip(coef_table["variable"], coef_table["coefficient"]))
    intercept = float(betas.pop("const", 0.0))
    points = scorecard.build_points_table(encoders_sel, betas, intercept, scaler)
    save_table(points, f"pd_scorecard_points{tag}.csv")
    save_json(_jsonable(scaler.to_dict()), f"pd_scorecard_scaler{tag}.json")

    woe_tr = transform_frame(X_tr, encoders_sel)
    woe_oot = transform_frame(X_oot, encoders_sel)
    # WoE matrix + target for the primary test window: exported so the optional R
    # script (r/replicate_scorecard.R) can refit the identical logit with glm().
    # WoE values are log-odds; rounding to 6 dp keeps the file small (loan-level,
    # ~446k rows) without changing any refit result.
    woe_export = woe_oot.round(6)
    woe_export.insert(0, "default_flag", np.asarray(y_oot, dtype=int))
    save_table(woe_export, f"pd_woe_oot{tag}.csv")
    score_tr = scorecard.score_from_logit(woe_tr, betas, intercept, scaler)
    score_oot = scorecard.score_from_logit(woe_oot, betas, intercept, scaler)
    _score_histogram(score_tr, score_oot, name=f"scorecard_distribution{tag}.png")

    # Benchmark sensitivity: what does the lender's own grade/int_rate add on top?
    # Fair comparison: the SAME wide feature set + the SAME LightGBM recipe, with
    # the lender's grade/int_rate added as extra columns. If the benchmark cannot
    # beat the challenger, our origination features already capture the lender's
    # information; if it can, the gap is the price of not using the lender model.
    benchmark = None
    benchmark_gini = None
    try:
        chal_num_b = list(num_all) + ["int_rate"]
        chal_cat_b = list(cat_all) + ["grade"]
        Xb_all, yb, _, _ = features.build_model_matrix(train, include_lender_grade=True)
        Xb_oot_all, yb_oot, _, _ = features.build_model_matrix(oot, include_lender_grade=True)
        b_cols = chal_num_b + chal_cat_b
        bmodel = pd_model.fit_tree_model(
            "lightgbm",
            Xb_all[b_cols],
            yb,
            chal_num_b,
            chal_cat_b,
            cfg.seed,
            run_cv=False,
            lgbm_params=lgbm_params or None,
            monotone_constraints=None,
        )
        pb = np.asarray(bmodel.predict_proba(Xb_oot_all[b_cols]), dtype=float)
        benchmark = round(float(M.auc(yb_oot, pb)), 4)
        benchmark_gini = round(float(M.gini(yb_oot, pb)), 4)
        print(f"    [{kind}] benchmark (wide features + lender grade/int_rate, LightGBM): "
              f"AUC={benchmark:.4f} Gini={benchmark_gini:.4f}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"    benchmark sensitivity skipped: {exc}")

    shap_importance = None
    lgbm_model = next((m for m in models if m.name == "lightgbm"), None)
    if kind == "mature" and lgbm_model is not None:
        try:
            shap_importance = interpret.shap_summary(
                lgbm_model, _model_matrix(lgbm_model), sample_size=5000, seed=cfg.seed
            )
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
        "n_features_challenger": len(chal_cols),
        "challenger_uses_full_features": bool(use_full),
        "lgbm_tuning": tuning_table,
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
        "calibration_comparison": calibration_comparison,
        "calibration_method": selected_method,
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
        "n_features_challenger": res.get("n_features_challenger"),
        "challenger_uses_full_features": res.get("challenger_uses_full_features"),
        "lgbm_tuning": _records(res["lgbm_tuning"]) if res.get("lgbm_tuning") is not None else None,
        "selected_variables": res["selected"],
        "models": _records(res["comparison"]),
        "champion": res["champion"],
        "champion_auc": res["champion_auc"],
        "champion_gini": res["champion_gini"],
        "champion_ks": res["champion_ks"],
        "benchmark_auc_lender_grade": res["benchmark_auc_lender_grade"],
        "benchmark_gini_lender_grade": res["benchmark_gini_lender_grade"],
        "calibration_metrics": _jsonable(res["calibration_metrics"]),
        "calibration_comparison": _records(res["calibration_comparison"]) if res.get("calibration_comparison") is not None else [],
        "calibration_method": res.get("calibration_method"),
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

    # Two-part model (block 4): boundary logistic + interior fractional GLM.
    two_part = lgd_ead.fit_lgd_two_part(X_tr, y_tr, num, cat, seed=cfg.seed)
    pred_two = lgd_ead.predict_lgd_two_part(two_part, X_te, num, cat)
    boundary_flag_te = lgd_ead.lgd_boundary_flags(y_te)
    save_json(
        {
            "boundary_rate_train": round(float(two_part["boundary_rate"]), 4),
            "boundary_rate_oot": round(float(boundary_flag_te.mean()), 4),
        },
        "lgd_two_part_boundary.json",
    )

    m_gamma = lgd_ead.regression_metrics(y_te.to_numpy(), pred_gamma)
    m_tree = lgd_ead.regression_metrics(y_te.to_numpy(), pred_tree)
    m_two = lgd_ead.regression_metrics(y_te.to_numpy(), pred_two)
    comparison = pd.DataFrame(
        [
            {"model": "gamma_glm", **m_gamma},
            {"model": "lightgbm", **m_tree},
            {"model": "two_part", **m_two},
        ]
    )
    save_table(comparison, "lgd_model_comparison.csv")

    # Grade-level calibration: mean predicted vs mean observed LGD by grade. This
    # is the metric that matters because LGD is consumed at portfolio/grade level.
    grade_cal = pd.DataFrame(
        {
            "gamma_glm": lgd_ead.lgd_by_grade_calibration(oot_def, y_te.to_numpy(), pred_gamma)["gap"],
            "lightgbm": lgd_ead.lgd_by_grade_calibration(oot_def, y_te.to_numpy(), pred_tree)["gap"],
            "two_part": lgd_ead.lgd_by_grade_calibration(oot_def, y_te.to_numpy(), pred_two)["gap"],
        }
    )
    grade_cal.insert(0, "grade", lgd_ead.lgd_by_grade_calibration(oot_def, y_te.to_numpy(), pred_two)["grade"])
    grade_cal.insert(1, "n", lgd_ead.lgd_by_grade_calibration(oot_def, y_te.to_numpy(), pred_two)["n"])
    grade_cal = grade_cal.rename(columns={c: f"{c}_gap" for c in grade_cal.columns if c.endswith(("glm", "lightgbm", "part"))})
    observed_by_grade = lgd_ead.lgd_by_grade_calibration(oot_def, y_te.to_numpy(), pred_two)
    save_table(observed_by_grade, "lgd_grade_calibration.csv")
    save_table(grade_cal, "lgd_grade_calibration_gaps.csv")

    # Portfolio-level calibration for each model (single number, easy to read).
    portfolio_lgd_cal = pd.DataFrame(
        [
            {
                "model": name,
                "mean_predicted": round(float(np.mean(pred)), 4),
                "mean_observed": round(float(np.mean(y_te)), 4),
                "gap": round(float(np.mean(pred) - np.mean(y_te)), 4),
            }
            for name, pred in (("gamma_glm", pred_gamma), ("lightgbm", pred_tree), ("two_part", pred_two))
        ]
    )
    save_table(portfolio_lgd_cal, "lgd_portfolio_calibration.csv")

    lgd_ead.lgd_scatter_plot(y_te.to_numpy(), pred_gamma, pred_tree)
    print(comparison.to_string(index=False))
    print(portfolio_lgd_cal.to_string(index=False))

    # Choose by grade-level calibration quality first (the realistic use case),
    # then loan-level MAE. Grade calibration = mean absolute grade gap.
    grade_mae = {
        "gamma_glm": float(grade_cal["gamma_glm_gap"].abs().mean()),
        "lightgbm": float(grade_cal["lightgbm_gap"].abs().mean()),
        "two_part": float(grade_cal["two_part_gap"].abs().mean()),
    }
    best = min(grade_mae, key=lambda k: grade_mae[k])
    save_json({k: round(v, 5) for k, v in grade_mae.items()}, "lgd_grade_gap_mae.json")

    # Score LGD for the whole out-of-time portfolio (EL = PD x LGD x EAD).
    # The PD basis is the calibrated champion score; the same portfolio is also
    # scored with the raw score so the effect of calibration on EL is visible.
    oot_frame = pd_result["oot_frame"]
    X_all = _lgd_feature_frame(oot_frame, num, cat)
    if best == "gamma_glm":
        lgd_hat = lgd_ead.predict_lgd_gamma(gamma_res, X_all, num, cat)
    elif best == "lightgbm":
        lgd_hat = np.clip(np.asarray(tree.predict(X_all), dtype=float), 0.0, 1.0)
    else:
        lgd_hat = lgd_ead.predict_lgd_two_part(two_part, X_all, num, cat)

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
        "pd_basis": "champion score, probability-calibrated on a held-out temporal slice",
        "n_defaults_train": int(len(y_tr)),
        "n_defaults_oot": int(len(y_te)),
        "realised_lgd_train": round(float(y_tr.mean()), 4),
        "realised_lgd_oot": round(float(y_te.mean()), 4),
        "median_lgd_oot": round(float(y_te.median()), 4),
        "boundary_rate_train": round(float(two_part["boundary_rate"]), 4),
        "boundary_rate_oot": round(float(boundary_flag_te.mean()), 4),
        "model_comparison": comparison,
        "portfolio_calibration": portfolio_lgd_cal,
        "grade_calibration": observed_by_grade,
        "grade_gap_mae": {k: round(v, 5) for k, v in grade_mae.items()},
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
        "median_lgd_oot", "boundary_rate_train", "boundary_rate_oot", "best_lgd_model", "ead_total_oot",
        "mean_pd_oot", "mean_pd_oot_raw", "mean_lgd_oot", "expected_loss_oot", "expected_loss_oot_raw_pd",
        "el_rate_oot", "el_rate_oot_raw_pd", "grade_gap_mae",
    }} | {
        "model_comparison": _records(res["model_comparison"]),
        "portfolio_calibration": _records(res["portfolio_calibration"]),
        "grade_calibration": _records(res["grade_calibration"]),
        "el_by_grade": _records(res["el_by_grade"]),
    }


# ---------------------------------------------------------------------------
# Stage 4: behavioral segmentation
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
    scores = res["k_selection"]
    chosen = scores[scores["k"] == res["k"]]
    chosen_row = chosen.iloc[0] if len(chosen) else None
    return {
        "k": res["k"],
        "features": CLUSTER_FEATURES,
        "silhouette": (round(float(chosen_row["silhouette"]), 4) if chosen_row is not None else None),
        "stability_ari": (round(float(chosen_row["stability_ari"]), 4) if chosen_row is not None else None),
        "n_degenerate_segments": res["n_degenerate_segments"],
        "degenerate_share_threshold": res["degenerate_share_threshold"],
        "k_selection": _records(scores),
        "kmeans_profile": _records(res["kmeans_profile"]),
        "gmm_profile": _records(res["gmm_profile"]) if res.get("gmm_profile") is not None else [],
        "gmm_stability_ari": res.get("gmm_stability_ari"),
        "gmm_spread": _jsonable(res.get("gmm_spread")),
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
    # Base run: stratified sampling of the systematic factor is the variance-
    # reduction technique of record (it beats plain sampling); antithetic
    # variates are kept as an honest negative result in the report.
    strat = bool(cfg.simulation.get("stratified_sampling", True))
    anti = bool(cfg.simulation.get("antithetic_variates", False))
    run = sim.simulate_losses(
        portfolio, rho=rho, n_simulations=n_sim, seed=cfg.seed, antithetic=anti, stratified=strat
    )
    stats = run["stats"]
    save_json(_jsonable(stats), "sim_loss_statistics.json")
    save_table(pd.DataFrame([stats]), "sim_loss_statistics.csv")
    sim.loss_distribution_plot(run["losses"], stats)

    vrr = sim.variance_reduction_report(portfolio, rho=rho, n_max=n_sim, seed=cfg.seed)
    save_table(vrr.round(4), "sim_variance_reduction.csv")
    plain = sim.convergence_study(portfolio, rho=rho, n_max=n_sim, seed=cfg.seed, antithetic=False)
    anti_cs = sim.convergence_study(portfolio, rho=rho, n_max=n_sim, seed=cfg.seed, antithetic=True)
    strat_cs = sim.convergence_study(portfolio, rho=rho, n_max=n_sim, seed=cfg.seed, stratified=True)
    sim.convergence_plot({"plain": plain, "antithetic": anti_cs, "stratified": strat_cs})
    save_table(plain.round(4), "sim_convergence.csv")
    save_table(anti_cs.round(4), "sim_convergence_antithetic.csv")
    save_table(strat_cs.round(4), "sim_convergence_stratified.csv")
    se_red_strat = float(vrr["se_reduction_stratified_pct"].iloc[-1])
    se_red_anti = float(vrr["se_reduction_antithetic_pct"].iloc[-1])
    print(f"    variance reduction at {n_sim:,} paths: stratified={se_red_strat:.1f}%  antithetic={se_red_anti:.1f}% (SE reduction)")

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
        rhos_shift=tuple(is_cfg.get("shift_grid", (0.5, 1.0, 1.5, 2.0, 2.5, 3.09))),
        level=0.999,
        n_paths=int(is_cfg.get("n_tail_paths", 20_000)),
        rho=rho,
        seed=cfg.seed,
    )
    save_table(is_sens.round(4), "sim_importance_sampling_sensitivity.csv")
    is_vr = sim.variance_reduction_comparison(
        portfolio,
        rho=rho,
        level=0.999,
        n_paths=int(is_cfg.get("n_tail_paths", 20_000)),
        shift=float(is_cfg.get("shift", 1.5)),
        seed=cfg.seed,
        n_replications=int(is_cfg.get("n_replications", 40)),
    )
    save_json(_jsonable(is_vr), "sim_importance_sampling_variance_reduction.json")
    save_table(pd.DataFrame([is_vr]).round(6), "sim_importance_sampling_variance_reduction.csv")
    print(
        "    importance sampling (99.9%): VarIS={:,} vs naive={:,}  ESS={:,.0f}  "
        "RMSE IS/naive={:.2f}x ({} reps)".format(
            int(is_out["var_importance_sampling"]),
            int(is_out["var_naive_same_paths"]),
            is_out["effective_sample_size"],
            is_vr["rmse_reduction_factor"],
            is_vr["n_replications"],
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
        "stratified": bool(run["stratified"]),
        "se_reduction_stratified_pct": round(se_red_strat, 2),
        "se_reduction_antithetic_pct": round(se_red_anti, 2),
        "statistics": {k: _jsonable(v) for k, v in stats.items()},
        "convergence": _records(plain),
        "rho_sensitivity": _records(sens),
        "variance_reduction": _records(vrr),
        "importance_sampling": _jsonable(is_out),
        "importance_sampling_sensitivity": _records(is_sens),
        "importance_sampling_variance_reduction": _jsonable(is_vr),
        "stress": _records(stress),
    }


# ---------------------------------------------------------------------------
# Stage 6: exploratory reject inference
# ---------------------------------------------------------------------------
def stage_reject(df: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """Exploratory reject-inference evidence (labeled exploratory).

    Nothing in the main PD/LGD/simulation chain depends on this stage; it exists
    to quantify - not fix - the selection bias introduced by training only on the
    lender's accepted loans. See ``src/reject_inference.py``.
    """
    print("[reject] exploratory reject-inference analysis ...")
    n = int(cfg.modeling.get("reject_sample_size", 200_000))
    out = reject_inference.run_reject_inference(df, n_rejected=n, seed=cfg.seed)
    save_table(out["common_variable_shift"], "reject_common_variable_shift.csv")
    save_json(_jsonable(out["extrapolation_gap"]), "reject_extrapolation_gap.json")
    print(out["common_variable_shift"].to_string(index=False))
    print(f"    illustrative PD gap (rejected - accepted): {out['extrapolation_gap'].get('illustrative_gap')}")
    return {
        "common_variable_shift": out["common_variable_shift"],
        "extrapolation_gap": out["extrapolation_gap"],
        "n_rejected_sampled": int(len(out["rejected_sample"])),
        "label": "EXPLORATORY - not used by any headline number",
    }


def _summarise_reject(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": res["label"],
        "n_rejected_sampled": res["n_rejected_sampled"],
        "common_variable_shift": _records(res["common_variable_shift"]),
        "extrapolation_gap": _jsonable(res["extrapolation_gap"]),
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

    need_frame = any(s in requested for s in ("quality", "eda", "pd", "lgd", "clustering", "simulation", "reject"))
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
    if "reject" in requested and df is not None:
        ctx["reject"] = stage_reject(df, cfg)
        summary["reject"] = _summarise_reject(ctx["reject"])

    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    save_json(summary, "summary.json")
    print(f"\n[done] pipeline finished in {summary['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
