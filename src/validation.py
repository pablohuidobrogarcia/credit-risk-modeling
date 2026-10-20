"""Out-of-time validation: discrimination, calibration, stability and plots."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from . import metrics as M
from .pd_model import FittedModel
from .utils import save_fig


def evaluate(model: FittedModel, X: pd.DataFrame, y: pd.Series, label: str) -> Dict[str, float]:
    """Discrimination metrics for one model on one sample."""
    p = model.predict_proba(X)
    return {
        "model": model.name,
        "sample": label,
        "n": int(len(y)),
        "default_rate": round(float(np.mean(y)), 4),
        "auc": round(M.auc(y, p), 4),
        "gini": round(M.gini(y, p), 4),
        "ks": round(M.ks(y, p), 4),
        "mean_pd": round(float(np.mean(p)), 4),
    }


def comparison_table(
    models: Sequence[FittedModel],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    label: str = "oot",
) -> pd.DataFrame:
    """AUC/Gini/KS table for every model on the out-of-time sample."""
    rows = []
    for m in models:
        row = evaluate(m, X_test, y_test, label)
        if m.cv_auc is not None:
            row["cv_auc"] = round(float(m.cv_auc), 4)
        else:
            row["cv_auc"] = np.nan
        if X_train is not None and y_train is not None:
            row["train_auc"] = round(M.auc(y_train, m.predict_proba(X_train)), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def _select_matrix(model: FittedModel, X: pd.DataFrame) -> pd.DataFrame:
    """Return the columns a model was trained on (raw trees need their own set)."""
    if model.kind == "raw" and all(c in X.columns for c in model.features):
        return X[model.features]
    return X


def comparison_table_multi(
    models: Sequence[FittedModel],
    X_test_all: pd.DataFrame,
    y_test: pd.Series,
    X_train_all: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    label: str = "oot",
) -> pd.DataFrame:
    """AUC/Gini/KS table when models consume *different* feature matrices.

    Linear scorecard models read the WoE-selected columns; the tree challengers
    read their own (wider) raw feature set. Each model is scored on the columns
    it was actually fitted on, so the comparison is apples to apples.
    """
    rows = []
    for m in models:
        row = evaluate(m, _select_matrix(m, X_test_all), y_test, label)
        row["cv_auc"] = round(float(m.cv_auc), 4) if m.cv_auc is not None else np.nan
        if X_train_all is not None and y_train is not None:
            row["train_auc"] = round(M.auc(y_train, m.predict_proba(_select_matrix(m, X_train_all))), 4)
        row["n_features"] = len(m.features)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def roc_plot(models: Sequence[FittedModel], X_test: pd.DataFrame, y_test: pd.Series, name: str = "roc_curves.png"):
    """ROC curves for all models on the out-of-time sample."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for m in models:
        p = m.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, lw=1.8, label=f"{m.name} (AUC={M.auc(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-time ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    return save_fig(fig, name)


def roc_plot_multi(
    models: Sequence[FittedModel],
    X_test_all: pd.DataFrame,
    y_test: pd.Series,
    name: str = "roc_curves.png",
):
    """ROC curves when models consume different feature matrices."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for m in models:
        p = m.predict_proba(_select_matrix(m, X_test_all))
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, lw=1.8, label=f"{m.name} (AUC={M.auc(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-time ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    return save_fig(fig, name)


def ks_plot(model: FittedModel, X_test: pd.DataFrame, y_test: pd.Series, name: str = "ks_curve.png"):
    """KS curve: cumulative bad and good distributions vs score threshold."""
    p = model.predict_proba(X_test)
    _, cum_bad, cum_good, ks_value = M.ks_curve(y_test, p)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(0, 1, len(cum_bad))
    ax.plot(x, cum_bad, label="Cumulative bads", color="crimson")
    ax.plot(x, cum_good, label="Cumulative goods", color="steelblue")
    ax.fill_between(x, cum_bad, cum_good, color="grey", alpha=0.15)
    ax.set_xlabel("Population (sorted by predicted PD, descending)")
    ax.set_ylabel("Cumulative share")
    ax.set_title(f"KS curve - {model.name} (KS={ks_value:.3f})")
    ax.legend(loc="lower right")
    ax.annotate(f"KS={ks_value:.3f}", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=11)
    return save_fig(fig, name)


def calibration_plot(model: FittedModel, X_test: pd.DataFrame, y_test: pd.Series, name: str = "calibration.png", n_bins: int = 10):
    """Predicted vs observed default rate by predicted-PD decile."""
    p = model.predict_proba(X_test)
    tab = M.calibration_table(y_test, p, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(tab["predicted"], tab["observed"], "o-", label="Observed vs predicted")
    lims = [0, max(tab["predicted"].max(), tab["observed"].max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title(f"Calibration - {model.name} (out-of-time)")
    ax.legend()
    return save_fig(fig, name), M.calibration_table(y_test, model.predict_proba(X_test))


def score_distribution_plot(
    model: FittedModel, X_test: pd.DataFrame, y_test: pd.Series, name: str = "score_distribution.png"
):
    """Predicted-PD distribution split by realised outcome."""
    p = model.predict_proba(X_test)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, np.quantile(p, 0.999), 60)
    ax.hist(p[y_test == 0], bins=bins, alpha=0.6, label="Non-default", color="steelblue", density=True)
    ax.hist(p[y_test == 1], bins=bins, alpha=0.6, label="Default", color="crimson", density=True)
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Density")
    ax.set_title(f"Score distribution - {model.name}")
    ax.legend()
    return save_fig(fig, name)


def stability_report(
    model: FittedModel,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num: Sequence[str],
    cat: Sequence[str],
) -> pd.DataFrame:
    """Per-variable PSI (train vs out-of-time) plus the PSI of the model score."""
    tab = M.psi_report(X_train, X_test, numeric=num, categorical=cat)
    p_train, p_test = model.predict_proba(X_train), model.predict_proba(X_test)
    score_psi = M.psi(p_train, p_test, bins=10)
    tab = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "variable": "__model_score__",
                        "psi": round(score_psi, 4),
                        "interpretation": "stable" if score_psi < 0.1 else ("monitor" if score_psi < 0.25 else "unstable"),
                    }
                ]
            ),
            tab,
        ],
        ignore_index=True,
    )
    tab["psi"] = tab["psi"].round(4)
    return tab


def score_by_year(
    model: FittedModel, X: pd.DataFrame, years: pd.Series
) -> pd.DataFrame:
    """Mean predicted PD vs realised default rate by origination year."""
    p = model.predict_proba(X)
    df = pd.DataFrame({"year": np.asarray(years), "p": p})
    out = df.groupby("year")["p"].mean().rename("mean_pd").to_frame()
    return out.reset_index()
