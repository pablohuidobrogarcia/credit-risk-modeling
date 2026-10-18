"""Discrimination, calibration and stability metrics for credit risk models."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_score))


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2 * AUC - 1 (Somers' D for a binary outcome)."""
    return 2.0 * auc(y_true, y_score) - 1.0


def ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic = max separation of good/bad CDFs."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def ks_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (thresholds, bad_cdf, good_cdf, ks_value) for a KS plot."""
    order = np.argsort(y_score)
    y = np.asarray(y_true)[order]
    s = np.asarray(y_score)[order]
    n_bad = y.sum()
    n_good = len(y) - n_bad
    cum_bad = np.cumsum(y) / max(n_bad, 1)
    cum_good = np.cumsum(1 - y) / max(n_good, 1)
    diff = np.abs(cum_good - cum_bad)
    i = int(np.argmax(diff))
    # Thin the curves for plotting.
    idx = np.linspace(0, len(s) - 1, min(500, len(s))).astype(int)
    return s[idx], cum_bad[idx], cum_good[idx], float(diff[i])


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two numeric samples.

    The bin edges are derived from ``expected`` (the reference/training
    population) so that the comparison is always anchored to the baseline.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # Degenerate (near-constant) variable: compare means only.
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf

    e_perc = np.histogram(expected, bins=edges)[0] / len(expected)
    a_perc = np.histogram(actual, bins=edges)[0] / len(actual)
    eps = 1e-6
    e_perc = np.clip(e_perc, eps, None)
    a_perc = np.clip(a_perc, eps, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """PSI for a categorical variable, using the reference categories."""
    e = expected.astype(str).value_counts(normalize=True)
    a = actual.astype(str).value_counts(normalize=True)
    cats = e.index.union(a.index)
    e_perc = e.reindex(cats).fillna(0).clip(lower=1e-6)
    a_perc = a.reindex(cats).fillna(0).clip(lower=1e-6)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def psi_report(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-variable PSI table (Table 4 style stability report)."""
    categorical = list(categorical or [])
    rows: list[Dict[str, float | str]] = []
    for col in numeric:
        if col in reference.columns and col in comparison.columns:
            rows.append({"variable": col, "psi": psi(reference[col].values, comparison[col].values)})
    for col in categorical:
        if col in reference.columns and col in comparison.columns:
            rows.append({"variable": col, "psi": psi_categorical(reference[col], comparison[col])})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

    def _band(v: float) -> str:
        if np.isnan(v):
            return "n/a"
        if v < 0.10:
            return "stable"
        if v < 0.25:
            return "monitor"
        return "unstable"

    if not out.empty:
        out["interpretation"] = out["psi"].map(_band)
    return out


def calibration_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Decile-based calibration table: predicted vs observed default rate."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    tab = (
        df.groupby("bin", observed=True)
        .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
        .reset_index(drop=True)
    )
    tab["diff"] = tab["predicted"] - tab["observed"]
    return tab


def confusion_at_cutoff(y_true: np.ndarray, y_prob: np.ndarray, cutoff: float) -> Dict[str, float]:
    """Confusion-matrix summary at a probability cutoff."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_prob) >= cutoff).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    total = tp + fp + fn + tn
    return {
        "cutoff": float(cutoff),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "approval_rate": (tn + fp) / total if total else float("nan"),
        "bad_rate_approved": fp / (fp + tn) if (fp + tn) else float("nan"),
    }
