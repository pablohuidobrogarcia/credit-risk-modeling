"""Data-quality report: missingness, outliers and leakage control."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, LEAKAGE_COLUMNS, NUMERIC_FEATURES
from .utils import save_json, save_table


def missingness_report(df: pd.DataFrame, max_vars: int = 40) -> pd.DataFrame:
    """Share of missing values per column, worst first."""
    miss = df.isna().mean().sort_values(ascending=False)
    out = (
        miss.head(max_vars)
        .rename("pct_missing")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    out["pct_missing"] = out["pct_missing"].round(4)
    return out


def outlier_report(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Descriptive statistics plus IQR-based outlier counts.

    ``n_outliers`` counts observations outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append(
            {
                "variable": col,
                "n": int(s.size),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p01": float(s.quantile(0.01)),
                "median": float(s.median()),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
                "n_outliers_iqr": int(((s < lower) | (s > upper)).sum()),
                "pct_outliers_iqr": round(float(((s < lower) | (s > upper)).mean()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_outliers_iqr", ascending=False).reset_index(drop=True)


def leakage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Show why each post-origination column must be dropped.

    For each leakage column we report the non-null share and how strongly it
    correlates with (or predicts) the default flag, which is exactly the
    "peeking into the future" problem.
    """
    rows = []
    y = df["default_flag"] if "default_flag" in df.columns else None
    for col in LEAKAGE_COLUMNS:
        if col not in df.columns:
            continue
        s = df[col]
        non_null = float(s.notna().mean())
        corr = np.nan
        try:
            if y is not None and pd.api.types.is_numeric_dtype(s):
                corr = float(np.corrcoef(s.fillna(s.median()).astype(float), y.fillna(0))[0, 1])
        except Exception:
            corr = np.nan
        rows.append(
            {
                "variable": col,
                "pct_non_null": round(non_null, 4),
                "pointbiserial_corr_with_default": None if np.isnan(corr) else round(corr, 4),
                "action": "dropped",
            }
        )
    return pd.DataFrame(rows)


def quality_summary(df: pd.DataFrame) -> Dict[str, object]:
    """High-level dataset summary used by the README and the report."""
    summary: Dict[str, object] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_defaults": int(df["default_flag"].sum()) if "default_flag" in df else None,
        "default_rate": round(float(df["default_flag"].mean()), 4) if "default_flag" in df else None,
        "origination_years": None,
        "features_used": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
        "numeric_features": len(NUMERIC_FEATURES),
        "categorical_features": len(CATEGORICAL_FEATURES),
    }
    if "issue_year" in df.columns:
        summary["origination_years"] = [int(df["issue_year"].min()), int(df["issue_year"].max())]
    return summary


def build_reports(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute and persist every data-quality artifact."""
    miss = missingness_report(df)
    outl = outlier_report(df, NUMERIC_FEATURES)
    leak = leakage_report(df)
    save_table(miss, "data_quality_missingness.csv")
    save_table(outl, "data_quality_outliers.csv")
    save_table(leak, "data_quality_leakage.csv")
    save_json(quality_summary(df), "data_quality_summary.json")
    return {"missingness": miss, "outliers": outl, "leakage": leak}
