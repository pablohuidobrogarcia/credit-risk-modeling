"""Variable selection: Information Value, correlation and VIF screens."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .woe_iv import WoEEncoder, iv_summary


def correlation_screen(
    W: pd.DataFrame, iv_table: pd.DataFrame, threshold: float = 0.7
) -> Tuple[List[str], pd.DataFrame]:
    """Drop the weaker of any pair of variables correlated above ``threshold``.

    Strength is the variable's Information Value. Returns (kept, correlation
    matrix).
    """
    corr = W.corr().abs()
    iv_map = dict(zip(iv_table["variable"], iv_table["iv"]))
    order = sorted(W.columns, key=lambda c: iv_map.get(c, 0.0), reverse=True)
    kept: List[str] = []
    dropped: List[dict] = []
    for col in order:
        conflict = None
        for k in kept:
            if corr.loc[col, k] > threshold:
                conflict = k
                break
        if conflict is None:
            kept.append(col)
        else:
            dropped.append({"variable": col, "correlated_with": conflict, "abs_corr": float(corr.loc[col, conflict])})
    return kept, pd.DataFrame(dropped)


def vif_table(W: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor for every column of a design matrix."""
    X = W.copy()
    X["const"] = 1.0
    values = X.to_numpy(dtype=float)
    rows = []
    for i, col in enumerate(X.columns):
        try:
            v = float(variance_inflation_factor(values, i))
        except Exception:  # pragma: no cover - singular designs
            v = float("nan")
        rows.append({"variable": col, "vif": v})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def select_variables(
    encoders: Dict[str, WoEEncoder],
    W: pd.DataFrame,
    iv_min: float = 0.02,
    corr_max: float = 0.7,
    vif_max: float = 5.0,
    forced: Sequence[str] = (),
) -> dict:
    """Full selection pipeline: IV -> correlation -> VIF.

    Returns a dict with the kept variables and the intermediate diagnostics so
    the decisions are auditable in the notebook/report.
    """
    iv = iv_summary(encoders)
    iv_pass = iv[iv["iv"] >= iv_min]["variable"].tolist()
    forced_set = [v for v in forced if v in iv["variable"].values]
    iv_pass = sorted(set(iv_pass) | set(forced_set), key=lambda c: -float(iv.set_index("variable").loc[c, "iv"]))

    corr_kept, corr_dropped = correlation_screen(W[iv_pass], iv, threshold=corr_max)

    vif = vif_table(W[corr_kept])
    vif_drop = vif[(vif["variable"] != "const") & (vif["vif"] > vif_max)]["variable"].tolist()
    # Drop iteratively, worst VIF first, recomputing.
    kept = list(corr_kept)
    vif_history = [vif]
    while True:
        vt = vif_table(W[kept])
        bad = vt[(vt["variable"] != "const") & (vt["vif"] > vif_max)]
        if bad.empty:
            break
        worst = str(bad.iloc[0]["variable"])
        kept.remove(worst)
        vif_history.append(vt)
        if len(kept) <= 1:
            break

    return {
        "iv_table": iv,
        "iv_pass": iv_pass,
        "correlation_dropped": corr_dropped,
        "selected": kept,
        "vif_final": vif_history[-1],
        "vif_dropped": vif_drop,
    }
