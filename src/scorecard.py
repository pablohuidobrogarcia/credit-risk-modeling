"""Convert a logistic-regression scorecard into a points scale.

The standard scaling is

.. math::
    \\text{score} = \\text{offset} + \\text{factor} \\cdot \\ln(\\text{odds})

with ``odds = P(good) / P(bad)``, ``factor = PDO / ln(2)`` and
``offset = base\\_score - factor \\cdot \\ln(base\\_odds)``. Each WoE bin then
receives a number of points proportional to ``-beta_i * WoE_i``.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .woe_iv import WoEEncoder


class ScorecardScaler:
    """Parameters of the linear score <-> log-odds mapping."""

    def __init__(self, base_score: float = 600.0, base_odds: float = 50.0, pdo: float = 20.0):
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.factor = pdo / np.log(2.0)
        self.offset = base_score - self.factor * np.log(base_odds)

    def points(self, log_odds_good: np.ndarray | float) -> np.ndarray | float:
        """Map log-odds of being good to a score."""
        return self.offset + self.factor * log_odds_good

    def to_dict(self) -> Dict[str, float]:
        return {
            "base_score": self.base_score,
            "base_odds": self.base_odds,
            "pdo": self.pdo,
            "factor": float(self.factor),
            "offset": float(self.offset),
        }


def build_points_table(
    encoders: Dict[str, WoEEncoder],
    betas: Dict[str, float],
    intercept: float,
    scaler: ScorecardScaler | None = None,
) -> pd.DataFrame:
    """Per-variable, per-bin points allocation.

    ``betas`` are the logistic coefficients on the WoE scale (all standardised
    the same way), ``intercept`` the model intercept.
    """
    scaler = scaler or ScorecardScaler()
    rows: List[Dict[str, object]] = []
    n_vars = max(len(betas), 1)
    # Distribute the intercept evenly across variables (standard convention).
    intercept_per_var = intercept / n_vars
    for feature, enc in encoders.items():
        beta = float(betas.get(feature, 0.0))
        for _, r in enc.table.iterrows():
            woe = float(r["woe"])
            points = -(woe * beta + intercept_per_var) * scaler.factor
            rows.append(
                {
                    "variable": feature,
                    "bin": str(r["bin"]),
                    "n": int(r["n"]),
                    "bad_rate": float(r["bad_rate"]),
                    "woe": woe,
                    "coefficient": beta,
                    "points": points,
                }
            )
    return pd.DataFrame(rows)


def score_from_woe(W: pd.DataFrame, points_table: pd.DataFrame, scaler: ScorecardScaler | None = None) -> np.ndarray:
    """Compute a score for every row of a WoE matrix.

    Uses the bin-level point allocation, so the score is exactly reproducible
    from the published scorecard (no floating-point drift).
    """
    scaler = scaler or ScorecardScaler()
    scores = np.zeros(len(W))
    for feature, group in points_table.groupby("variable"):
        if feature not in W.columns:
            continue
        # Points are a deterministic function of WoE within a variable, so we can
        # map each row's WoE straight to its points (exactly reproducible).
        woe_to_points = dict(zip(np.round(group["woe"], 10), group["points"]))
        col = W[feature].to_numpy()
        pts = np.array([woe_to_points.get(round(float(v), 10), 0.0) for v in col])
        scores += pts
    return scores


def score_from_logit(X_woe: pd.DataFrame, betas: Dict[str, float], intercept: float, scaler: ScorecardScaler | None = None) -> np.ndarray:
    """Direct score = offset + factor * (intercept + X @ betas) ... on the *bad* scale.

    Here ``betas`` are on the WoE matrix (higher WoE -> higher risk), so the
    good-bad odds derivation flips the sign: a borrower is *good* when the logit
    is low.
    """
    scaler = scaler or ScorecardScaler()
    beta_vec = np.array([betas.get(c, 0.0) for c in X_woe.columns])
    logit_bad = intercept + X_woe.to_numpy() @ beta_vec
    return scaler.points(-logit_bad)
