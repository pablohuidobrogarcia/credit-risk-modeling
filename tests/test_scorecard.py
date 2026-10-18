"""Points-based scorecard scaling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.scorecard import ScorecardScaler, build_points_table, score_from_logit
from src.woe_iv import fit_numeric


def test_scaler_factor_and_offset():
    s = ScorecardScaler(base_score=600, base_odds=50, pdo=20)
    assert abs(s.factor - 20 / np.log(2)) < 1e-12
    assert abs(s.offset - (600 - s.factor * np.log(50))) < 1e-9
    d = s.to_dict()
    assert set(d) == {"base_score", "base_odds", "pdo", "factor", "offset"}


def test_points_increase_with_odds_of_being_good():
    s = ScorecardScaler()
    assert s.points(2.0) > s.points(0.0) > s.points(-2.0)
    assert abs(s.points(0.0) - s.offset) < 1e-9


def test_scorecard_monotone_in_woe():
    rng = np.random.default_rng(0)
    n = 8000
    x = pd.Series(rng.normal(size=n))
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-1.5 * x))).astype(float))
    enc = fit_numeric(x, y, "x", max_bins=5)
    # beta > 0: higher WoE (riskier bin) must receive fewer points.
    scaler = ScorecardScaler()
    table = build_points_table({"x": enc}, {"x": 1.0}, intercept=-1.0, scaler=scaler)
    ordered = table.sort_values("woe")
    assert (np.diff(ordered["points"].to_numpy()) <= 1e-9).all()


def test_score_from_logit_end_to_end():
    scaler = ScorecardScaler()
    W = pd.DataFrame({"a": [0.0, 1.0, -1.0]})
    scores = score_from_logit(W, {"a": 0.5}, intercept=0.0, scaler=scaler)
    assert scores.shape == (3,)
    # higher WoE -> higher bad logit -> lower score
    assert scores[1] < scores[0] < scores[2]
