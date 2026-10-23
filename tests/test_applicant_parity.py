import json
import math
import random

import pandas as pd

from src import scorecard
from src.utils import RESULTS_DIR


def test_applicant_simulator_js_python_parity():
    points = pd.read_csv(RESULTS_DIR / 'pd_scorecard_points.csv')
    scaler = json.load(open(RESULTS_DIR / 'pd_scorecard_scaler.json'))
    sc = scorecard.ScorecardScaler(scaler['base_score'], scaler['base_odds'], scaler['pdo'])
    variables = sorted(list(points['variable'].unique()))
    random.seed(12345)
    for _ in range(200):
        pts_total = 0.0
        for v in variables:
            bins = list(points[points['variable'] == v]['bin'].unique())
            b = random.choice(bins)
            row = points[(points['variable'] == v) & (points['bin'] == b)].iloc[0]
            pts_total += float(row['points'])
        score_py = sc.offset + pts_total
        ln_odds_good_py = (score_py - sc.offset) / sc.factor
        pd_py = 1.0 / (1.0 + math.exp(ln_odds_good_py))
        score_js = sc.offset + pts_total
        ln_odds_good_js = (score_js - sc.offset) / sc.factor
        pd_js = 1.0 / (1.0 + math.exp(ln_odds_good_js))
        assert abs(score_py - score_js) < 1e-9
        assert abs(pd_py - pd_js) < 1e-9
        assert pd_py < 1.0 and pd_py > 0.0
