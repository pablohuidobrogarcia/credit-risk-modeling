#!/usr/bin/env python
import json
import numpy as np
import pandas as pd
from pathlib import Path
from src import simulation as sim
from src.utils import RESULTS_DIR
from src import scorecard


def build_risk_surface(summary):
    stats = summary['simulation']['statistics']
    total_ead = summary['simulation']['total_ead']
    rhos = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]
    pds = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]
    z_var = np.zeros((len(pds), len(rhos)))
    z_es = np.zeros((len(pds), len(rhos)))
    z_el = np.zeros((len(pds), len(rhos)))
    ead_w_pd = summary['simulation'].get('ead_weighted_pd', 0.16)
    base_port = sim.BucketedPortfolio(
        ead=np.array([total_ead*0.1, total_ead*0.2, total_ead*0.3, total_ead*0.4]),
        lgd=np.array([0.8,0.85,0.87,0.9]),
        pd=np.array([ead_w_pd*0.5, ead_w_pd*0.8, ead_w_pd, ead_w_pd*1.2]),
        n=np.array([100000,200000,300000,400000]),
        labels=['g1','g2','g3','g4'],
    )
    rng = np.random.default_rng(42)
    for i, pm in enumerate(pds):
        for j, rho in enumerate(rhos):
            p_st = sim.BucketedPortfolio(
                ead=base_port.ead,
                lgd=base_port.lgd,
                pd=np.clip(base_port.pd * float(pm), 1e-9, 1-1e-9),
                n=base_port.n,
                labels=base_port.labels,
            )
            out = sim.simulate_losses(p_st, rho=float(rho), n_simulations=50000,
                                      seed=int(rng.integers(0,10_000_000)),
                                      antithetic=True, stratified=False)
            st = out['stats']
            z_var[i,j] = float(st['var_0.999']) / total_ead * 100.0
            z_es[i,j] = float(st['es_0.999']) / total_ead * 100.0
            z_el[i,j] = float(st['expected_loss']) / total_ead * 100.0
    base_idx_r = rhos.index(0.12)
    base_idx_p = pds.index(1.0)
    base_var = z_var[base_idx_p, base_idx_r]
    headline = stats['var_0.999_pct_ead'] * 100.0
    return {
        'rhos': rhos,
        'pd_multipliers': pds,
        'z_var_0.999': z_var.tolist(),
        'z_es_0.999': z_es.tolist(),
        'z_el': z_el.tolist(),
        'n_paths_per_point': 50000,
        'base_case': {'rho': 0.12, 'pd_mult': 1.0},
        'base_case_var_0.999_pct_ead': float(base_var),
        'headline_var_0.999_pct_ead': float(headline),
    }


def build_scenario_explorer(summary):
    stats = summary['simulation']['statistics']
    total_ead = summary['simulation']['total_ead']
    ead_w_pd = summary['simulation'].get('ead_weighted_pd', 0.16)
    port = sim.BucketedPortfolio(
        ead=np.array([total_ead*0.1, total_ead*0.2, total_ead*0.3, total_ead*0.4]),
        lgd=np.array([0.8,0.85,0.87,0.9]),
        pd=np.array([ead_w_pd*0.5, ead_w_pd*0.8, ead_w_pd, ead_w_pd*1.2]),
        n=np.array([100000,200000,300000,400000]),
        labels=['g1','g2','g3','g4'],
    )
    rhos = [0.05, 0.08, 0.10, 0.12, 0.15]
    pds = [1.0, 1.2, 1.5]
    lgds = [1.0, 1.1]
    rng = np.random.default_rng(12345)
    scenarios = []
    hist_map = {}
    for rho in rhos:
        for pm in pds:
            for lm in lgds:
                pst = sim.BucketedPortfolio(
                    ead=port.ead,
                    lgd=np.clip(port.lgd*float(lm), 0,1),
                    pd=np.clip(port.pd*float(pm), 1e-9,1-1e-9),
                    n=port.n,
                    labels=port.labels,
                )
                out = sim.simulate_losses(pst, rho=float(rho), n_simulations=30000,
                                          seed=int(rng.integers(0,10_000_000)),
                                          antithetic=False, stratified=True)
                losses = np.asarray(out['losses'], dtype=float)
                h = sim.histogram_export(losses, port.total_ead, n_bins=60)
                key = f'rho{rho:.3f}_pd{pm:.2f}_lgd{lm:.2f}'
                hist_map[key] = {'bin_edges_pct_ead': h['bin_edges_pct_ead'], 'counts': h['counts'], 'exact': h['exact']}
                scenarios.append({'key': key, 'rho': float(rho), 'pd_mult': float(pm), 'lgd_mult': float(lm)})
    base_key = 'rho0.120_pd1.00_lgd1.00'
    return {
        'scenarios': scenarios,
        'histograms': hist_map,
        'n_paths_per_scenario': 30000,
        'base_case_key': base_key,
        'base_case_exact': hist_map[base_key]['exact'],
        'headline_stats': stats,
    }


def build_applicant_simulator(summary):
    points = pd.read_csv(RESULTS_DIR / 'pd_scorecard_points.csv')
    scaler = json.load(open(RESULTS_DIR / 'pd_scorecard_scaler.json'))
    sc = scorecard.ScorecardScaler(scaler['base_score'], scaler['base_odds'], scaler['pdo'])
    try:
        woe_test = pd.read_csv(RESULTS_DIR / 'pd_woe_oot.csv')
        coef = pd.read_csv(RESULTS_DIR / 'pd_logit_coefficients.csv')
        betas = dict(zip(coef['variable'], coef['coefficient']))
        if 'const' in betas:
            intercept = betas.pop('const')
        else:
            intercept = 0.0
        scores_test = []
        for _, row in woe_test.head(100000).iterrows():
            logit_bad = intercept + sum(betas.get(c,0)*row.get(c,0) for c in betas if c in row)
            s = sc.offset + sc.factor * (-logit_bad)
            if np.isfinite(s):
                scores_test.append(float(s))
        hist, bin_edges = np.histogram(scores_test, bins=50)
        cum = np.cumsum(hist).astype(float)
        cum_pct = (cum / len(scores_test)).tolist()
        return {
            'points_table': points.to_dict(orient='records'),
            'scaler': scaler,
            'variables': sorted(list(points['variable'].unique())),
            'test_score_hist': hist.tolist(),
            'test_score_bins': bin_edges.tolist(),
            'test_score_cum_pct': cum_pct,
            'test_n': len(scores_test),
            'gini': summary['pd']['champion_gini'],
            'auc': summary['pd']['champion_auc'],
        }
    except Exception:
        return {
            'points_table': points.to_dict(orient='records'),
            'scaler': scaler,
            'variables': sorted(list(points['variable'].unique())),
            'gini': summary['pd']['champion_gini'],
            'auc': summary['pd']['champion_auc'],
        }
