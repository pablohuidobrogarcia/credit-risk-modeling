from src.utils import RESULTS_DIR
import json, numpy as np, pandas as pd
from src import scorecard

s = json.load(open('reports/results/summary.json'))
scaler = json.load(open(str(RESULTS_DIR)+'/pd_scorecard_scaler.json'))
sc = scorecard.ScorecardScaler(scaler['base_score'], scaler['base_odds'], scaler['pdo'])
woe_test = pd.read_csv(str(RESULTS_DIR)+'/pd_woe_oot.csv')
coef = pd.read_csv(str(RESULTS_DIR)+'/pd_logit_coefficients.csv')
betas = dict(zip(coef['variable'], coef['coefficient']))
if 'const' in betas:
    intercept = betas.pop('const')
else:
    intercept = 0.0
scores = []
flags = []
for _, row in woe_test.iterrows():
    logit_bad = intercept + sum(betas.get(c,0)*row.get(c,0) for c in betas if c in row)
    sc_score = sc.offset + sc.factor * (-logit_bad)
    if np.isfinite(sc_score):
        scores.append(float(sc_score))
        flags.append(int(row.get('default_flag',0)))
scores = np.array(scores)
flags = np.array(flags)
bins = np.linspace(float(scores.min()), float(scores.max()), 81)
goods_counts, _ = np.histogram(scores[flags==0], bins=bins)
bads_counts, _ = np.histogram(scores[flags==1], bins=bins)
ce = {
    'cutoff': s['pd'].get('cutoff'),
    'ks_cutoff': s['pd'].get('cutoff'),
    'confusion_oot': s['pd'].get('confusion_oot'),
    'bins': bins.tolist(),
    'goods_counts': goods_counts.tolist(),
    'bads_counts': bads_counts.tolist(),
    'n_goods': int((flags==0).sum()),
    'n_bads': int((flags==1).sum()),
}
with open('reports/results/showcase_cutoff_explorer.json','w') as f:
    json.dump(ce, f)
print('ok')
