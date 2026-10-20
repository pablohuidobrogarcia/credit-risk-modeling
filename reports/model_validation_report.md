# Model Validation Report — Lending Club Credit Risk Pipeline

**Scope.** Independent validation of the PD scorecard/challengers, the LGD/EAD/EL model,
the customer segmentation and the Monte Carlo portfolio loss simulation produced by
`python -m src.pipeline all`.

**Evidence base.** Every figure in this report is read from
[`reports/results/summary.json`](results/summary.json) and the CSV artefacts in
`reports/results/`. Nothing is quoted from memory or from a different run.
Run timestamp/seed: seed **42**; full pipeline wall time **1113.5 s**.

**Verdict in one paragraph.** The pipeline is reproducible, leakage-controlled and
honestly documented, but the *model performance is moderate and the out-of-time test
window is right-censored*, which limits what can be concluded. The PD champion reaches
AUC 0.6736 (Gini 0.3473) out of time, while a plain logit given the lender's own
`grade`/`int_rate` reaches AUC 0.7022 (Gini 0.4043). The LGD model has a **negative**
out-of-time R². The portfolio forecast is not usable. The Monte Carlo layer is sound and
its dominant assumption (asset correlation) is disclosed and swept.

---

## 1. Population, sample and target

| Item | Value | Source |
|---|---|---|
| Accepted loans (population) | 2,260,668 | `summary.json → data.population.n_accepted` |
| Resolved loans | 1,345,311 | `data.population.n_resolved` |
| — Fully Paid | 1,076,712 | `data.population.n_fully_paid` |
| — Charged Off | 268,559 | `data.population.n_charged_off` |
| — Default | 40 | `data.population.n_default_status` |
| Modeling sample | 500,000 (37.2 % of resolved) | `data.n_rows` |
| Sample default rate | 19.99 % (99,948 defaults) | `data.default_rate`, `quality.n_defaults` |
| Vintages covered | 2007 – 2018 | `data.years` |

**Target definition.** `default_flag = 1` for `loan_status in ('Charged Off','Default')`,
`0` for `Fully Paid`, undefined otherwise. Only resolved loans enter the sample, so no
loan is labelled with a future-unknown state.

**Assessment.** The definition is consistent, reproducible and standard for this dataset.
The sample is a deterministic, year-stratified draw (hash of `id` within `issue_year`),
which preserves the vintage mix; the 0.03 pp gap between sample (19.99 %) and population
(19.96 %) resolved default rates confirms the stratification did not distort the target.

**Finding S-1 (accepted risk).** The sample contains **only accepted** applications.
The scorecard is therefore conditional on the lender's own approval policy and must not
be used to re-score rejected applications.

---

## 2. Data quality and leakage control

* 62 columns examined; 30 candidate features (22 numeric, 8 categorical) documented in
  `data_quality_missingness.csv` / `data_quality_outliers.csv`.
* **Leakage control passed.** All post-origination fields are dropped before modeling:
  `recoveries`, `collection_recovery_fee`, `last_pymnt_*`, `last_credit_pull_d`,
  `last_fico_range_*`, `total_pymnt*`, `total_rec_*`, `out_prncp*`, `total_cuml_recovered`,
  `hardship_*`, `debt_settlement_*`. This is the single most important control in the
  project and it is enforced in `src/features.py`, shared by every model.
* `grade`, `sub_grade`, `int_rate` are **excluded** from the main feature set (they are
  the lender's own model output) and used only in the explicit benchmark of §3.6.
* Winsorisation bounds, encoders and WoE maps are fitted on the **training window only**.

**Finding Q-1 (accepted).** Outliers are extreme but handled by winsorisation rather than
deletion, which is the conservative choice for scorecard stability.

---

## 3. PD models

### 3.1 Split

Chronological, never random:

| | Window | Loans | Default rate |
|---|---|---|---|
| Train | 2007 – 2016 | 416,141 | 19.73 % |
| OOT test | 2017 – 2018 | 83,859 | 21.29 % |

### 3.2 Variable selection

30 variables binned with a WoE/IV procedure; **10 kept** after IV ranking, correlation
screening and VIF: `term`, `loan_amnt_to_income`, `dti`, `verification_status`,
`tot_cur_bal`, `loan_amnt`, `annual_inc`, `total_rev_hi_lim`, `home_ownership`,
`revol_util`.

Strongest IV: `term` 0.1968 (medium), `loan_amnt_to_income` 0.1196 (medium), then a long
tail of weak predictors (`dti` 0.0739, `verification_status` 0.0532, …). **No variable
reaches a "strong" IV band** — an early, honest signal that the achievable discrimination
is limited.

### 3.3 Out-of-time discrimination (2017–2018)

| Model | AUC | Gini | KS | CV AUC | Train AUC |
|---|---|---|---|---|---|
| **lightgbm (champion)** | **0.6736** | **0.3473** | **0.2524** | 0.6734 | 0.7024 |
| random_forest | 0.6678 | 0.3356 | 0.2447 | 0.6770 | 0.7129 |
| logit_l2 (Ridge) | 0.6609 | 0.3217 | 0.2332 | 0.6715 | 0.6717 |
| logit_l1 (Lasso) | 0.6608 | 0.3215 | 0.2332 | 0.6715 | 0.6717 |
| logit (scorecard) | 0.6608 | 0.3215 | 0.2332 | — | 0.6717 |
| probit | 0.6604 | 0.3209 | 0.2326 | — | 0.6717 |
| decision_tree | 0.6146 | 0.2292 | 0.1733 | 0.6431 | 0.6448 |

Statsmodels scorecards: logit pseudo-R² = 0.0616, probit pseudo-R² = 0.0615.

**Assessment.**

* The champion is selected on OOT AUC; the margin over the interpretable logit is small
  (**+0.0128 AUC**), and the tree models show the classic in-sample/out-of-sample decay
  (lightgbm train 0.7024 → OOT 0.6736; random forest 0.7129 → 0.6678) while the logit
  barely decays (0.6717 → 0.6608). For a production decisioning use case the logit
  scorecard is the defensible choice; the tree is a monitoring challenger.
* **Finding P-1.** Absolute discrimination is moderate (Gini 0.35). For unsecured US
  consumer credit this is realistic, but it is not a strong model.

### 3.4 Benchmark: how much of the ranking comes from the lender?

| Feature set | AUC | Gini |
|---|---|---|
| Without `grade`/`int_rate` (champion lightgbm) | 0.6736 | 0.3473 |
| With `grade`/`int_rate` (plain logit) | **0.7022** | **0.4043** |

**Finding P-2 (important).** A plain logistic regression that is allowed to see the
lender's own `grade` and `int_rate` outperforms our best model built from application
data by **+0.0286 AUC / +0.0570 Gini**. Part of the discrimination a deployed scorecard
would achieve is therefore inherited from the lender's risk engine rather than learned
here. The claim "the model keeps almost all the discriminatory power without the
lender's variables" is **not** supported by the numbers and is not made anywhere in this
repository.

### 3.5 Stability

* Score PSI (train vs OOT): **0.0060** — stable.
* Variable PSI: all stable except `revol_util` **0.1436** ("monitor"); next largest
  `loan_amnt_to_income` 0.0386, `loan_amnt` 0.0307. **Finding P-3:** `revol_util`'s
  population shift should be monitored in any re-scoring.

### 3.6 Calibration

Isotonic calibrator fitted on the training window, applied out of time (monotone, so
AUC/Gini/KS are unchanged):

| Basis (OOT) | Mean predicted | Observed | Calibration-in-the-large | Slope | Brier |
|---|---|---|---|---|---|
| raw champion | 0.1907 | 0.2129 | 0.1378 | 0.8951 | 0.15770 |
| calibrated | 0.1899 | 0.2129 | 0.1428 | 0.7658 | 0.15816 |

Decile table (`pd_calibration_champion.csv`) shows systematic under-prediction in the
middle of the score range (deciles 3–8 differ by −0.022 to −0.051) and near-perfect
agreement in the extreme deciles.

**Finding P-4 (material).** The model **under-predicts the 2017–2018 default rate**
(19.07 % predicted vs 21.29 % observed). The train-window default rate was 19.73 %, so a
level shift between windows explains part of it. The calibrator, fitted in-sample, does
**not** repair the gap out of time (Brier 0.15816 vs 0.15770 raw; slope degrades from
0.8951 to 0.7658). EL is therefore reported under both bases (§5).

### 3.7 Cutoff / confusion matrix (OOT)

Cutoff = 0.201619 (training default rate, raw score):

| TP | FP | FN | TN | Precision | Recall | Approval rate | Bad rate approved |
|---|---|---|---|---|---|---|---|
| 9,881 | 20,664 | 7,970 | 45,344 | 0.3235 | 0.5535 | 78.71 % | 31.31 % |

**Finding P-5.** This cutoff is a *statistical* default, not an economic optimum: the
31.31 % bad rate among approved loans is high, and a business cutoff (rate/return
optimisation) would be set elsewhere. Any decision use requires a proper cost matrix.

### 3.8 Interpretability

SHAP (champion, `pd_shap_importance.csv`) versus logit coefficients
(`pd_shap_vs_logit.csv`). Top SHAP drivers: `term` (0.3504), `dti` (0.2195),
`total_rev_hi_lim` (0.1662), `verification_status` (0.1403), `home_ownership` (0.1306).

**Finding P-6.** The two views agree on the top and bottom of the list but not in the
middle. `home_ownership` ranks 5th by SHAP and 2nd by logit coefficient; `dti` ranks 2nd
by SHAP and 4th by coefficient. Reviewers should treat SHAP importance as
interaction-aware and IV/coefficients as marginal — they are not interchangeable.

---

## 4. LGD / EAD

Sample: defaults in train 82,089; defaults in OOT 17,849.

| | Train | OOT |
|---|---|---|
| Realised LGD (mean) | 0.8825 | **0.9296** |
| Median LGD (OOT) | | **1.0000** |

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **gamma_glm (chosen)** | **0.0827** | **0.1059** | **−0.2251** |
| lightgbm | 0.0833 | 0.1062 | −0.2327 |

**Finding L-1 (material).** Both out-of-time R² are **negative** — neither model beats
predicting the mean, and the tree does not beat the Gamma GLM. The LGD model is **not
usable for individual loan predictions**. Its only defensible use is as a
portfolio/grade-level average.

**Finding L-2.** Observed LGD is nearly degenerate (mean 0.9296, median 1.0000), so
there is very little cross-sectional signal to learn. Mean predicted LGD by grade is
essentially flat (0.8858 for A → 0.8792 for G), which is consistent with the realised
figure but confirms the model carries no grade-level differentiation either.

---

## 5. Expected loss

OOT portfolio: EAD **$1,212,952,700**; mean PD 0.1899 (calibrated) / 0.1907 (raw);
mean LGD 0.8855.

| PD basis | Expected loss | EL / EAD |
|---|---|---|
| isotonic-calibrated champion PD (primary) | **$232,135,558.69** | **0.1914** |
| raw champion PD (secondary) | $229,960,618.07 | 0.1896 |

By grade (`el_by_grade.csv`): EL rate rises monotonically 0.0916 (A) → 0.3210 (G) while
mean LGD stays flat ~0.88, i.e. **the EL ranking is entirely a PD ranking** — consistent
with L-1/L-2.

**Finding E-1.** A 19.14 % EL rate is very high in absolute terms and is a direct
consequence of the 19–21 % default rate in this (resolved-only, censored) sample; it
should not be read as the EL of a fresh, fully censored 2017–2018 book.

---

## 6. Segmentation

k = 2 chosen automatically by maximum silhouette (see `clustering_k_selection.csv`).

| k | Inertia | Silhouette | Largest cluster share | Degenerate clusters |
|---|---|---|---|---|
| **2** | **391,701.92** | **0.2148** | 0.6822 | 0 |
| 3 | 337,016.18 | 0.1634 | 0.4677 | 0 |
| 4 | 311,703.01 | 0.1431 | 0.3530 | 0 |
| 5 | 293,342.21 | 0.1527 | 0.3375 | 0 |
| 6 | 277,865.05 | 0.1293 | 0.2789 | 0 |
| 7 | 265,958.87 | 0.1143 | 0.1932 | 0 |
| 8 | 257,218.58 | 0.1032 | 0.1763 | 0 |

Segment profiles:

| Segment | Share | Default rate | Loan amount | Annual income | DTI | History (months) |
|---|---|---|---|---|---|---|
| 0 | 41.35 % | 0.2105 | $20,751 | $98,753 | 20.81 | 229.9 |
| 1 | 58.65 % | 0.1924 | $9,967 | $60,481 | 16.49 | 170.8 |

**Finding C-1 (material).** After winsorisation no segment is degenerate (0 flagged; the
pre-fix pipeline produced `ValueError: Number of labels is 1` on the silhouette
subsample — see §7). But the two segments differ mainly in **size, income and loan
amount**, with a default-rate gap of only **1.81 pp** (0.2105 vs 0.1924). They are
descriptive customer groups, **not** a risk-rating substitute.

**Finding C-2.** Silhouette monotonically favours the smallest k tested (0.2148 at k = 2
vs 0.1634 at k = 3): the "optimal" k is at the boundary of the search grid, so k = 2 is
the metric-driven answer rather than a clearly correct business segmentation.

**Finding C-3.** Ward hierarchical clustering on a 5,000-loan sample recovers the same
structure (two groups of 39.4 % / 60.6 % with default rates 0.156 / 0.219), which
corroborates the k-means split at sample scale.

---

## 7. Engineering validation performed during this work

| Issue found | Root cause | Resolution |
|---|---|---|
| `ValueError: Number of labels is 1` in the silhouette step | sklearn's `sample_size=` draws a plain random subsample; with very unbalanced clusters it can contain a single label. Outliers (`annual_inc` up to 9.3M, `dti` 999) created such clusters. | Winsorise at the 1st/99th percentile before standardising; compute the silhouette on a **stratified** subsample containing every cluster and return `NaN` instead of raising. |
| `TypeError: Must provide 'func' or tuples` in `profile_segments` | Mixing named-tuple aggregation with plain `col="mean"` kwargs in `DataFrame.agg`. Never exercised before because clustering had never completed. | Rebuilt the aggregation with `groupby()[cols].mean()` + `insert`. |
| Hierarchical profile length mismatch | Linkage was computed on a 5,000-loan subsample while labels were applied to 120,000 rows. | Subsample once and profile the *same* rows. |
| Scorecard points all zero; SHAP/coefficient merge broken | `reset_index(names=…)` on an ndarray gave positional integers as variable names. | Use `res.model.exog_names` / `res.params.index`. |
| Degenerate risk from `MiniBatchKMeans` as default | Measured on the 120k sample: exact `KMeans` inertia 1,174,937 / silhouette 0.2200 vs MiniBatch 1,184,469 / 0.1889. | MiniBatch kept as an opt-in fallback above 100k rows (the sample used is 120k for the study, 40k–120k in practice); exact KMeans remains the default. |

Test suite: **52 unit tests**, all passing on synthetic data; `ruff check src tests` clean;
CI runs lint + tests on Python 3.10 and 3.11 without the dataset.

---

## 8. Monte Carlo portfolio loss simulation

Portfolio: **70 buckets** (grade × PD decile), total EAD **$1,212,952,700**,
EAD-weighted PD **0.2153**, EAD-weighted LGD **0.8848**. PDs are the out-of-time
calibrated champion scores; LGDs are the Gamma GLM output per loan.

Base case, 100,000 paths, antithetic variates, ρ = 0.12:

| Metric | Amount | % of EAD |
|---|---|---|
| Expected loss | $232,133,301 | 19.14 % |
| Unexpected loss (std) | $100,196,831 | 8.26 % |
| VaR 99 % | $511,658,266 | 42.18 % |
| ES 99 % | $558,512,895 | 46.05 % |
| VaR 99.9 % | $618,407,258 | 50.98 % |
| ES 99.9 % | $659,303,409 | 54.36 % |
| Economic capital (99 %) | $279,524,965 | 23.05 % |
| Economic capital (99.9 %) | $386,273,957 | 31.85 % |

### 8.1 Asset correlation (the dominant assumption)

The Basel IRB PD-dependent retail formula, evaluated at the portfolio PD (0.2153),
implies ρ = **0.0301**. The base case is ρ = 0.12. Both are inside the retail band; the
grid makes the choice transparent:

| ρ | EL (% EAD) | UL (% EAD) | VaR 99 % | VaR 99.9 % | ES 99 % |
|---|---|---|---|---|---|
| 0.02 | 19.13 % | 3.33 % | 27.59 % | 30.75 % | 28.97 % |
| 0.0301 (Basel implied) | 19.13 % | 4.09 % | 29.66 % | 33.69 % | 31.43 % |
| 0.05 | 19.14 % | 5.29 % | 33.13 % | 38.47 % | 35.47 % |
| 0.10 | 19.14 % | 7.52 % | 39.90 % | 47.83 % | 43.38 % |
| **0.12 (base)** | **19.14 %** | **8.26 %** | **42.18 %** | **50.98 %** | **46.05 %** |
| 0.15 | 19.14 % | 9.27 % | 45.39 % | 55.35 % | 49.73 % |

**Finding M-1 (material).** EL is essentially invariant to ρ (19.13 %–19.14 %), while
VaR 99.9 % moves from **30.75 % to 55.35 % of EAD** — an 80 % increase — purely from the
correlation assumption. Any capital number from this model is only meaningful together
with its ρ; the repository therefore never quotes one in isolation.

### 8.2 Convergence and variance reduction

Convergence study saved to `sim_convergence.csv` (plain) and
`sim_convergence_antithetic.csv`; the loss distribution stabilises with the number of
paths (standard error falls from $7.95M at 100 paths to $0.32M at 100,000 paths under
plain Monte Carlo).

| Paths | SE (plain) | SE (antithetic) | SE reduction |
|---|---|---|---|
| 100 | 7,946,680 | 7,949,456 | −0.03 % |
| 1,000 | 3,159,726 | 3,162,086 | −0.07 % |
| 10,000 | 1,008,113 | 1,008,468 | −0.04 % |
| 100,000 | 317,269 | 316,849 | **+0.13 %** |

**Finding M-2.** **Antithetic variates give no meaningful variance reduction here**
(between −0.07 % and +0.13 %). With bucketed binomial draws the systematic factor is
already averaged across 70 buckets, so pairing paths adds almost nothing. The result is
reported rather than hidden; a Latin-hypercube or importance-sampling scheme would be the
next thing to try.

### 8.3 Stress scenarios (ρ = 0.12)

| Scenario | EAD-weighted PD | Mean LGD | EL | EL % EAD | VaR 99.9 % |
|---|---|---|---|---|---|
| baseline | 0.2153 | 0.8848 | $232,133,301 | 19.14 % | $618,407,258 |
| PD +30 % | 0.2799 | 0.8848 | $301,744,963 | 24.88 % | $694,932,362 |
| PD +50 % | 0.3230 | 0.8848 | $348,152,135 | 28.70 % | $736,645,004 |
| LGD +10 % | 0.2153 | 0.9733 | $255,341,059 | 21.05 % | $680,237,347 |
| PD +50 % & LGD +10 % | 0.3230 | 0.9733 | $382,958,992 | 31.57 % | $810,297,968 |

**Finding M-3.** Stress is close to linear in PD and LGD (as expected: EL = PD × LGD ×
EAD), while the tail beyond EL grows more slowly, so the *relative* capital buffer
actually shrinks under stress. That is a property of the Vasicek formulation, worth
stating explicitly to avoid over-reading the stress grid.

---

## 9. Summary of findings

| ID | Severity | Finding |
|---|---|---|
| **7.1 / S-1** | **High** | Right-censoring: only 38.17 % (2017) and 11.37 % (2018) of the vintages that form the OOT window were resolved. All OOT metrics are conditional on a biased subsample. |
| **7.1** | **High** | Removing the censored vintages cuts the forecast test RMSE from 0.0981 to 0.0148 (≈ 6.6×), proving censoring — not the model — drives most of the reported error. |
| **P-2** | **High** | Lender `grade`/`int_rate` add +0.0286 AUC / +0.0570 Gini over our best application-data model. |
| **L-1 / L-2** | **High** | LGD R² is negative for both models; LGD is usable only as a grade/portfolio average. |
| **M-1** | **High** | VaR 99.9 % spans 30.75 %–55.35 % of EAD across the ρ grid; ρ dominates the capital figure. |
| **P-4** | Medium | Out-of-time under-prediction (19.07 % vs 21.29 %); in-sample calibrator does not repair it (Brier worsens). |
| **P-1** | Medium | Absolute discrimination is moderate (Gini 0.3473); trees show real in/out decay. |
| **M-2** | Medium | Antithetic variates achieve no variance reduction in this bucketed design. |
| **C-1 / C-2** | Medium | Segments differ by only 1.81 pp in default rate; silhouette prefers the smallest k tested. |
| **P-3** | Low | `revol_util` PSI 0.1436 ("monitor"). |
| **P-5** | Low | Cutoff is statistical, not economic (31.31 % bad rate among approved). |
| **Q-1** | Low | Outliers handled by winsorisation rather than removal (conservative). |

## 10. Recommendation

1. **Use** the logistic scorecard (`logit_l2`) as the production model for
   discrimination, with the lightgbm score as a monitored challenger; the AUC difference
   (0.6609 vs 0.6736) does not justify the loss of explainability.
2. **Do not use** the LGD model at loan level; use the realised/grade-level LGD
   (≈ 0.88–0.93) and revisit with a two-stage (recovery) model on a mature vintage only.
3. **Do not use** the SARIMAX forecast for planning; if vintage trend monitoring is
   required, restrict it to vintages with a resolved share above ~90 % (≤ 2015 here).
4. **Report every capital figure with its ρ**, using the Basel implied ρ (0.0301 at this
   PD) as the reference and 0.12–0.15 as a stress band.
5. **Before any decision use**, re-estimate the cutoff against an explicit cost/return
   matrix and validate on a vintage that is fully matured.
