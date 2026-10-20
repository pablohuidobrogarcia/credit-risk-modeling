# Model Validation Report — Lending Club Credit Risk Pipeline

**Scope.** Independent validation of the PD scorecard/challengers, the LGD/EAD/EL model,
the customer segmentation and the Monte Carlo portfolio loss simulation produced by
`python -m src.pipeline all`.

**Evidence base.** Every figure in this report is read from
[`reports/results/summary.json`](results/summary.json) and the CSV artefacts in
`reports/results/`. Nothing is quoted from memory or from a different run.
Run seed: **42**; full pipeline wall time **550.2 s**.

**Verdict in one paragraph.** The pipeline is reproducible and leakage-controlled, and the
validation design now **removes right-censoring from the headline metrics by construction**:
the primary out-of-time window is built from `(year, term)` cells that are ≥ 90 % resolved
(train 2010–2013 / test 2014–2015, 36-month loans only). On that honest window the PD
champion reaches AUC 0.6157 (Gini 0.2315), and a plain logit given the lender's own
`grade`/`int_rate` reaches AUC 0.6700 (Gini 0.3399). Testing on the old, censored
2017–2018 window would have reported AUC 0.6736 (Gini 0.3473) — **the censored window
inflates AUC by +0.0579**, which is why it is now only a labelled secondary analysis. The
LGD model still has a **negative** out-of-time R². The portfolio forecast is weak. The
Monte Carlo layer is sound, its dominant assumption (asset correlation) is disclosed and
swept, and its importance-sampling estimator is **verified against a closed-form benchmark**
and shown to cut the tail-VaR RMSE by **4.36×** at equal paths.

---

## 1. Population, sample and target

| Item | Value | Source |
|---|---|---|
| Accepted loans (population) | 2,260,668 | `summary.json → data.population.n_accepted` |
| Resolved loans | 1,345,350 | `data.population.n_resolved` |
| — Fully Paid | 1,076,751 | `data.population.n_fully_paid` |
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

### 3.1 Split — derived from the data, not fixed by hand

The primary out-of-time window is restricted to `(issue_year, term)` cells whose
*population* is ≥ 90 % resolved (`modeling.min_resolved_share`), so the metrics are not
contaminated by right-censoring. The rule and its evidence are written to
`summary.json → pd.split_design`.

| | Window | Loans | Default rate | Resolved share |
|---|---|---|---|---|
| Train | 2010 – 2013, term 36 m | 61,800 | 12.36 % | 99.59 % |
| OOT test | 2014 – 2015, term 36 m | 165,597 | 14.47 % | 99.97 % |

The 60-month cells of 2014–2015 are only 82.85 % / 67.0 % resolved and 2009 (36 m) is
89.30 % resolved, so all are excluded from the primary split.

**Secondary (immature) window.** The exact same code path is re-run on the old
issue-year ≥ 2017 window so the censoring effect is *measured*:

| | Window | Loans | Default rate | Resolved share |
|---|---|---|---|---|
| Train | 2007 – 2016 | 416,141 | 19.73 % | — |
| OOT test | 2017 – 2018 | 83,859 | 21.29 % | 38.17 % / 11.37 % |

**Finding C-0 (High — the reason for the redesign).** Training/testing on the mature
window versus the censored window changes the champion AUC from **0.6736** to **0.6157**,
i.e. **the censored window inflates discounting power by +0.0579 AUC**. All headline
metrics in this report are stated on the mature window; the censored figure is quoted only
as the secondary comparison in §3.3.

### 3.2 Variable selection

30 variables binned with a WoE/IV procedure on the **training window only**; **7 kept**
after IV ranking, correlation screening and VIF: `annual_inc`, `installment_to_income`,
`dti`, `revol_util`, `home_ownership`, `credit_history_months`, `purpose`.

Strongest IV on the mature training window: `annual_inc` 0.0991, `installment_to_income`
0.0732, then `loan_amnt_to_income` 0.0499, `dti` 0.0444. **Every variable is in the "weak"
IV band** — an honest, early signal that the achievable discrimination is limited, and
weaker than the censored window would have suggested.

### 3.3 Out-of-time discrimination (primary window, 2014–2015)

| Model | AUC | Gini | KS | CV AUC | Train AUC |
|---|---|---|---|---|---|
| **random_forest (champion)** | **0.6157** | **0.2315** | **0.1702** | 0.6304 | 0.7140 |
| logit_l2 (Ridge) | 0.6113 | 0.2226 | 0.1646 | 0.6302 | 0.6307 |
| probit | 0.6113 | 0.2227 | 0.1643 | — | 0.6307 |
| logit (scorecard) | 0.6112 | 0.2224 | 0.1641 | — | 0.6307 |
| logit_l1 (Lasso) | 0.6112 | 0.2223 | 0.1641 | 0.6303 | 0.6307 |
| lightgbm | 0.6068 | 0.2137 | 0.1589 | 0.6171 | 0.7897 |
| decision_tree | 0.5611 | 0.1221 | 0.1215 | 0.5736 | 0.5769 |

**Comparison against the censored secondary window** (`pd_window_comparison.csv`):

| Window | OOT years | Champion | AUC | Gini | KS | Calibration gap | Score PSI |
|---|---|---|---|---|---|---|---|
| **Mature (primary)** | 2014–2015 | random_forest | **0.6157** | 0.2315 | 0.1702 | −0.0251 | 0.0089 |
| Immature (secondary) | 2017–2018 | lightgbm | **0.6736** | 0.3473 | 0.2524 | −0.0229 | 0.0060 |

**Assessment.**

* On the honest window the champion is a random forest, but it beats the interpretable
  logit scorecard by only **+0.0044 AUC**. LightGBM *overfits* (train AUC 0.7897 → OOT
  0.6068) and lands below the penalised logits. For production the logit scorecard is the
  defensible choice; the tree is a monitoring challenger.
* **Finding P-1 (Medium).** Absolute discrimination is moderate (Gini 0.2315). For
  unsecured US consumer credit this is realistic, but it is not a strong model — and it is
  *weaker* than the censored window implied.

### 3.4 Benchmark: how much of the ranking comes from the lender?

| Feature set | AUC | Gini |
|---|---|---|
| Without `grade`/`int_rate` (champion random_forest) | 0.6157 | 0.2315 |
| With `grade`/`int_rate` (plain logit) | **0.6700** | **0.3399** |

**Finding P-2 (High).** A plain logistic regression allowed to see the lender's own
`grade` and `int_rate` outperforms our best model built from application data by
**+0.0543 AUC / +0.1173 Gini**. Part of the discrimination a deployed scorecard would
achieve is therefore inherited from the lender's risk engine rather than learned here. The
claim "the model keeps almost all the discriminatory power without the lender's variables"
is **not** supported by the numbers and is not made anywhere in this repository.

### 3.5 Stability

* Score PSI (train vs OOT): **0.0089** — stable.
* Every kept variable is "stable": largest `purpose` 0.0820, `dti` 0.0565,
  `credit_history_months` 0.0476, `revol_util` 0.0332. **Finding P-3 (Low):** no variable
  breaches the "monitor" threshold on the mature window; population shift is not an issue
  in this window, though `purpose` is the one to watch.

### 3.6 Calibration

Isotonic calibrator fitted on the training window, applied out of time (monotone, so
AUC/Gini/KS are unchanged):

| Basis (OOT) | Mean predicted | Observed | Calibration-in-the-large | Slope | Brier |
|---|---|---|---|---|---|
| raw champion | 0.1213 | 0.1447 | 0.2034 | **0.9429** | **0.12176** |
| calibrated | 0.1196 | 0.1447 | 0.2194 | 0.3864 | 0.12370 |

**Finding P-4 (Medium).** The model **under-predicts the 2014–2015 default rate**
(12.13 % predicted vs 14.47 % observed); the train-window rate was 12.36 %, so a level
shift explains part of it. The in-sample calibrator does **not** repair the gap out of
time: its Brier is worse (0.12370 vs 0.12176) and its slope collapses to **0.3864** (vs
0.9429 raw). EL is therefore reported under both bases (§5).

### 3.7 Cutoff / confusion matrix (OOT)

Cutoff = **0.129212**, KS-optimal on the **training** window, applied out of time:

| TP | FP | FN | TN | Precision | Recall | Approval rate | Bad rate approved |
|---|---|---|---|---|---|---|---|
| 12,800 | 52,326 | 11,154 | 89,317 | 0.1965 | 0.5344 | 85.53 % | **36.94 %** |

**Finding P-5 (Low).** This cutoff is a *statistical* default, not an economic optimum: the
36.94 % bad rate among approved loans is high, and a business cutoff (rate/return
optimisation) would be set elsewhere. Any decision use requires a proper cost matrix.

### 3.8 Interpretability

SHAP (champion, `pd_shap_importance.csv`) versus logit coefficients
(`pd_shap_vs_logit.csv`). Top SHAP drivers: `purpose` (0.2162), `annual_inc` (0.2040),
`revol_util` (0.1728), `installment_to_income` (0.1693), `dti` (0.1379).

**Finding P-6 (Low).** The two views are broadly consistent; `purpose` ranks first by SHAP
though it is mid-pack by IV, which reflects interactions the marginal IV cannot see.
Reviewers should treat SHAP importance as interaction-aware and IV/coefficients as
marginal — they are not interchangeable.

---

## 4. LGD / EAD

Sample: defaults in train **7,641**; defaults in OOT **23,953** (the mature OOT window has
*more* defaults than the train window because 2014–2015 were worse vintages).

| | Train | OOT |
|---|---|---|
| Realised LGD (mean) | 0.8809 | **0.8691** |
| Median LGD (OOT) | | **0.8964** |

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **gamma_glm (chosen, lower MAE)** | **0.0996** | 0.1550 | **−0.0846** |
| lightgbm | 0.1027 | **0.1532** | **−0.0586** |

**Finding L-1 (High).** Both out-of-time R² are **negative** — neither model beats
predicting the mean. The tree has the marginally better (still negative) R² and the Gamma
GLM the lower MAE; neither ordering changes the conclusion. The LGD model is **not usable
for individual loan predictions**; its only defensible use is as a portfolio/grade-level
average.

**Finding L-2 (High).** Observed LGD is nearly degenerate (mean 0.8691, median 0.8964),
so there is very little cross-sectional signal to learn. Mean predicted LGD by grade is
almost flat (0.8634 for A → 0.8366 for G), which is consistent with the realised figure
but confirms the model carries little grade-level differentiation.

---

## 5. Expected loss

OOT portfolio (primary window): EAD **$2,104,710,850**; mean PD 0.1196 (calibrated) /
0.1213 (raw); mean LGD 0.8649.

| PD basis | Expected loss | EL / EAD |
|---|---|---|
| isotonic-calibrated champion PD (primary) | **$207,232,901** | **0.0985** |
| raw champion PD (secondary) | $214,069,395 | 0.1017 |

By grade (`el_by_grade`): EL rate rises monotonically **0.0600 (A) → 0.2167 (G)** while
mean LGD stays almost flat (0.8634 → 0.8366), i.e. **the EL ranking is essentially a PD
ranking** — consistent with L-1/L-2.

**Finding E-1 (Medium).** A 9.85 % EL rate is high in absolute terms and is a direct
consequence of the ~14 % default rate in this (resolved-only) sample; it should not be
read as the EL of a fresh, still-performing book.

---

## 6. Segmentation

k chosen on **silhouette *and* bootstrap stability** (`clustering.select_k`), not on a
bare grid boundary. See `clustering_k_selection.csv`:

| k | Inertia | Silhouette | Stability (ARI) | Largest cluster share | Degenerate clusters |
|---|---|---|---|---|---|
| **2** | **196,134.22** | **0.2185** | **0.9629** | 0.6860 | 0 |
| 3 | 168,523.21 | 0.1618 | 0.9778 | 0.4642 | 0 |
| 4 | 155,299.14 | 0.1497 | 0.9208 | 0.3629 | 0 |
| 5 | 146,148.44 | 0.1558 | 0.9443 | 0.3403 | 0 |
| 6 | 138,386.10 | 0.1377 | 0.8974 | 0.2903 | 0 |
| 7 | 132,406.86 | 0.1154 | 0.8847 | 0.2008 | 0 |
| 8 | 127,879.34 | 0.1165 | 0.8942 | 0.1895 | 0 |

Segment profiles:

| Segment | Share | Default rate | Loan amount | Annual income | DTI | History (months) |
|---|---|---|---|---|---|---|
| 0 | 31.21 % | 0.1909 | $21,196 | $112,013 | 20.17 | 238.4 |
| 1 | 68.79 % | 0.2040 | $11,355 | $60,109 | 17.42 | 175.6 |

**Finding C-1 (Medium).** After winsorisation no segment is degenerate (0 flagged). The two
segments differ mainly in **size, income and loan amount**, with a default-rate spread of
only **1.31 pp** (0.1909 vs 0.2040). They are descriptive customer groups, **not** a
risk-rating substitute.

**Finding C-2 (Medium).** Silhouette monotonically favours the smallest k tested. The
selection rule (`0.6·silhouette + 0.4·stability`, boundary kept only if no interior k comes
within 5 %) returns **k = 2** honestly: the combined score for k = 2 (≈ 0.94) is far above
the best interior candidate k = 3 (≈ 0.67), so the boundary is not merely an artefact.
Stability (ARI 0.9629) confirms the split is reproducible. It remains a *metric-driven*
choice, not an obviously "business-correct" one.

**Finding C-3 (Low).** Ward hierarchical clustering on a 5,000-loan subsample recovers a
comparable two-group structure (39.36 % / 60.64 % with default rates 0.156 / 0.219), which
corroborates the k-means split at sample scale.

---

## 7. Engineering validation performed during this work

| Issue found | Root cause | Resolution |
|---|---|---|
| Right-censoring biased the OOT metrics | The OOT window (2017–2018) was only 38.17 % / 11.37 % resolved. | Validation split derived from `(year, term)` resolved shares: primary = cells ≥ 90 % resolved (train 2010–2013, test 2014–2015, term 36); censored window kept as a labelled secondary comparison. |
| `ValueError: Number of labels is 1` in the silhouette step | sklearn's `sample_size=` draws a plain random subsample; with very unbalanced clusters it can contain a single label. | Winsorise at the 1st/99th percentile before standardising; compute the silhouette on a **stratified** subsample containing every cluster and return `NaN` instead of raising. |
| k selected on a bare grid boundary | Maximum-silhouette selection returned k = 2 (the first k tested) with no evidence it was stable. | `select_k` scores `0.6·silhouette + 0.4·bootstrap-ARI` and rejects a boundary winner unless no interior k is within 5 %; stability is reported per k. |
| `TypeError: Must provide 'func' or tuples` in `profile_segments` | Mixing named-tuple aggregation with plain `col="mean"` kwargs in `DataFrame.agg`. | Rebuilt the aggregation with `groupby()[cols].mean()` + `insert`. |
| Hierarchical profile length mismatch | Linkage computed on a 5,000-loan subsample while labels were applied to 120,000 rows. | Subsample once and profile the *same* rows. |
| Scorecard points all zero; SHAP/coefficient merge broken | `reset_index(names=…)` on an ndarray gave positional integers as variable names. | Use `res.model.exog_names` / `res.params.index`. |
| Importance sampling appeared to increase variance | The reported SE used `se_plain/√ESS`, which is valid for weighted *means* but not for a weighted empirical **quantile**; it inflated the IS error (2.96× when the true ratio was 0.21×) and invited a wrong "granular ⇒ idiosyncratic" explanation. | Replaced with a batched-quantile SE plus a replication-based RMSE study; added a closed-form Vasicek unbiasedness unit test. IS now measures a **4.36×** RMSE reduction; the negative finding is withdrawn (`Finding M-3`). |
| Degenerate risk from `MiniBatchKMeans` as default | Exact `KMeans` measured better *and* faster on the 120k sample (inertia 1,174,937 / silhouette 0.2200 vs 1,184,469 / 0.1889). | MiniBatch kept as an opt-in fallback above 250k rows; exact KMeans is the default for both the study and the shipped fit. |

Test suite: **68 unit tests**, all passing on synthetic data; `ruff check src tests` clean;
CI runs lint + tests on Python 3.10 and 3.11 without the dataset. A dedicated
`tests/test_leakage.py` enforces that the split never mixes train/test rows and that
winsorisation bounds, WoE encoders and the calibrator are fitted on train only.

---

## 8. Monte Carlo portfolio loss simulation

Portfolio: **69 buckets** (grade × PD decile), total EAD **$2,104,710,850**,
EAD-weighted PD **0.1138**, EAD-weighted LGD **0.8593**. PDs are the out-of-time
calibrated champion scores; LGDs are the Gamma GLM output per loan.

Base case, 100,000 paths, antithetic variates, ρ = 0.12:

| Metric | Amount | % of EAD |
|---|---|---|
| Expected loss | $207,271,841 | 9.85 % |
| Unexpected loss (std) | $114,905,715 | 5.46 % |
| VaR 99 % | $564,131,680 | 26.80 % |
| ES 99 % | $638,449,813 | 30.33 % |
| VaR 99.9 % | $735,620,950 | 34.95 % |
| ES 99.9 % | $807,437,113 | 38.36 % |
| Economic capital (99 %) | $356,859,839 | 16.96 % |
| Economic capital (99.9 %) | $528,349,109 | 25.10 % |

### 8.1 Asset correlation (the dominant assumption)

The Basel IRB PD-dependent retail formula, evaluated at the portfolio PD (0.1138),
implies ρ = **0.0324**. The base case is ρ = 0.12. Both are inside the retail band; the
grid makes the choice transparent:

| ρ | EL (% EAD) | UL (% EAD) | VaR 99 % | VaR 99.9 % |
|---|---|---|---|---|
| 0.02 | 9.84 % | 2.17 % | 15.60 % | **17.97 %** |
| 0.0324 (Basel implied) | 9.84 % | 2.77 % | 17.46 % | 20.69 % |
| 0.05 | 9.84 % | 3.45 % | 19.68 % | 24.02 % |
| 0.10 | 9.85 % | 4.96 % | 24.93 % | 32.12 % |
| **0.12 (base)** | **9.85 %** | **5.46 %** | **26.80 %** | **34.95 %** |
| 0.15 | 9.85 % | 6.16 % | 29.48 % | **39.05 %** |

**Finding M-1 (High).** EL is essentially invariant to ρ, while VaR 99.9 % moves from
**17.97 % to 39.05 % of EAD** — more than a doubling — purely from the correlation
assumption. Any capital number from this model is only meaningful together with its ρ;
the repository therefore never quotes one in isolation.

### 8.2 Convergence and variance reduction

Convergence study saved to `sim_convergence.csv` (plain) and
`sim_convergence_antithetic.csv`; the loss distribution stabilises as paths grow.

| Paths | SE (plain) | SE (antithetic) | SE reduction |
|---|---|---|---|
| 100,000 | 364,043 | 363,362 | **+0.19 %** |

**Finding M-2 (Medium).** **Antithetic variates give no meaningful variance reduction here**
(mean SE falls by only 0.19 % at 100,000 paths). With bucketed binomial draws the systematic
factor is already averaged across 69 buckets, so pairing paths adds almost nothing. The
result is reported rather than hidden.

**Finding M-3 (Medium).** **Importance sampling on the systematic factor works, and it was
verified rather than asserted.** Shifting `Z` towards the loss tail and reweighting by the
likelihood ratio `w(z) = φ(z)/φ(z + shift)` yields an unbiased 99.9 % VaR: a unit test
(`test_importance_sampling_is_unbiased_for_the_closed_form_var`) confirms the reweighted
estimator converges to the closed-form Vasicek VaR of an infinitely granular homogeneous
portfolio, and the factor quantile matching the 99.9 % level (`|Φ⁻¹(0.001)| ≈ 3.09`) is
reported alongside each run. On the real 69-bucket portfolio the single-run estimates are
IS $733.5M vs naive $741.5M (shift 1.5), and — using a **replication-based RMSE study** rather
than the earlier, invalid `se/√ESS` scaling — IS reduces the tail-VaR RMSE by **4.36×** at
equal paths (RMSE 3.23M vs 14.10M over 40 replications); the batched standard error agrees
(SE ratio **0.21×**). The real cost is the effective sample size: it falls to **2,276 of
20,000 paths (11.4 %)**, and pushing the shift to the factor quantile (≈ 3.09) collapses it to
≈ 23 and biases the weighted quantile **upward**, which is why the shift is tuned to 1.5. The
shift curve is in `sim_importance_sampling_sensitivity.csv` and the RMSE study in
`sim_importance_sampling_variance_reduction.csv`.

> **Correction to an earlier draft.** A previous version of this report claimed IS *increased*
> variance by 2.96× and attributed that to the tail being idiosyncratic (a granularity story).
> Both were wrong: the 2.96× figure came from an inapplicable `se/√ESS` formula, and the
> granularity explanation does not hold — for a granular portfolio the tail *is* driven by the
> systematic factor, which is exactly what the shift exploits. The estimator and its error
> metric were fixed, a closed-form unit test was added, and the negative finding was withdrawn.

### 8.3 Stress scenarios (ρ = 0.12)

| Scenario | EAD-weighted PD | Mean LGD | EL | EL % EAD | VaR 99.9 % |
|---|---|---|---|---|---|
| baseline | 0.1138 | 0.8593 | $207,271,841 | 9.85 % | $735,620,950 |
| PD +30 % | 0.1479 | 0.8593 | $269,425,581 | 12.80 % | $843,491,761 |
| PD +50 % | 0.1707 | 0.8593 | $310,850,307 | 14.77 % | $906,680,785 |
| LGD +10 % | 0.1138 | 0.9452 | $227,999,026 | 10.83 % | $809,183,045 |
| PD +50 % & LGD +10 % | 0.1707 | 0.9452 | $341,935,338 | 16.25 % | $997,348,863 |

**Finding M-4 (Low).** Stress is close to linear in PD and LGD (EL = PD × LGD × EAD), as
expected. That is a property of the Vasicek formulation, worth stating explicitly to avoid
over-reading the stress grid.

---

## 9. Summary of findings

| ID | Severity | Finding |
|---|---|---|
| **C-0** | **High** | Testing on the censored 2017–2018 window inflates AUC by +0.0579 (0.6736 vs 0.6157). The primary split now removes this by construction. |
| **P-2** | **High** | Lender `grade`/`int_rate` add +0.0543 AUC / +0.1173 Gini over our best application-data model. |
| **L-1 / L-2** | **High** | LGD R² is negative for both models; LGD is usable only as a grade/portfolio average. |
| **M-1** | **High** | VaR 99.9 % spans 17.97 %–39.05 % of EAD across the ρ grid; ρ dominates the capital figure. |
| **E-1** | Medium | EL of 9.85 % of EAD reflects a high (~14 %) default rate in a resolved-only sample. |
| **P-4** | Medium | Out-of-time under-prediction (12.13 % vs 14.47 %); in-sample calibrator does not repair it (Brier worsens, slope 0.39). |
| **P-1** | Medium | Absolute discrimination is moderate (Gini 0.2315); trees show real in/out decay. |
| **M-2** | Medium | Antithetic variates give no meaningful reduction (+0.19 % SE) in this bucketed design. |
| **M-3** | Low | Importance sampling **does** reduce tail-VaR variance (RMSE 4.36× smaller) but cuts the effective sample size to 11.4 %; shift is tuned, not arbitrary. |
| **C-1 / C-2** | Medium | Segments differ by only 1.31 pp in default rate; k = 2 is metric-driven though now stability-backed. |
| **P-3** | Low | All variable PSIs stable on the mature window (largest `purpose` 0.0820). |
| **P-5** | Low | Cutoff is statistical, not economic (36.94 % bad rate among approved). |
| **S-1** | Low | Accepted-only sample: reject inference is out of scope. |
| **Q-1** | Low | Outliers handled by winsorisation rather than removal (conservative). |

## 10. Recommendation

1. **Use** the logistic scorecard (`logit_l2`) as the production model for discrimination,
   with the random-forest score as a monitored challenger; the AUC difference
   (0.6113 vs 0.6157) does not justify the loss of explainability.
2. **Do not use** the LGD model at loan level; use the realised/grade-level LGD
   (≈ 0.84–0.87) and revisit with a two-stage (recovery) model on a mature vintage only.
3. **Do not use** the SARIMAX forecast for planning; if vintage trend monitoring is
   required, restrict it to vintages with a resolved share above ~90 % (≤ 2015 here).
4. **Report every capital figure with its ρ**, using the Basel implied ρ (0.0324 at this
   PD) as the reference and 0.12–0.15 as a stress band.
5. **Before any decision use**, re-estimate the cutoff against an explicit cost/return
   matrix and validate on a vintage that is fully matured.
