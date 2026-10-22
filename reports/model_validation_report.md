# Model Validation Report — Lending Club Credit Risk Pipeline

**Scope.** Independent validation of the PD scorecard/challengers, the LGD/EAD/EL model,
the customer segmentation and the Monte Carlo portfolio loss simulation produced by
`python -m src.pipeline all`.

**Evidence base.** Every figure in this report is read from
[`reports/results/summary.json`](results/summary.json) and the CSV artifacts in
`reports/results/`. Nothing is quoted from memory or from a different run.
Run seed: **42**; full pipeline wall time **2850.4 s**.

**Verdict in one paragraph.** The pipeline is reproducible and leakage-controlled, and the
validation design **removes right-censoring from the headline metrics by construction**:
the primary out-of-time window is built from `(year, term)` cells that are ≥ 90 % resolved
(train 2010–2013 / test 2014–2015, 36-month loans only). On that honest window the PD
champion (`lightgbm`, tuned by temporal CV inside the train window only) reaches AUC 0.6572
(Gini 0.3144), and a LightGBM given the lender's own `grade`/`int_rate` on the same wide
feature set reaches AUC 0.6799 (Gini 0.3599). Testing on the old, censored 2017–2018 window
would have reported AUC 0.7152 — **the censored window inflates AUC by +0.0580**, which is
why it is now only a labeled secondary analysis. The LGD model still has a **negative**
out-of-time R² (all three candidates). Probability calibration, honestly, **does not improve
the test-window Brier here**. The portfolio forecast is weak and loses to naive benchmarks.
The Monte Carlo layer is sound, its dominant assumption (asset correlation) is disclosed and
swept, and its importance-sampling estimator is **verified against a closed-form benchmark**
and shown to cut the tail-VaR RMSE by **5.11×** at equal paths.

> **What changed in this revision.** This report was updated to reflect a rework that grew the
> modeling sample to the full resolved population, moved the challengers onto the wide
> leakage-free feature set, added temporal LightGBM tuning, a four-way calibration study, a
> two-part LGD model, a Gaussian-mixture segmentation alternative, a benchmarked forecast and
> stratified sampling. Numbers that this rework *did not* improve (calibration, the forecast,
> antithetic/stratified variance reduction) are reported as null results rather than removed.

---

## 1. Population, sample and target

| Item | Value | Source |
|---|---|---|
| Accepted loans (population) | 2,260,668 | `summary.json → data.population.n_accepted` |
| Resolved loans | 1,345,350 | `data.population.n_resolved` |
| — Fully Paid | 1,076,751 | `data.population.n_fully_paid` |
| — Charged Off | 268,559 | `data.population.n_charged_off` |
| — Default | 40 | `data.population.n_default_status` |
| Modeling sample | 1,345,350 (the full resolved population) | `data.n_rows` |
| Sample default rate | 19.96 % (268,599 defaults) | `data.default_rate`, `quality.n_defaults` |
| Vintages covered | 2007 – 2018 | `data.years` |

**Target definition.** `default_flag = 1` for `loan_status in ('Charged Off','Default')`,
`0` for `Fully Paid`, undefined otherwise. Only resolved loans enter the sample, so no
loan is labeled with a future-unknown state.

**Assessment.** The definition is consistent, reproducible and standard for this dataset.
The shipped `modeling.sample_size` (1,400,000) exceeds the resolved population, so the
modeling sample *is* the full 1,345,350 resolved loans and the sample default rate (19.96 %)
equals the population figure exactly — there is no sampling distortion to report.

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
* Winsorization bounds, encoders and WoE maps are fitted on the **training window only**.

**Finding Q-1 (accepted).** Outliers are extreme but handled by winsorization rather than
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
| Train | 2010 – 2013, term 36 m | 166,459 | 12.39 % | 99.59 % |
| OOT test | 2014 – 2015, term 36 m | 445,596 | 14.46 % | 99.97 % |

The 60-month cells of 2014–2015 are only 82.85 % / 67.08 % resolved and 2009 (36 m) is
89.30 % resolved, so all are excluded from the primary split.

**Secondary (immature) window.** The exact same code path is re-run on the old
issue-year ≥ 2017 window so the censoring effect is *measured*:

| | Window | Loans | Default rate | Resolved share |
|---|---|---|---|---|
| Train | issue year < 2017 | — | — | — |
| OOT test | 2017 – 2018 | 225,639 | 21.29 % | 40.12 % / 11.97 % (36 m) |

**Finding C-0 (High — the reason for the redesign).** Training/testing on the mature
window versus the censored window changes the champion AUC from **0.7152** to **0.6572**,
i.e. **the censored window inflates discounting power by +0.0580 AUC**. All headline
metrics in this report are stated on the mature window; the censored figure is quoted only
as the secondary comparison in §3.3. Note that the *default-rate* bias is **not
one-directional**: 2017 (40.12 % resolved) shows a higher observed rate (23.13 %) while 2018
(11.97 % resolved) shows a lower one (15.76 %), because early defaults and early prepayments
are both over-represented and their mix differs by vintage (per-vintage table in
`summary.json → eda.resolution_by_year_term`). What leaks into the score is resolution
*timing*, not the level of the rate.

### 3.2 Variable selection

30 variables binned with a WoE/IV procedure on the **training window only**; **7 kept**
after IV ranking, correlation screening and VIF: `annual_inc`, `installment_to_income`,
`dti`, `revol_util`, `home_ownership`, `inq_last_6mths`, `purpose`.

Strongest IV on the mature training window: `annual_inc` 0.0955, `installment_to_income`
0.0657, then `loan_amnt_to_income` 0.0445, `dti` 0.0410, `revol_util` 0.0383. **Every
variable is in the "weak" IV band** — an honest, early signal that the achievable
discrimination is limited, and weaker than the censored window would have suggested.

Note the deliberate asymmetry: the **challengers** (§3.3) do not use these 7 WoE inputs;
they consume the **full leakage-free origination set** (30 variables: 22 numeric, 8
categorical) via a `ColumnTransformer`, so the model comparison measures *modeling
capacity* rather than the scorecard's variable selection.

### 3.3 Out-of-time discrimination (primary window, 2014–2015)

| Model | AUC | Gini | KS |
|---|---|---|---|
| **lightgbm (champion, temporally tuned + monotone)** | **0.6572** | **0.3144** | **0.2274** |
| random_forest | 0.6468 | 0.2936 | 0.2088 |
| probit | 0.6219 | 0.2438 | 0.1770 |
| logit_l2 (Ridge, scorecard) | 0.6218 | 0.2436 | 0.1766 |
| logit | 0.6217 | 0.2435 | 0.1764 |
| logit_l1 (Lasso) | 0.6217 | 0.2435 | 0.1764 |
| decision_tree | 0.5509 | 0.1018 | 0.1018 |

The LightGBM hyper-parameters were chosen by **temporal cross-validation inside the train
window only** (fit 2010–2012, validate 2013; best valid AUC 0.6447). The 2014–2015 test
window was never touched during tuning.

**Comparison against the censored secondary window** (`pd_window_comparison.csv`):

| Window | OOT years | Champion | AUC | Gini | KS | Calibration gap | Score PSI |
|---|---|---|---|---|---|---|---|
| **Mature (primary)** | 2014–2015 | lightgbm | **0.6572** | 0.3144 | 0.2274 | −0.0229 | 0.0145 |
| Immature (secondary) | 2017–2018 | lightgbm | **0.7152** | 0.4303 | 0.3129 | −0.0243 | 0.0028 |

**Assessment.**

* On the honest window the champion is LightGBM; it beats the interpretable logit scorecard
  by **+0.0354 AUC** — a real but modest margin, because the challengers now see the wide
  feature set. For production the logit scorecard remains the more defensible choice on
  explainability grounds; the tree is a monitoring challenger.
* **Finding P-1 (Medium).** Absolute discrimination is moderate (Gini 0.3144) — better than
  the earlier baseline (0.2315) but still modest for unsecured US consumer credit, and it is
  *weaker* than the censored window implies.

### 3.4 Benchmark: how much of the ranking comes from the lender?

| Feature set | AUC | Gini |
|---|---|---|
| Without `grade`/`int_rate` (champion lightgbm) | 0.6572 | 0.3144 |
| With `grade`/`int_rate` (**same wide set + same LightGBM recipe**) | **0.6799** | **0.3599** |

**Finding P-2 (High).** The benchmark was re-run *fairly* — the same wide feature set and
the same LightGBM recipe, plus the lender's own `grade`/`int_rate` — so the comparison is
apples-to-apples. It still outperforms our best model built from application data by
**+0.0227 AUC / +0.0455 Gini**. Part of the discrimination a deployed scorecard would achieve
is therefore inherited from the lender's risk engine rather than learned here. The claim
"the model keeps almost all the discriminatory power without the lender's variables" is
**not** supported by the numbers and is not made anywhere in this repository.

### 3.5 Stability

* Score PSI (train vs OOT): **0.0145** — stable (well below the 0.10 "monitor" level).
* **Finding P-3 (Low):** no variable breaches the "monitor" threshold on the mature window;
  population shift is not an issue in this window.

### 3.6 Calibration — the honest result is that it does not help here

Four calibrators were compared in `src/calibration.py`. Every calibrator is fit on
**out-of-sample** scores from a later temporal slice (model trained 2010–2012, scoring 2013),
and the method is selected on a separate inner slice (2010–2011 model scoring 2012) — never
on the test window. The `intercept` calibrator genuinely fixes the slope at 1.

| Method | val Brier | val CITL | test Brier | test CITL | test slope |
|---|---|---|---|---|---|
| none | 0.1159 | 0.2111 | **0.11950** | 0.1844 | 0.9087 |
| intercept | 0.1160 | 0.2284 | 0.11960 | 0.2021 | 0.9088 |
| platt | 0.1153 | 0.2175 | 0.11970 | 0.1982 | 1.1095 |
| **isotonic (selected on validation)** | **0.1152** | 0.2333 | 0.11970 | 0.1986 | 1.0765 |

**Finding P-4 (Medium).** Validation selects **isotonic** (best val Brier 0.1152), but on the
test window the **raw** score has the lowest Brier (0.11950 vs 0.11970) and the raw model is
already close to calibrated out of time (slope 0.9087, CITL 0.184). The calibration choice
made on validation therefore **does not improve the test-window Brier here**. The model
still **under-predicts the 2014–2015 default rate** by about **2.3 pp** (isotonic mean
predicted 0.1218 vs observed 0.1446); the train-window rate was 12.39 %, so a level shift
explains part of it. This is reported as a null result rather than hidden; the calibrated
PDs are still used downstream for EL/simulation because that is the operationally correct
pipeline, but no improvement is claimed.

### 3.7 Cutoff / confusion matrix (OOT)

Cutoff = **0.136088**, KS-optimal on the **training** window, applied out of time:

| TP | FP | FN | TN | Precision | Recall | Approval rate | Bad rate approved |
|---|---|---|---|---|---|---|---|
| 34,960 | 123,446 | 29,487 | 257,703 | 0.2207 | 0.5425 | 85.54 % | **32.39 %** |

**Finding P-5 (Low).** This cutoff is a *statistical* default, not an economic optimum: the
32.39 % bad rate among approved loans is high, and a business cutoff (rate/return
optimization) would be set elsewhere. Any decision use requires a proper cost matrix.

### 3.8 Interpretability

SHAP (champion) versus logit coefficients. Top SHAP drivers: `revol_util` (0.2064),
`annual_inc` (0.1779), `inq_last_6mths` (0.1625), `installment_to_income` (0.1577),
`purpose` (0.1403).

**Finding P-6 (Low).** The two views are broadly consistent; `revol_util` ranks first by
SHAP though it is mid-pack by IV, which reflects interactions the marginal IV cannot see.
Reviewers should treat SHAP importance as interaction-aware and IV/coefficients as
marginal — they are not interchangeable.

---

## 4. LGD / EAD

Sample: defaults in train **20,622**; defaults in OOT **64,445** (the mature OOT window has
*more* defaults than the train window because 2014–2015 were worse vintages).

| | Train | OOT |
|---|---|---|
| Realised LGD (mean) | 0.8801 | **0.8678** |
| Median LGD (OOT) | | **0.8964** |
| Boundary mass (`LGD = 1`) | 5.20 % | **26.86 %** |

Realised LGD piles up at 0 and 1, and the `LGD = 1` mass **shifts sharply across vintages**
(5.20 % train vs 26.86 % OOT), so a two-part model (logistic boundary + fractional severity)
is fitted alongside the Gamma GLM and a LightGBM regressor. The champion is chosen on
grade-level calibration first, then loan-level MAE.

| Model | MAE | RMSE | R² | Grade-gap MAE | Portfolio gap |
|---|---|---|---|---|---|
| gamma_glm | 0.1004 | 0.1572 | **−0.0758** | 0.00504 | −0.0011 |
| **lightgbm (champion)** | 0.1015 | **0.1538** | **−0.0294** | **0.00245** | +0.0009 |
| two_part | **0.1000** | 0.1559 | −0.0578 | 0.00637 | +0.0021 |

**Finding L-1 (High).** All three out-of-time R² are **negative** — none beats predicting
the mean. The two-part model has the best loan-level MAE, but the plain LightGBM regressor
wins on both RMSE and grade-level calibration, so it is the champion. The LGD model is
**not usable for individual loan predictions**; its only defensible use is as a
portfolio/grade-level average.

**Finding L-2 (High).** Observed LGD is nearly degenerate (mean 0.8678, median 0.8964), so
there is very little cross-sectional signal to learn. Mean predicted LGD by grade is almost
flat (0.8707 for A → 0.8571 for F), which is consistent with the realised figure but
confirms the model carries little grade-level differentiation.

---

## 5. Expected loss

OOT portfolio (primary window): EAD **$5,670,341,550**; mean PD 0.1218 (calibrated) /
0.1233 (raw); mean LGD 0.8663.

| PD basis | Expected loss | EL / EAD |
|---|---|---|
| isotonic-calibrated champion PD (primary) | **$574,794,646** | **0.1014** |
| raw champion PD (secondary) | $579,736,269 | 0.1022 |

By grade (`el_by_grade`): EL rate rises monotonically **0.0657 (A) → 0.1874 (G)** while
mean LGD stays almost flat (0.8672 → 0.8690), i.e. **the EL ranking is essentially a PD
ranking** — consistent with L-1/L-2.

**Finding E-1 (Medium).** A 10.14 % EL rate is high in absolute terms and is a direct
consequence of the ~14 % default rate in this (resolved-only) sample; it should not be
read as the EL of a fresh, still-performing book.

---

## 6. Segmentation

k chosen on **silhouette *and* bootstrap stability** (`clustering.select_k`), not on a
bare grid boundary. See `clustering_k_selection.csv`:

| k | Inertia | Silhouette | Stability (ARI) | Largest cluster share | Degenerate clusters |
|---|---|---|---|---|---|
| **2** | **192,959.17** | **0.1840** | **0.9566** | 0.5523 | 0 |
| 3 | 168,761.02 | 0.1538 | 0.9105 | 0.3694 | 0 |
| 4 | 155,626.46 | 0.1378 | 0.9198 | 0.2834 | 0 |
| 5 | 144,820.00 | 0.1414 | 0.9454 | 0.2670 | 0 |
| 6 | 137,693.87 | 0.1235 | 0.7325 | 0.2457 | 0 |
| 7 | 131,071.14 | 0.1239 | 0.9134 | 0.1948 | 0 |
| 8 | 126,885.01 | 0.1135 | 0.7638 | 0.1613 | 0 |

k-means segment profiles:

| Segment | Share | Default rate | Loan amount | Annual income | DTI | History (months) |
|---|---|---|---|---|---|---|
| 0 | 55.08 % | 0.2021 | $18,511 | $92,169 | 20.27 | 220.2 |
| 1 | 44.92 % | 0.1967 | $9,403 | $56,723 | 15.84 | 164.3 |

Gaussian-mixture segment profiles (`GMMSegmenter`, soft/ellipsoidal clusters on the same
`log1p`-transformed features):

| Segment | Share | Default rate | Loan amount | Annual income | DTI | History (months) |
|---|---|---|---|---|---|---|
| 0 | 20.60 % | **0.1498** | $10,703 | $91,835 | 15.86 | 198.3 |
| 1 | 79.40 % | **0.2126** | $15,385 | $72,202 | 18.91 | 194.3 |

**Finding C-1 (Medium).** After winsorization no segment is degenerate (0 flagged). The
**k-means** segments differ mainly in **size, income and loan amount**, with a default-rate
spread of only **0.54 pp** (0.1967 vs 0.2021) — descriptive customer groups, **not** a
risk-rating substitute.

**Finding C-2 (High).** The **Gaussian-mixture** alternative separates the default rate
materially better: **6.28 pp** (14.98 % vs 21.26 %) with bootstrap ARI **0.978**. It is the
first segmentation in this project with a usable default-rate gap, and it is reported as the
better behavioral segmentation. Even so, both remain descriptive groupings, not calibrated
ratings.

**Finding C-3 (Medium).** Silhouette monotonically favours the smallest k tested. The
selection rule (`0.6·silhouette + 0.4·stability`, boundary kept only if no interior k comes
within 5 %) returns **k = 2** honestly: the combined score for k = 2 is well above the best
interior candidate k = 3, so the boundary is not merely an artifact. Stability (ARI 0.9566)
confirms the split is reproducible. It remains a *metric-driven* choice, not an obviously
"business-correct" one.

**Finding C-4 (Low).** Ward hierarchical clustering on a subsample recovers a comparable
two-group structure (29.92 % / 70.08 % with default rates 0.1564 / 0.2041), which
corroborates a two-group split at sample scale.

---

## 7. Engineering validation performed during this work

| Issue found | Root cause | Resolution |
|---|---|---|
| Right-censoring biased the OOT metrics | The OOT window (2017–2018) was only 38.17 % / 11.37 % resolved. | Validation split derived from `(year, term)` resolved shares: primary = cells ≥ 90 % resolved (train 2010–2013, test 2014–2015, term 36); censored window kept as a labeled secondary comparison. |
| `ValueError: Number of labels is 1` in the silhouette step | sklearn's `sample_size=` draws a plain random subsample; with very unbalanced clusters it can contain a single label. | Winsorize at the 1st/99th percentile before standardising; compute the silhouette on a **stratified** subsample containing every cluster and return `NaN` instead of raising. |
| k selected on a bare grid boundary | Maximum-silhouette selection returned k = 2 (the first k tested) with no evidence it was stable. | `select_k` scores `0.6·silhouette + 0.4·bootstrap-ARI` and rejects a boundary winner unless no interior k is within 5 %; stability is reported per k. |
| `TypeError: Must provide 'func' or tuples` in `profile_segments` | Mixing named-tuple aggregation with plain `col="mean"` kwargs in `DataFrame.agg`. | Rebuilt the aggregation with `groupby()[cols].mean()` + `insert`. |
| Hierarchical profile length mismatch | Linkage computed on a 5,000-loan subsample while labels were applied to 120,000 rows. | Subsample once and profile the *same* rows. |
| Scorecard points all zero; SHAP/coefficient merge broken | `reset_index(names=…)` on an ndarray gave positional integers as variable names. | Use `res.model.exog_names` / `res.params.index`. |
| Importance sampling appeared to increase variance | The reported SE used `se_plain/√ESS`, which is valid for weighted *means* but not for a weighted empirical **quantile**; it inflated the IS error and invited a wrong "granular ⇒ idiosyncratic" explanation. | Replaced with a batched-quantile SE plus a replication-based RMSE study; added a closed-form Vasicek unbiasedness unit test. IS now measures a **5.11×** RMSE reduction. |
| Degenerate risk from `MiniBatchKMeans` as default | Exact `KMeans` measured better *and* faster on the 120k sample. | MiniBatch kept as an opt-in fallback above 250k rows; exact KMeans is the default for both the study and the shipped fit. |
| SHAP crashed on the champion | The monotone-constrained LightGBM is wrapped in `MonotoneLGBMClassifier`; `TreeExplainer` received the wrapper, not the booster. | `interpret.py` unwraps via `getattr(clf, "model_", clf)` before explainer construction. |
| Monotone constraints broke inside the sklearn pipeline | The constraint vector must match the post-`ColumnTransformer` width, which is only known at fit time. | New `MonotoneLGBMClassifier` resolves the vector at fit time and declares `estimator_type="classifier"` via `__sklearn_tags__`. |
| Silent all-NaN bureau columns in early vintages | Five variables (`tot_coll_amt`, `tot_cur_bal`, `total_rev_hi_lim`, `mths_since_last_major_derog`, `limit_utilisation`) do not exist before 2012, so they are *entirely* missing in the 2010–2011 tuning/calibration slices; `SimpleImputer` dropped them (with a warning) and changed the feature width. | Set `keep_empty_features=True` so an all-missing column is carried as a constant instead of dropped; added a regression test that fits on a slice where a column is all-NaN under `warnings.simplefilter("error")`. |
| Forecast benchmark was silently broken | The vintage series had interior holes (months below the resolved-loan volume threshold), which made MASE `NaN`, Holt-Winters non-convergent and Ljung-Box `NaN`. | New `_contiguous_monthly` keeps the longest gap-free monthly block (85 months) so all four forecast arms are scored on the same grid; ETS/MASE/Ljung-Box now populate. |
| All-NaN warning path / SHAP list-type output | Newer SHAP returns a list of ndarray for binary LightGBM; the code assumed an array. | Output type handled; `shap_top` now populates. |

Test suite: all unit tests pass on synthetic data; `ruff check src tests` clean;
CI runs lint + tests on Python 3.10 and 3.11 without the dataset. The live count is read
from `summary.json` (`test_count`, written by
`python -m src.report_tables --write-test-count` from the pytest collection) and reappears
in the generated README results block, so a stale figure fails
`tests/test_report_integrity.py`. A dedicated
`tests/test_leakage.py` enforces that the split never mixes train/test rows and that
winsorization bounds, WoE encoders and the calibrator are fitted on train only, and
`tests/test_report_integrity.py` fails if the README/`summary.json` numbers drift.

---

## 8. Monte Carlo portfolio loss simulation

Portfolio: **70 buckets** (grade × PD decile), total EAD **$5,670,341,550**,
EAD-weighted PD **0.1172**, EAD-weighted LGD **0.8625**. PDs are the out-of-time
calibrated champion scores; LGDs are the champion LGD model output per loan.

Base case, 100,000 paths, **stratified** sampling of the systematic factor, ρ = 0.12:

| Metric | Amount | % of EAD |
|---|---|---|
| Expected loss | $574,731,161 | 10.14 % |
| Unexpected loss (std) | $333,542,731 | 5.88 % |
| VaR 99 % | $1,617,286,614 | 28.52 % |
| ES 99 % | $1,837,432,176 | 32.40 % |
| VaR 99.9 % | $2,120,320,304 | 37.39 % |
| ES 99.9 % | $2,306,245,549 | 40.67 % |
| Economic capital (99 %) | $1,042,555,453 | 18.39 % |
| Economic capital (99.9 %) | $1,545,589,143 | 27.25 % |

### 8.1 Asset correlation (the dominant assumption)

The Basel IRB PD-dependent retail formula, evaluated at the portfolio PD (0.1172),
implies ρ = **0.0322**. The base case is ρ = 0.12. Both are inside the retail band; the
grid makes the choice transparent:

| ρ | EL (% EAD) | UL (% EAD) | VaR 99 % | VaR 99.9 % |
|---|---|---|---|---|
| 0.02 | 10.14 % | 2.33 % | 16.37 % | **18.92 %** |
| 0.0322 (Basel implied) | 10.14 % | 2.96 % | 18.32 % | 21.80 % |
| 0.05 | 10.14 % | 3.72 % | 20.76 % | 25.53 % |
| 0.10 | 10.14 % | 5.34 % | 26.50 % | 34.20 % |
| **0.12 (base)** | **10.14 %** | **5.88 %** | **28.52 %** | **37.39 %** |
| 0.15 | 10.14 % | 6.64 % | 31.43 % | **41.67 %** |

**Finding M-1 (High).** EL is essentially invariant to ρ, while VaR 99.9 % moves from
**18.92 % to 41.67 % of EAD** — more than a doubling — purely from the correlation
assumption. Any capital number from this model is only meaningful together with its ρ;
the repository therefore never quotes one in isolation.

### 8.2 Convergence and variance reduction

Convergence study saved to `sim_convergence.csv` (plain), `sim_convergence_antithetic.csv`
and `sim_convergence_stratified.csv`; the loss distribution stabilises as paths grow.

| Scheme | SE at 100,000 paths | SE reduction vs plain |
|---|---|---|
| plain | — | — |
| antithetic | — | **0.18 %** |
| stratified | — | **0.50 %** |

**Finding M-2 (Medium).** **Neither antithetic nor stratified sampling gives a meaningful
variance reduction here** (0.18 % and 0.50 % respectively). The reason is structural: with
~70 buckets the idiosyncratic per-bucket default noise dominates the single systematic
factor, so stratifying `Z` barely moves the total-loss variance, and the antithetic pairing
is a no-op on bucketed binomial draws. A replication test confirms stratified sampling
*does* reduce the SE on a homogeneous portfolio where the factor dominates, so the technique
is kept as the variance-reduction method of record — but the practical gain on this portfolio
is small, and this is reported rather than hidden.

**Finding M-3 (Low).** **Importance sampling on the systematic factor works, and it was
verified rather than asserted.** Shifting `Z` towards the loss tail and reweighting by the
likelihood ratio `w(z) = φ(z)/φ(z + shift)` yields an unbiased 99.9 % VaR: a unit test
(`test_importance_sampling_is_unbiased_for_the_closed_form_var`) confirms the reweighted
estimator converges to the closed-form Vasicek VaR of an infinitely granular homogeneous
portfolio, and the factor quantile matching the 99.9 % level (`|Φ⁻¹(0.001)| ≈ 3.09`) is
reported alongside each run. On the real 70-bucket portfolio the single-run estimates are
IS $2,120.4M vs naive $2,144.7M (shift 1.5), and — using a **replication-based RMSE study**
over 40 replications — IS reduces the tail-VaR RMSE by **5.11×** at equal paths (RMSE 9.25M
vs 47.28M); the batched standard error agrees (SE ratio **0.19×**). The real cost is the
effective sample size: it falls to **2,276 of 20,000 paths (11.4 %)**, and pushing the shift
to the factor quantile (≈ 3.09) collapses it further and biases the weighted quantile
**upward**, which is why the shift is tuned to 1.5. The shift curve is in
`sim_importance_sampling_sensitivity.csv` and the RMSE study in
`sim_importance_sampling_variance_reduction.csv`.

### 8.3 Stress scenarios (ρ = 0.12)

| Scenario | EAD-weighted PD | Mean LGD | EL | EL % EAD | VaR 99.9 % |
|---|---|---|---|---|---|
| baseline | 0.1172 | 0.8625 | $575,061,033 | 10.14 % | $2,124,846,785 |
| PD +30 % | 0.1523 | 0.8625 | $747,468,417 | 13.18 % | $2,445,639,962 |
| PD +50 % | 0.1758 | 0.8625 | $862,414,536 | 15.21 % | $2,630,254,724 |
| LGD +10 % | 0.1172 | 0.9488 | $632,567,136 | 11.16 % | $2,337,331,464 |
| PD +50 % & LGD +10 % | 0.1758 | 0.9488 | $948,655,990 | 16.73 % | $2,893,280,197 |

**Finding M-4 (Low).** Stress is close to linear in PD and LGD (EL = PD × LGD × EAD), as
expected. That is a property of the Vasicek formulation, worth stating explicitly to avoid
over-reading the stress grid.

---

## 9. Summary of findings

| ID | Severity | Finding |
|---|---|---|
| **C-0** | **High** | Testing on the censored 2017–2018 window inflates AUC by +0.0580 (0.7152 vs 0.6572). The primary split now removes this by construction. |
| **P-2** | **High** | Lender `grade`/`int_rate` add +0.0227 AUC / +0.0455 Gini over our best application-data model, even when the benchmark gets the same wide feature set and recipe. |
| **L-1 / L-2** | **High** | LGD R² is negative for all three models; the `LGD=1` boundary mass shifts 5.20 % → 26.86 % across windows; LGD is usable only as a grade/portfolio average. |
| **M-1** | **High** | VaR 99.9 % spans 18.92 %–41.67 % of EAD across the ρ grid; ρ dominates the capital figure. |
| **C-2** | **High** | k-means segmentation separates almost nothing (0.54 pp DR spread); the Gaussian-mixture alternative is materially better (6.28 pp) but still descriptive. |
| **E-1** | Medium | EL of 10.14 % of EAD reflects a high (~14 %) default rate in a resolved-only sample. |
| **P-4** | Medium | Calibration does **not** improve the test-window Brier (raw 0.11950 vs isotonic 0.11970); the model still under-predicts by ~2.3 pp. |
| **P-1** | Medium | Absolute discrimination is moderate (Gini 0.3144); the tree champion beats the logit scorecard by only +0.0354 AUC. |
| **M-2** | Medium | Antithetic (+0.18 %) and stratified (+0.50 %) variance reduction are negligible on this bucketed portfolio; importance sampling is the technique that works. |
| **P-3** | Low | Score PSI 0.0145 on the mature window: stable population. |
| **P-5** | Low | Cutoff is statistical, not economic (32.39 % bad rate among approved). |
| **S-1** | Low | Accepted-only sample; reject inference now quantifies the gap (+0.058 illustrative PD) but does not fix it. |
| **Q-1** | Low | Outliers handled by winsorization rather than removal (conservative). |

## 10. Recommendation

1. **Use** the logistic scorecard (`logit_l2`) as the production model for discrimination,
   with the LightGBM challenger as a monitored alternative; the AUC difference
   (0.6218 vs 0.6572) is real but the tree lacks the scorecard's explainability, so the
   choice is a governance trade-off, not a pure-performance one.
2. **Do not use** the LGD model at loan level; use the realised/grade-level LGD
   (≈ 0.85–0.87) and revisit with a two-stage (recovery) model on a mature vintage only.
3. **Do not use** the SARIMAX forecast for planning — it loses to naive and ETS benchmarks;
   if vintage trend monitoring is required, restrict it to vintages with a resolved share
   above ~90 % (≤ 2015 here) and always report the naive benchmark alongside.
4. **Report every capital figure with its ρ**, using the Basel implied ρ (0.0322 at this
   PD) as the reference and 0.12–0.15 as a stress band.
5. **Prefer the Gaussian-mixture segmentation** over k-means for any behavioral use, but
   treat it as descriptive — not a calibrated rating.
6. **Before any decision use**, re-estimate the cutoff against an explicit cost/return
   matrix and validate on a vintage that is fully matured.
