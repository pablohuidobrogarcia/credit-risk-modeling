# Credit Risk Modeling Pipeline — Lending Club

![CI](https://github.com/Pablo-dot1/credit-risk-lending-club/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

An end-to-end, reproducible credit-risk pipeline built on the public Lending Club
loan book (2007–2018):

**data quality → SQL/vintage analysis → default-rate trend monitoring → PD scorecard and
challengers → LGD / EAD / expected loss → customer segmentation → single-factor
(Vasicek) Monte Carlo portfolio loss simulation.**

> **Honesty rule for this repository:** every number quoted in this README and in
> [`reports/model_validation_report.md`](reports/model_validation_report.md) is read from
> [`reports/results/summary.json`](reports/results/summary.json), which a single run of
> `python -m src.pipeline all` regenerates. If a metric is not in that file, it is not
> written anywhere in the documentation.

---

## Contents

1. [Results at a glance](#results-at-a-glance)
2. [The validation split (read this first)](#the-validation-split-read-this-first)
3. [Data](#1-data)
4. [Pipeline](#2-pipeline)
5. [Methodology](#3-methodology)
6. [Quick start](#4-quick-start)
7. [Repository layout](#5-repository-layout)
8. [Key design decisions](#6-key-design-decisions)
9. [Limitations — read this before using anything here](#7-limitations--read-this-before-using-anything-here)
10. [Reproducibility](#8-reproducibility)
11. [License](#9-license)

---

## Results at a glance

All headline metrics below are measured on the **primary (fully-mature) out-of-time
window**: train 2010–2013, test 2014–2015, 36-month loans only (see
[the validation split](#the-validation-split-read-this-first)). The late 2017–2018 window
is reported separately as the censoring-exposed secondary analysis.

| Stage | Headline result (primary / mature window) |
|---|---|
| **Sample** | 500,000 resolved loans, vintages 2007–2018, default rate **19.99 %** |
| **Trend analysis** | 74 monthly vintages (2009-11 → 2015-12, mature only), mean default rate **16.31 %** |
| **PD — champion** | Random Forest, OOT 2014–2015: **AUC 0.6157 / Gini 0.2315 / KS 0.1702** |
| **PD — benchmark** (with lender `grade`/`int_rate`) | **AUC 0.6700 / Gini 0.3399** |
| **LGD — champion** | Gamma GLM, MAE **0.0996**, R² **−0.0846** |
| **Expected loss** | **$207.2M** on $2,104.71M EAD → **9.85 %** of EAD (primary OOT portfolio) |
| **Segmentation** | **k = 2** (silhouette 0.2185, bootstrap ARI 0.9629), **0** degenerate segments |
| **Monte Carlo** (100,000 paths, ρ = 0.12) | EL **9.85 %**, UL **5.46 %**, VaR 99 % **26.80 %**, VaR 99.9 % **34.95 %** of EAD |

**Secondary (immature, censored) window — the same PD code path, for comparison:**

| Window | OOT years | Champion | AUC | Gini | KS | Default rate | Resolved share |
|---|---|---|---|---|---|---|---|
| **Mature (primary)** | 2014–2015 | random_forest | **0.6157** | 0.2315 | 0.1702 | 14.47 % | 99.97 % |
| Immature (secondary) | 2017–2018 | lightgbm | **0.6736** | 0.3473 | 0.2524 | 21.29 % | 38.17 % / 11.37 % |

**The censored window flatters the PD model by ≈ +0.058 AUC** (0.6736 vs 0.6157) — that
gap is the whole point of the redesign. The LGD/EL/simulation stages are run **only on the
primary window**; the secondary window is a PD-only comparison (`summary.json →
pd.window_comparison`).

**Read the [limitations](#7-limitations--read-this-before-using-anything-here) before
interpreting any of these numbers.** The headline AUC is moderate, the LGD model still has
a negative out-of-time R², and even the mature window is a single, historical credit cycle.

## The validation split (read this first)

The loan book ends at **2018-12**, but a loan only enters the modeling sample once it has a
*final* outcome (Fully Paid / Charged Off / Default). A 60-month loan issued in 2016 cannot
be Fully Paid before 2021, so the late vintages are **right-censored**: only their *fast*
resolutions are observed, which biases their default rate upwards. Training or testing on
those vintages therefore mixes a real signal with a measurement artefact.

The pipeline derives the validation split **from the data** instead of fixing a year by
hand. `data_prep.resolution_by_year_term()` computes, for every (origination year × term)
cell of the *population*, the share of loans resolved by the cut-off; `split_design()` then
records the rule and the evidence in `summary.json → pd.split_design`.

**Rule:** a `(year, term)` cell is usable for the **primary** split when at least **90 %**
of its population is resolved (`modeling.min_resolved_share`).

| | Primary (mature) | Secondary (immature) |
|---|---|---|
| **Train** | 2010–2013, term = 36 m | issue year < 2017 |
| **Test** | 2014–2015, term = 36 m | 2017–2018 |
| **Train resolution** | **99.59 %** resolved | — |
| **Test resolution** | **99.97 %** resolved | 38.17 % (2017) → 11.37 % (2018) |
| **OOT loans** | 165,597 (DR 14.47 %) | 83,859 (DR 21.29 %) |

The 60-month cells of 2014–2015 are only **82.85 %** / **67 %** resolved, so they are
excluded from the primary split; 2009 (36 m, 89.30 % resolved) misses the 90 % bar by a
narrow margin and is excluded too. The full per-cell table is in
`summary.json → eda.resolution_by_year_term`.

The secondary window is kept **deliberately**: the identical code path is re-run on it and
the two are compared side by side (`summary.json → pd.window_comparison`), which is how the
censoring effect is *measured* rather than asserted.

---

## 1. Data

### Source

| | |
|---|---|
| **Dataset** | Lending Club Loan Data (accepted + rejected applications), distributed on Kaggle as `wordsforthewiselending-club` |
| **Files** | `accepted_2007_to_2018Q4.csv.gz`, `rejected_2007_to_2018Q4.csv.gz` |
| **Where to put them** | `data/raw/` (this folder is **git-ignored**; the pipeline never downloads anything) |
| **Period** | 2007 – 2018 Q4 |

Download the dataset from Kaggle and drop both `*.csv.gz` files into `data/raw/`.
`src/data_prep.py` detects them by name (`accepted` / `rejected`), so the exact
capitalisation does not matter.

### Accepted vs rejected

* **Accepted** loans carry the target (`loan_status`), the terms and the full
  payment history → this is what the models are built on. It is converted to a typed
  Parquet file at `data/processed/accepted.parquet`.
* **Rejected** applications have no observed outcome (no `loan_status`), so **no
  PD/LGD/EL model can be trained or validated on them**. They are only mentioned in the
  [limitations](#7-limitations--read-this-before-using-anything-here) section: a score
  built exclusively on accepted loans is conditioned on the lender's own approval
  policy and therefore *cannot* be re-used to re-decide those rejected applications.

### Sample

The full accepted file does not fit comfortably in memory on a laptop, so the modeling
sample is a **year-stratified random sample** built with DuckDB:

* `data/processed/modeling_sample.parquet` — 500,000 rows, stratified by origination
  year so that every vintage keeps its relative weight.
* The sampling is deterministic: rows are ordered by a hash of `id` inside each
  `issue_year`, no RNG is involved, and the same sample is rebuilt on every machine.

The accepted file holds **2,260,668 loans**, of which **1,345,350 are resolved**
(1,076,751 Fully Paid, 268,559 Charged Off, 40 Default). The modeling sample is therefore
**500,000 / 1,345,350 ≈ 37.2 %** of the resolved population, with a default rate of
**19.99 %** (population resolved default rate: **19.96 %**). These population counts are
written to `summary.json → data.population` by `data_prep.population_totals()`.

### Target definition

```sql
CASE
  WHEN loan_status IN ('Charged Off', 'Default')            THEN 1  -- default
  WHEN loan_status = 'Fully Paid'                           THEN 0
  ELSE NULL                                                    -- not yet resolved
END
```

* **Default = `loan_status in ('Charged Off', 'Default')`.**
* The modeling sample contains **resolved loans only** (`Fully Paid`, `Charged Off`,
  `Default`). Loans still `Current`, `In Grace`, `Late` or `Issued` are excluded:
  including them would mean training on a label that is only known *later*.
* This is an **ever-green, time-inconsistent** definition in the literal sense that a
  loan is labelled with its final state, but a loan that never completes its term can
  only be "default" if it charged off early — see
  [right-censoring](#71-right-censoring-is-now-handled-by-design).

---

## 2. Pipeline

| Stage | Command | What it does | Key artefacts |
|---|---|---|---|
| Data | `data` | Raw CSV → typed Parquet, resolved-loan modeling sample | `data/processed/*.parquet` |
| Quality | `quality` | Missingness, outliers, leakage audit | `data_quality_*.csv/json` |
| SQL | `sql` | Vintage and default-rate queries over Parquet (Hive/Spark stand-in) | `sql_0*_*.csv` |
| EDA | `eda` | Monthly default-rate series, censoring diagnostic, decomposition, stationarity tests, SARIMAX forecast | `ts_*.csv`, `ts_*.json`, `ts_*.png` |
| PD | `pd` | WoE binning + IV, variable selection, logit/probit scorecard, regularised and tree challengers, OOT validation on the mature window, secondary censored-window comparison, PSI, SHAP | `pd_*.csv/json`, `roc_curves.png`, `ks_curve.png`, `calibration.png`, `shap_summary.png` |
| LGD/EAD | `lgd` | LGD on defaults (Gamma GLM vs tree), EAD from outstanding principal, EL = PD × LGD × EAD | `lgd_*.csv`, `el_*.csv` |
| Clustering | `clustering` | k-means + hierarchical segmentation, elbow/silhouette, PCA, segment profiles | `clustering_*.csv`, `clustering_*.png` |
| Simulation | `simulation` | Vasicek single-factor Monte Carlo: EL, UL, VaR, ES, ρ sensitivity, antithetic variance reduction, stress | `sim_*.csv`, `mc_*.png` |

Run everything with:

```bash
python -m src.pipeline all          # or: .\run.ps1   /   make all
python -m src.pipeline pd lgd       # individual stages
```

The final `reports/results/summary.json` collects every headline number.

---

## 3. Methodology

### 3.1 Data quality and leakage control

* Missingness is reported per column, together with the imputation policy actually used
  downstream (median for numeric, explicit `"Missing"` level for categorical).
* Outliers are reported per column at the 1st/99th percentile **and** winsorised on the
  training window only, so no information from the test window leaks into the fit.
* **Leakage control:** every post-origination variable is dropped before modeling —
  `recoveries`, `collection_recovery_fee`, `last_pymnt_d`, `last_pymnt_amnt`,
  `last_credit_pull_d`, `last_fico_range_*`, `total_pymnt`, `total_pymnt_inv`,
  `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `out_prncp`,
  `out_prncp_inv`, `total_cuml_recovered`, `hardship_*`, `debt_settlement_*`.
  These are *consequences* of the outcome and would make any "prediction" circular.

### 3.2 SQL (Hive/Spark stand-in)

The query suite in [`sql/`](sql/) is written in portable, HiveQL-flavoured SQL and
executed with **DuckDB** directly over the Parquet files — the same queries run unchanged
on Hive or Spark by swapping the reader. Statements cover:

* `01_vintage_analysis.sql` — origination volume and default-rate-by-vintage
  (`accepted.parquet`, grain: year × month).
* `02_default_rate_by_grade_year.sql` — default rate by `grade` × year and the
  year-on-year grade migration matrix.
* `03_data_quality_and_leakage.sql` — missingness and the leakage-column audit.

### 3.3 EDA and time-series monitoring

Monthly default rate by origination vintage, classical decomposition
(trend/seasonal/residual), ADF + KPSS stationarity tests, ACF/PACF inspection, and a
SARIMAX model with a **chronological** train/test split, Ljung–Box test on the residuals
and an out-of-sample forecast with confidence bands.

Because the data end at 2018-12, a dedicated
[censoring diagnostic](#71-right-censoring-is-now-handled-by-design) reports, for every
vintage and term, how much of it was actually resolved by the cut-off. The **primary**
series is restricted to the mature vintages (issue year ≤ 2015, 74 months); the full series
(110 months, immature tail included) is estimated in parallel as the secondary analysis, so
the effect of censoring on the forecast is measurable rather than asserted.

### 3.4 PD — scorecard and challengers

1. **WoE binning + IV** on 30 numeric/categorical candidates; variables kept after
   IV ranking, correlation screening and VIF.
2. **Interpretable models:** statsmodels logit (coefficients + odds ratios) and probit
   GLM, both on the WoE matrix.
3. **Points scorecard** from the logit: base score 600, base odds 1:50, PDO 20.
4. **Challengers:** Lasso/Ridge logit (CV over `C`), decision tree, random forest and
   LightGBM on the raw features inside a `ColumnTransformer` pipeline.
5. **Out-of-time validation:** train = 2010–2013, test = 2014–2015, 36-month loans only
   (the fully-mature primary split described at the top). Winsorisation bounds, encoders,
   WoE maps, the isotonic calibrator and the KS-optimal cutoff are all fit on train only.
   Reported: AUC, Gini, KS, ROC and KS curves, decile calibration, confusion matrix at
   the train-selected cutoff, and PSI per variable and per score. The same stack is re-run
   on the censored 2017–2018 window and the two are tabulated in
   `summary.json → pd.window_comparison`.
6. **Interpretation:** SHAP (TreeExplainer) for the champion, and a direct
   SHAP-vs-logit-coefficient comparison table.
7. **Benchmark sensitivity:** the same logit re-run *with* the lender's own
   `grade` / `int_rate` (variables the lender computed, hence not available to us before
   origination in a real application-decision setting), so the reader can see exactly
   how much discrimination is being borrowed from the lender's own model.

### 3.5 LGD / EAD / EL

* **LGD** = `1 − recoveries / exposure` computed on loans that reached a default
  outcome (Charged Off / Default), which is the standard recovery-based definition.
* **EAD** = outstanding funded principal at the moment of default.
* Models: a **Gamma GLM** (log link, the natural family for a `(0,1)` severity) with
  numeric + one-hot categorical terms, versus a LightGBM regressor; both compared on the
  same out-of-time defaults by MAE/RMSE/R².
* **EL = PD × LGD × EAD** per loan, aggregated by `grade` and for the whole
  out-of-time portfolio.

### 3.6 Segmentation

k-means on 12 behavioural features (loan size, income, DTI, utilisation, credit
histories, balances, affordability ratios) with a rigorous preprocessing chain:

* **Winsorisation at the 1st/99th percentile before standardising** — on raw values a
  handful of extreme borrowers (`annual_inc` up to millions, `dti = 999`) dominate the
  Euclidean distance and k-means degenerates into “one cluster with ~99.9 % of the
  loans”.
* Median imputation, then `StandardScaler`.
* The **same algorithm is used for the k-selection study and the shipped fit**: exact
  KMeans with 10 restarts. MiniBatchKMeans exists only as an opt-in fallback above 250k
  rows, because on the 120k-row segmentation sample exact KMeans was measured to be both
  faster and better (inertia 1,174,937 vs 1,184,469; silhouette 0.2200 vs 0.1889).
* **Silhouette on a stratified subsample** that always contains every cluster — the
  plain sklearn `sample_size=` subsample returns a single label on unbalanced
  clusterings and raises `ValueError`.
* **k is chosen on silhouette *and* stability**, never on a bare grid boundary
  (`clustering.select_k`): each candidate is scored as
  `0.6 · silhouette + 0.4 · bootstrap-ARI` (both normalised over the grid), and a boundary
  winner is only kept when no interior k comes within 5 % of it. Stability is the mean
  Adjusted Rand Index of the labels across bootstrap resamples (`cluster_stability`).
* Segments holding **less than 1 % of the sample are flagged `degenerate`** instead of
  being presented as business segments, and the cluster **default-rate spread** is reported
  so a segmentation that separates nothing cannot be sold as useful.
* Elbow + silhouette + stability curves, Ward hierarchical clustering with a truncated
  dendrogram, 2-D PCA projection, and a profile per segment (size, share, default rate,
  mean features).

### 3.7 Monte Carlo portfolio loss simulation

A **single-factor Vasicek / Basel-style** model: conditional on the systematic factor,
each bucket (grade × PD decile) draws defaults from a binomial whose conditional PD is
the Gaussian-correlation transform of the PD, then losses are `Σ LGD × EAD × default`.

* Inputs are the **out-of-time (calibrated) PD predictions** and the **LGD model output**
  per loan, bucketed by `grade` × PD decile, so the correlation structure is applied at
  portfolio level rather than to a single average PD.
* **Asset correlation** is documented and *swept*: a base case inside the Basel IRB
  retail band plus a sensitivity grid over `ρ ∈ {0.02, 0.05, 0.10, 0.15}`, and the
  PD-dependent correlation implied by the Basel retail formula at the observed portfolio
  PD. The loss distribution is extremely sensitive to `ρ`, so no capital figure is
  quoted without its `ρ`.
* Reported: expected loss, unexpected loss, VaR 99 % and 99.9 %, expected shortfall,
  convergence versus the number of paths, an **antithetic-variance reduction** report,
  and an **importance-sampling** estimator on the systematic factor for the 99.9 % VaR.
  The IS estimator is not asserted but *verified*: a unit test checks it is unbiased
  against the closed-form Vasicek VaR of an infinitely granular homogeneous portfolio,
  and its spread is measured by a replication-based RMSE study
  (`variance_reduction_comparison`) rather than the (invalid for quantiles) `se / √ESS`
  scaling. At the tuned shift the IS estimator cuts the tail-VaR RMSE by
  **4.36×** versus plain Monte Carlo at equal paths; the full shift curve, ESS and
  effective sample size are reported, and PD/LGD stress scenarios complete the stage.

---

## 4. Quick start

```bash
# 1. environment
python -m venv .venv && . .venv/Scripts/activate        # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt                      # pytest + ruff + jupyter

# 2. data: put accepted_2007_to_2018Q4.csv.gz in data/raw/

# 3. run
python -m src.pipeline all          # full pipeline
python -m src.pipeline eda pd       # a subset of stages

# 4. quality gates
python -m ruff check src tests
python -m pytest -q
```

`make all`, `make test`, `make lint` and `.\run.ps1` are equivalent entry points.

The unit tests run on **synthetic data**, so CI passes without the dataset.

---

## 5. Repository layout

```
.
├── config/config.yaml          # seed, split years, WoE/IV cuts, simulation settings
├── data/                       # git-ignored: raw CSVs, processed Parquet
├── notebooks/
│   ├── 01_data_and_quality.ipynb
│   ├── 02_eda_and_trends.ipynb
│   ├── 03_pd_scorecard.ipynb
│   ├── 04_lgd_ead_expected_loss.ipynb
│   ├── 05_clustering_segments.ipynb
│   └── 06_monte_carlo_simulation.ipynb
├── reports/
│   ├── figures/                # every chart referenced by the docs
│   ├── results/                # CSV/JSON artifacts + summary.json
│   └── model_validation_report.md
├── sql/                        # HiveQL-style queries executed with DuckDB
├── src/                        # typed, documented modules (one per stage)
├── tests/                      # pytest suite on synthetic data
├── .github/workflows/ci.yml    # ruff + pytest on 3.10 and 3.11
├── Makefile / run.ps1
├── pyproject.toml / requirements*.txt
└── README.md
```

---

## 6. Key design decisions

| Decision | Why |
|---|---|
| Target = `Charged Off`/`Default`, resolved loans only | The only outcome that is *observed*. Current/late loans would be an unknown label. |
| `grade`, `sub_grade`, `int_rate` **excluded** from the main feature set | These are the lender's own model outputs: using them would not test our ability to score a borrower, only our ability to copy the lender. They are kept as an explicit *benchmark* model instead. |
| All post-origination payment/recovery variables dropped | Pure outcome leakage. |
| Chronological (out-of-time) split, never random | Random splits leak the credit cycle across train and test and overstate performance. |
| Primary split restricted to ≥ 90 %-resolved `(year, term)` cells | Removes right-censoring bias from the headline metrics; the censored window is kept as a labelled secondary comparison. |
| Winsorisation/encoders fitted on train only | Test-window statistics must not influence the fit. |
| Linear/GLM models on the WoE matrix; tree models on raw features | WoE is what makes an interpretable logistic scorecard behave linearly; trees do not need it. |
| DuckDB over Parquet for SQL | Same queries run on Hive/Spark; no cluster needed to reproduce. |
| Winsorise before clustering | Prevents degenerate clusters driven by outliers. |
| Silhouette on a stratified sample | Plain random subsamples raise `ValueError: Number of labels is 1` on unbalanced clusterings. |
| Simulation at bucket level, not one average PD | Correlation risk is a *portfolio* effect; averaging PD first destroys the tail. |

---

## 7. Limitations — read this before using anything here

### 7.1 Right-censoring is now handled by design (and its size is measured)

The dataset stops at 2018-12 and the modeling sample only contains loans with a
**resolved** outcome, so a loan issued in 2017–2018 can only enter the sample if it
defaulted or was fully repaid *early*. That makes the late vintages non-representative:
they under-represent loans that were still performing and over-represent fast defaults.

**The redesign handles this by construction.** The primary split is trained and tested only
on `(year, term)` cells whose population is ≥ 90 % resolved
(see [the validation split](#the-validation-split-read-this-first)): train 2010–2013
(36 m, 99.59 % resolved), test 2014–2015 (36 m, 99.97 % resolved). The censored window is
retained as a labelled *secondary* analysis, and the difference between the two is the
measured censoring effect:

| Window | OOT years | Test resolution | Champion | AUC | Gini | KS | Default rate |
|---|---|---|---|---|---|---|---|
| **Mature (primary)** | 2014–2015 | 99.97 % | random_forest | **0.6157** | 0.2315 | 0.1702 | 14.47 % |
| Immature (secondary) | 2017–2018 | 38.17 % / 11.37 % | lightgbm | **0.6736** | 0.3473 | 0.2524 | 21.29 % |

Testing on the censored window **inflates AUC by +0.0579** (0.6736 vs 0.6157). That gap is
a measurement artefact, not model quality — it is exactly why the earlier head-line figure
of 0.6736 is no longer quoted as the headline.

Per-vintage resolved share (population, from `data/processed/accepted.parquet`):

| Origination year | Loans | Resolved | Resolved share | Default rate (resolved) |
|---|---|---|---|---|
| 2011 | 21,721 | 21,721 | 100.0 % | 15.18 % |
| 2012 | 53,367 | 53,367 | 100.0 % | 16.20 % |
| 2013 | 134,814 | 134,804 | 99.99 % | 15.60 % |
| 2014 | 235,629 | 223,103 | 94.68 % | 18.45 % |
| 2015 | 421,095 | 375,546 | 89.18 % | 20.19 % |
| 2016 | 434,407 | 293,105 | 67.47 % | 23.29 % |
| **2017** | 443,579 | 169,321 | **38.17 %** | 23.13 % |
| **2018** | 495,242 | 56,318 | **11.37 %** | 15.76 % |

**Effect on the forecast** — the same SARIMAX order (1,1,1)(1,0,1,12) estimated on the
mature vintages (primary) versus the full series (secondary):

| Series | Months | Train RMSE | Test RMSE |
|---|---|---|---|
| Full 2009-11 → 2018-12 (secondary) | 110 | 0.027972 | **0.098120** |
| Mature vintages only (≤ 2015, primary) | 74 | 0.031061 | **0.014797** |

The out-of-sample error **drops by a factor of ≈ 6.6** (0.0981 → 0.0148) once the censored
vintages are removed, confirming that censoring — not the model — drives most of the
reported forecast error on the full series.

**What is still not fixed.** Censoring is a *design* fix for the metrics, not a repair of
the underlying data: the scorecard cannot be deployed to score a *new* borrower whose loan
has not yet matured, and the mature window is an older, easier credit environment than the
censored one. The right long-run fix remains survival/hazard modelling of the
time-to-default, which is out of scope here.

### 7.2 Discrimination is moderate, and the lender's own variables do better

Out-of-time on the **primary (mature) window** (2014–2015), champion **Random Forest**:

| Feature set | AUC | Gini | KS |
|---|---|---|---|
| Without `grade`/`int_rate` — champion **random_forest** | **0.6157** | **0.2315** | 0.1702 |
| Without `grade`/`int_rate` — interpretable logit scorecard (`logit_l2`) | 0.6113 | 0.2226 | 0.1647 |
| With `grade`/`int_rate` — plain logit benchmark | **0.6700** | **0.3399** | — |

Two honest readings:

* The lender's own grade and rate add **+0.0543 AUC / +0.1173 Gini** over the best model
  built only from loan-application data. A meaningful share of the discrimination a
  deployed scorecard would reach is inherited from information we do not have at
  application time; this model is *not* a replacement for the lender's score.
* The boosted-tree/forest champion beats the interpretable logit scorecard by only
  **+0.0044 AUC**. On this window the extra complexity buys almost nothing, which is worth
  knowing before choosing a challenger in production.

Full OOT table (see [`pd_model_comparison.csv`](reports/results/pd_model_comparison.csv)):

| Model | AUC | Gini | KS | CV AUC | Train AUC |
|---|---|---|---|---|---|
| random_forest (champion) | 0.6157 | 0.2315 | 0.1702 | 0.6304 | 0.7140 |
| logit_l2 | 0.6113 | 0.2226 | 0.1646 | 0.6302 | 0.6307 |
| probit | 0.6113 | 0.2227 | 0.1643 | — | 0.6307 |
| logit (scorecard) | 0.6112 | 0.2224 | 0.1641 | — | 0.6307 |
| logit_l1 | 0.6112 | 0.2223 | 0.1641 | 0.6303 | 0.6307 |
| lightgbm | 0.6068 | 0.2137 | 0.1589 | 0.6171 | 0.7897 |
| decision_tree | 0.5611 | 0.1221 | 0.1215 | 0.5736 | 0.5769 |

The forest champion beats the interpretable logit scorecard by only **+0.0044 AUC**. On
this window the extra complexity buys almost nothing, which is worth knowing before
choosing a challenger in production. Note also that LightGBM *overfits* (train AUC 0.7897
vs OOT 0.6068) and lands below the penalised logits out of time.

### 7.3 The default-rate forecast is weak

* ADF fails to reject a unit root (statistic −1.5506, p = 0.5083) and the series is **not
  stationary**, so any ARIMA forecast is fragile. (On the mature series the KPSS p-value
  falls outside the look-up table, i.e. it does not reject either — the mature series is
  better behaved but not stationary in the ADF sense.)
* **Ljung–Box rejects white-noise residuals** on the mature series too: lag 10 p = 0.0026,
  lag 20 p = 0.0365 — the model leaves serial structure unexplained.
* Train RMSE **0.031061** vs test RMSE **0.014797**: on the *mature* series the
  out-of-sample error is actually lower than in-sample, i.e. the mature series is easy to
  forecast but that is precisely because it excludes the regime shift that matters.

**The forecast should not be used for planning.** It is included to demonstrate the
diagnostic workflow (decomposition, ADF/KPSS, ACF/PACF, Ljung–Box, chronological
backtest), not as a usable vintage forecast.

### 7.4 LGD is not usable for individual predictions

Out-of-time on the primary window (2014–2015 defaults; train 7,641 defaults, test 23,953):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| gamma_glm (best MAE) | 0.0996 | 0.1550 | **−0.0846** |
| lightgbm | 0.1027 | 0.1532 | **−0.0586** |

Both out-of-time R² are still **negative**: neither model beats predicting the mean, and on
this window the tree has the marginally better (still negative) R² while the Gamma GLM has
the lower MAE. This is expected given the data — observed LGD on this portfolio is
extremely concentrated (mean LGD OOT = **0.8691**, median = **0.8964**, because most
charge-offs are never recovered) — but it means:

* The LGD output must **not** be used as a loan-level prediction. Its only defensible
  use is as a **portfolio/grade-level average**, close to the realised mean.
* Consequently the **9.85 % EL rate is driven almost entirely by PD and by the ~86 % LGD
  level, not by any LGD discriminating power.**

### 7.5 Calibration

An isotonic calibrator was fitted on the training window and applied out of time (it is
monotone, so AUC/Gini/KS are unchanged):

| Basis (OOT, primary window) | Mean predicted | Observed | Calibration-in-the-large | Slope | Brier |
|---|---|---|---|---|---|
| raw champion score | 0.1196 | 0.1447 | 0.1218 | 0.5809 | **0.12176** |
| isotonic-calibrated | 0.1196 | 0.1447 | 0.1196 | 0.3860 | **0.12370** |

Two honest observations:

* The model **under-predicts the OOT default rate** (11.96 % predicted vs 14.47 %
  observed). The 2014–2015 default rate is genuinely higher than the 2010–2013 training
  rate (12.36 %), so part of this is a real level shift.
* The calibrator was fitted on training-window scores, so it inherits an optimistic
  in-sample fit; applied OOT its slope collapses to **0.3860** (vs 0.5809 raw) and its
  Brier is **worse** (0.12370 vs 0.12176). The raw score is already conservatively
  *low*, and the isotonic map makes the probabilities *flatter*, not better. The
  calibrated PDs are used for EL/simulation because that is the operationally correct
  pipeline, but the improvement is **not** taken for granted — and the EL is reported both
  ways:
  **$207.23M (calibrated PD, 9.85 % of EAD)** vs **$214.07M (raw PD, 10.17 % of EAD)**.

### 7.6 Everything else

* **Selection bias — rejected applications.** Only *accepted* loans have an outcome. Any
  score built here is conditioned on the lender's approval policy and **cannot** be used
  to re-decide previously rejected applications. The rejected file contributes nothing
  to the models.
* **No macro-economic variables.** The simulation applies PD/LGD *shocks*; it does not
  model the credit cycle through a real macro factor, so the "stress" scenarios are
  shocks, not forecasts.
* **Vasicek correlation is an assumption.** The Basel retail formula at the portfolio PD
  (EAD-weighted PD 0.1138) implies ρ = **0.0324**, whereas the simulation base case is
  ρ = 0.12. VaR 99.9 % ranges from **17.97 %** of EAD (ρ = 0.02) to **39.05 %**
  (ρ = 0.15) — the choice of ρ matters more than anything else in the capital number.
  That is why the full sensitivity grid is reported instead of a single figure.
* **Importance sampling works, but it is not a free lunch.** Shifting the systematic factor
  towards the loss tail and reweighting by the likelihood ratio gives an unbiased 99.9 %
  VaR (733.5M IS vs 741.5M naive at shift 1.5; the estimator is checked against the
  closed-form Vasicek VaR in the test suite) and it **genuinely reduces the estimator's
  spread**: a replication-based RMSE study puts the IS tail-VaR RMSE at **4.36× smaller**
  than plain Monte Carlo at equal paths (3.23M vs 14.10M), matched by the batched standard
  error (SE ratio **0.21×** vs naive). The caveat is that the win comes at a real cost — the
  effective sample size falls to 2,276 of 20,000 paths (11.4 %) — and pushing the shift out
  to the factor quantile (≈ 3.09) collapses it further (ESS ≈ 23) and biases the estimate
  upward. The shift is therefore tuned to 1.5 and the whole curve is reported in
  [`sim_importance_sampling_sensitivity.csv`](reports/results/sim_importance_sampling_sensitivity.csv),
  with the RMSE study in
  [`sim_importance_sampling_variance_reduction.csv`](reports/results/sim_importance_sampling_variance_reduction.csv).
* **Segmentation is descriptive, not predictive.** k = 2 is chosen (silhouette 0.2185,
  bootstrap ARI 0.9629, 0 degenerate segments) but the two segments' default rates are
  close (19.09 % vs 20.40 % — a **1.31 pp spread**), so the segments are **not** a useful
  risk-rating substitute. Silhouette monotonically prefers the smallest k tested and the
  combined score confirms k = 2 well outside the 5 % boundary tolerance; it remains a
  metric-driven choice, not an obviously "business-correct" one.
* **Cutoff choice.** The confusion matrix uses the KS-optimal cutoff selected on the
  **training** window (0.129212) and applied out of time, not a business-optimal cutoff:
  on the OOT set it approves 85.53 % of loans with a **36.94 %** bad rate among approved
  (precision 0.1965, recall 0.5344).
* **Antithetic variates help only marginally** (mean SE falls by **0.19 %** at 100,000
  paths; the convergence table shows the reduction is negligible at every path count).
  With bucketed binomial draws the pairing adds almost nothing; reporting a null result is
  more useful than hiding it.
* **Single dataset, single period, US consumer credit only.** Nothing here is evidence
  about another product, country or underwriting regime, and the 2018 data vintage means
  the scorecard is a historical exercise.

---

## 8. Reproducibility

* Single global seed (`config/config.yaml`), applied to numpy/scipy/sklearn/LightGBM.
* The stratified sample is hash-based, hence machine-independent.
* Every artifact under `reports/results/` and `reports/figures/` is regenerated by
  `python -m src.pipeline all`; nothing is hand-edited.
* `python -m ruff check src tests` and `python -m pytest -q` must be clean before a
  commit (68 tests).
* CI runs lint + tests on Python 3.10 and 3.11 **without** the dataset.

---

## 9. License

MIT — see [LICENSE](LICENSE). The Lending Club data is redistributed by Kaggle under
its own terms and is **not** included in this repository.
