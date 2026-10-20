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
2. [Data](#1-data)
3. [Pipeline](#2-pipeline)
4. [Methodology](#3-methodology)
5. [Quick start](#4-quick-start)
6. [Repository layout](#5-repository-layout)
7. [Key design decisions](#6-key-design-decisions)
8. [Limitations — read this before using anything here](#7-limitations--read-this-before-using-anything-here)
9. [Reproducibility](#8-reproducibility)
10. [License](#9-license)

---

## Results at a glance

| Stage | Headline result |
|---|---|
| **Sample** | 500,000 resolved loans, vintages 2007–2018, default rate **19.99 %** (99,948 defaults) |
| **Trend analysis** | 110 monthly vintages (2009-11 → 2018-12), mean default rate **17.41 %** |
| **PD — champion** | LightGBM, OOT 2017–2018: **AUC 0.6736 / Gini 0.3473 / KS 0.2524** |
| **PD — benchmark** (with lender `grade`/`int_rate`) | **AUC 0.7022 / Gini 0.4043** |
| **LGD — champion** | Gamma GLM, MAE **0.0827**, R² **−0.2251** |
| **Expected loss** | **$232.1M** on $1,212.95M EAD → **19.14 %** of EAD (OOT portfolio) |
| **Segmentation** | **k = 2** (silhouette 0.2148), **0** degenerate segments |
| **Monte Carlo** (100,000 paths, ρ = 0.12) | EL **19.14 %**, UL **8.26 %**, VaR 99 % **42.18 %**, VaR 99.9 % **50.98 %** of EAD |

**Read the [limitations](#7-limitations--read-this-before-using-anything-here) before
interpreting any of these numbers.** The headline AUC is moderate, the LGD model has a
negative out-of-time R², and the out-of-time window is right-censored — all of it is
stated with numbers below.

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

The accepted file holds **2,260,668 loans**, of which **1,345,311 are resolved**
(1,076,712 Fully Paid, 268,559 Charged Off, 40 Default). The modeling sample is therefore
**500,000 / 1,345,311 ≈ 37.2 %** of the resolved population, with a default rate of
**19.99 %** (population resolved default rate: **19.96 %**).

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
  [right-censoring](#71-right-censoring-is-the-main-limitation).

---

## 2. Pipeline

| Stage | Command | What it does | Key artefacts |
|---|---|---|---|
| Data | `data` | Raw CSV → typed Parquet, resolved-loan modeling sample | `data/processed/*.parquet` |
| Quality | `quality` | Missingness, outliers, leakage audit | `data_quality_*.csv/json` |
| SQL | `sql` | Vintage and default-rate queries over Parquet (Hive/Spark stand-in) | `sql_0*_*.csv` |
| EDA | `eda` | Monthly default-rate series, censoring diagnostic, decomposition, stationarity tests, SARIMAX forecast | `ts_*.csv`, `ts_*.json`, `ts_*.png` |
| PD | `pd` | WoE binning + IV, variable selection, logit/probit scorecard, regularised and tree challengers, OOT validation, PSI, SHAP | `pd_*.csv/json`, `roc_curves.png`, `ks_curve.png`, `calibration.png`, `shap_summary.png` |
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
[censoring diagnostic](#71-right-censoring-is-the-main-limitation) reports, for every
vintage, how much of it was actually resolved by the cut-off; the SARIMAX fit is
repeated on mature vintages only so the effect of censoring on the forecast is
measurable rather than asserted.

### 3.4 PD — scorecard and challengers

1. **WoE binning + IV** on 30 numeric/categorical candidates; variables kept after
   IV ranking, correlation screening and VIF.
2. **Interpretable models:** statsmodels logit (coefficients + odds ratios) and probit
   GLM, both on the WoE matrix.
3. **Points scorecard** from the logit: base score 600, base odds 1:50, PDO 20.
4. **Challengers:** Lasso/Ridge logit (CV over `C`), decision tree, random forest and
   LightGBM on the raw features inside a `ColumnTransformer` pipeline.
5. **Out-of-time validation:** train = vintages ≤ 2016, test = 2017–2018. Winsorisation
   bounds, encoders and WoE maps are fit on train only.
   Reported: AUC, Gini, KS, ROC and KS curves, decile calibration, confusion matrix at
   the train-selected cutoff, and PSI per variable and per score.
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
* **MiniBatchKMeans** for fitting samples above 100k rows, exact KMeans (10 restarts)
  for the smaller k-selection sample.
* **Silhouette on a stratified subsample** that always contains every cluster — the
  plain sklearn `sample_size=` subsample returns a single label on unbalanced
  clusterings and raises `ValueError`.
* Segments holding **less than 1 % of the sample are flagged `degenerate`** instead of
  being presented as business segments.
* Elbow + silhouette curve, Ward hierarchical clustering with a truncated dendrogram,
  2-D PCA projection, and a profile per segment (size, share, default rate, mean
  features).

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
  and PD/LGD stress scenarios.

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
| Winsorisation/encoders fitted on train only | Test-window statistics must not influence the fit. |
| Linear/GLM models on the WoE matrix; tree models on raw features | WoE is what makes an interpretable logistic scorecard behave linearly; trees do not need it. |
| DuckDB over Parquet for SQL | Same queries run on Hive/Spark; no cluster needed to reproduce. |
| Winsorise before clustering | Prevents degenerate clusters driven by outliers. |
| Silhouette on a stratified sample | Plain random subsamples raise `ValueError: Number of labels is 1` on unbalanced clusterings. |
| Simulation at bucket level, not one average PD | Correlation risk is a *portfolio* effect; averaging PD first destroys the tail. |

---

## 7. Limitations — read this before using anything here

### 7.1 Right-censoring is the main limitation

The dataset stops at 2018-12 and the modeling sample only contains loans with a
**resolved** outcome, so a loan issued in 2017–2018 can only enter the sample if it
defaulted or was fully repaid *early*. That makes the late vintages non-representative:
they under-represent loans that were still performing and over-represent fast defaults.

How bad it is, per vintage (population, from `data/processed/accepted.parquet`):

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

* The **out-of-time test window (2017–2018) is exactly the censored part**: only 38.17 %
  of the 2017 vintage and 11.37 % of the 2018 vintage had resolved. Every OOT metric in
  this repository (AUC 0.6736, Gini 0.3473, KS 0.2524, EL 19.14 % of EAD) is therefore
  measured on a *conditional* subsample, not on a random sample of those vintages.
* The monthly default-rate series is biased for the same reason: it is computed on
  resolved loans only, so late vintages are inflated by early defaults
  (2018 monthly observations come from 11.37 % of that vintage).

**Effect on the forecast** — the same SARIMAX order (1,1,1)(1,0,1,12) re-estimated on
vintages up to 2015 only (where ≥ 89.18 % of each vintage is resolved):

| Series | Months | Train RMSE | Test RMSE |
|---|---|---|---|
| Full 2009-11 → 2018-12 | 110 | 0.027972 | **0.098120** |
| Mature vintages only (≤ 2015) | 74 | 0.031061 | **0.014797** |

The out-of-sample error **drops by a factor of ≈ 6.6** (0.0981 → 0.0148) once the
censored vintages are removed, which confirms that censoring — not the model — drives
most of the reported forecast error. The honest reading is: *on the full sample the
forecast is not usable*, and the confident-looking mature-sample result is only
available for vintages that are already closed.

If the exact 2018 vintage value mattered, the correct fix would be survival/hazard
modelling or restricting the analysis to a mature window — neither is done here.

### 7.2 Discrimination is moderate, and the lender's own variables do better

Out-of-time (2017–2018), including the lender's `grade` / `int_rate` as features:

| Feature set | AUC | Gini |
|---|---|---|
| Without `grade`/`int_rate` — champion **LightGBM** | **0.6736** | **0.3473** |
| With `grade`/`int_rate` — plain logit benchmark | **0.7022** | **0.4043** |

The lender's own grade and rate add **+0.0286 AUC / +0.0570 Gini** over the best model
built only from loan-application data. In other words, our model captures a large part of
the lender's ranking but not all of it, and a meaningful share of the discrimination a
deployed scorecard would reach is inherited from information we do not have at
application time. The model is *not* a replacement for the lender's score.

Full OOT table (see [`pd_model_comparison.csv`](reports/results/pd_model_comparison.csv)):

| Model | AUC | Gini | KS | CV AUC |
|---|---|---|---|---|
| lightgbm (champion) | 0.6736 | 0.3473 | 0.2524 | 0.6734 |
| random_forest | 0.6678 | 0.3356 | 0.2447 | 0.6770 |
| logit_l2 | 0.6609 | 0.3217 | 0.2332 | 0.6715 |
| logit_l1 | 0.6608 | 0.3215 | 0.2332 | 0.6715 |
| logit (scorecard) | 0.6608 | 0.3215 | 0.2332 | — |
| probit | 0.6604 | 0.3209 | 0.2326 | — |
| decision_tree | 0.6146 | 0.2292 | 0.1733 | 0.6431 |

The boosted tree beats the interpretable logit scorecard by only **0.0128 AUC** while
being far harder to explain — worth knowing before choosing a challenger in production.

### 7.3 The default-rate forecast is weak

* ADF fails to reject a unit root (statistic −0.5568, p = 0.8804) and KPSS rejects
  stationarity (statistic 0.5652, p = 0.027): the series is **not stationary**, so any
  ARIMA forecast is fragile.
* **Ljung–Box rejects white-noise residuals**: lag 10 p = 0.000095, lag 20 p = 0.0025 —
  the model leaves serial structure unexplained.
* Train RMSE **0.027972** vs test RMSE **0.098120**: out-of-sample error is **3.5×**
  the in-sample error. Section 7.1 shows ≈ 6.6× of that gap is censoring, not the model.

**The forecast should not be used for planning.** It is included to demonstrate the
diagnostic workflow (decomposition, ADF/KPSS, ACF/PACF, Ljung–Box, chronological
backtest), not as a usable vintage forecast.

### 7.4 LGD is not usable for individual predictions

| Model | MAE | RMSE | R² |
|---|---|---|---|
| gamma_glm (chosen) | 0.0827 | 0.1059 | **−0.2251** |
| lightgbm | 0.0833 | 0.1062 | **−0.2327** |

Both out-of-time R² are **negative**: neither model beats predicting the mean, and the
tree cannot beat the Gamma GLM either. This is expected given the data — observed LGD on
this portfolio is extremely concentrated (realised LGD = **0.9296**, median = **1.0000**,
because most charge-offs are never recovered) — but it means:

* The LGD output must **not** be used as a loan-level prediction. Its only defensible
  use is as a **portfolio/grade-level average**: mean LGD is 0.8858 (A) down to 0.8792
  (G), i.e. essentially flat, close to the 0.9296 realised figure.
* Consequently the 19.14 % EL rate is driven almost entirely by PD and by the ~89 % LGD
  level, not by any LGD discriminating power.

### 7.5 Calibration

An isotonic calibrator was fitted on the training window and applied out of time (it is
monotone, so AUC/Gini/KS are unchanged):

| Basis (OOT) | Mean predicted | Observed | Calibration-in-the-large | Slope | Brier |
|---|---|---|---|---|---|
| raw champion score | 0.1907 | 0.2129 | 0.1378 | 0.8951 | **0.15770** |
| isotonic-calibrated | 0.1899 | 0.2129 | 0.1428 | 0.7658 | **0.15816** |

Two honest observations:

* The model **under-predicts the OOT default rate** (19.07 % predicted vs 21.29 %
  observed). The 2017–2018 default rate is genuinely higher than the 2007–2016 training
  rate (19.73 %), so part of this is a *level shift the model never saw*.
* The calibrator was fitted on training-window scores fitted on the same data, so it
  inherits an optimistic in-sample fit; applied OOT it is **slightly worse** than the raw
  score (Brier 0.15816 vs 0.15770, slope 0.7658 vs 0.8951). The calibrated PDs are used
  for EL/simulation because that is the operationally correct pipeline, but the
  improvement is **not** taken for granted — and the EL is reported both ways:
  **$232.14M (calibrated PD, 19.14 % of EAD)** vs **$229.96M (raw PD, 18.96 %)**.

### 7.6 Everything else

* **Selection bias — rejected applications.** Only *accepted* loans have an outcome. Any
  score built here is conditioned on the lender's approval policy and **cannot** be used
  to re-decide previously rejected applications. The rejected file contributes nothing
  to the models.
* **No macro-economic variables.** The simulation applies PD/LGD *shocks*; it does not
  model the credit cycle through a real macro factor, so the "stress" scenarios are
  shocks, not forecasts.
* **Vasicek correlation is an assumption.** The Basel retail formula at the portfolio PD
  (EAD-weighted PD 0.2153) implies ρ = **0.0301**, whereas the simulation base case is
  ρ = 0.12. VaR 99.9 % ranges from **30.75 %** of EAD (ρ = 0.02) to **55.35 %**
  (ρ = 0.15) — the choice of ρ matters more than anything else in the capital number.
  That is why the full sensitivity grid is reported instead of a single figure.
* **Segmentation is descriptive, not predictive.** The two segments differ in size,
  income and loan size, but their default rates are close (21.05 % vs 19.24 %), so the
  segments are **not** a useful risk-rating substitute. Silhouette also monotonically
  prefers the smallest k tested (0.2148 at k = 2 vs 0.1634 at k = 3); k = 2 is the
  metric-driven choice, not an obviously "business-correct" one.
* **Cutoff choice.** The confusion matrix uses the cutoff that matches the training
  default rate (0.201619 on the raw score), not a business-optimal cutoff: on the OOT set
  it approves 78.71 % of loans with a 31.31 % bad rate among approved.
* **Antithetic variates did not help here** (see the variance-reduction table: SE
  reductions between −0.07 % and +0.13 %). With bucketed binomial draws the pairing
  adds almost nothing; reporting a null result is more useful than hiding it.
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
  commit (52 tests).
* CI runs lint + tests on Python 3.10 and 3.11 **without** the dataset.

---

## 9. License

MIT — see [LICENSE](LICENSE). The Lending Club data is redistributed by Kaggle under
its own terms and is **not** included in this repository.
