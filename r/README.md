# R replication of the WoE logistic scorecard (optional)

> **Status in this environment: SKIPPED — R is not installed.**
> `Rscript` was not found on the machine where this repository was last built, so
> the R replication below has **not been run here** and **no R-produced numbers
> are quoted anywhere** in `README.md` or the validation report. The script is
> kept so the comparison is reproducible for anyone who does have R; if you run
> it, please replace this note with your actual output.

## What this replicates

The Python pipeline fits the interpretable PD scorecard with
`statsmodels.Logit` on the Weight-of-Evidence (WoE) matrix (`src/pd_model.py`,
`fit_statsmodels_woe`). This folder reproduces that exact model with R's
`glm(..., family = binomial())` and compares:

1. the **coefficients** (Python vs R), and
2. the **out-of-time AUC** on the same test window.

The two should agree to within numerical tolerance because both are fitting the
same log-likelihood on the same design matrix. Any material difference would
point at a binning/encoding discrepancy rather than at the optimiser.

## Inputs

`replicate_scorecard.R` expects the Python pipeline to have already run, so the
following files exist under `reports/results/`:

| File | Purpose |
|---|---|
| `pd_scorecard_points.csv` | WoE bins and points per variable (the binning evidence) |
| `pd_logit_coefficients.csv` | Python `Logit` coefficients (the reference) |
| `pd_model_comparison.csv` | Python out-of-time AUC |
| `pd_woe_oot.csv` | WoE matrix + `default_flag` for the test window |

`pd_woe_oot.csv` is written by the pipeline for exactly this cross-check (see
`stage_pd` → `save_table(..., "pd_woe_oot.csv")`). If the file is absent, run
`python -m src.pipeline pd` first.

## Running it

```sh
Rscript r/replicate_scorecard.R
```

The script prints a coefficient comparison table and the R AUC, and writes
`reports/results/r_scorecard_comparison.csv`.

## Why this is optional

The Python scorecard is already validated by the unit tests and the out-of-time
metrics; the R run is a *plausibility cross-check* on the estimator, not a
deliverable number. It is therefore deliberately kept out of the critical path
and out of CI.
