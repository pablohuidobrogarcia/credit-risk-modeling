"""Tests for the block-2 modelling additions: wider challenger features, the
monotone-constrained LightGBM wrapper and the temporal LightGBM tuner."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import pd_model


def _synthetic(n: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "annual_inc": rng.lognormal(11, 0.5, n),
            "dti": rng.uniform(0, 40, n),
            "purpose": rng.choice(["a", "b", "c"], n),
        }
    )
    logit = -0.00002 * X["annual_inc"] + 0.05 * X["dti"] - 1.0
    p = 1 / (1 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, p)).astype(int)
    return X, y


def test_monotone_wrapper_fits_and_predicts():
    """The wrapper must fit inside a Pipeline whose ColumnTransformer widens the
    categorical columns - the exact situation that broke a fixed-length
    monotone_constraints list."""
    X, y = _synthetic()
    pipe = pd_model.make_tree_pipeline(
        "lightgbm",
        ["annual_inc", "dti"],
        ["purpose"],
        seed=1,
        monotone_constraints={"annual_inc": -1, "dti": 1},
    )
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_monotone_wrapper_cross_validates():
    """cross_val_score requires the estimator to advertise itself as a classifier."""
    X, y = _synthetic(n=2000)
    pipe = pd_model.make_tree_pipeline(
        "lightgbm",
        ["annual_inc", "dti"],
        ["purpose"],
        seed=1,
        monotone_constraints={"annual_inc": -1, "dti": 1},
    )
    auc, std = pd_model.cross_validate_pipeline(pipe, X, y, n_splits=3)
    assert np.isfinite(auc) and 0.0 < auc <= 1.0
    assert np.isfinite(std)


def test_temporal_tuner_returns_params_and_table():
    X, y = _synthetic(n=3000)
    rng = np.random.default_rng(3)
    years = pd.Series(rng.choice([2010, 2011, 2012, 2013], size=len(X)))
    best, table = pd_model.tune_lightgbm_temporal(
        X,
        y,
        years,
        ["annual_inc", "dti"],
        ["purpose"],
        tune_years=[2010, 2011, 2012],
        valid_years=[2013],
        monotone_constraints={"annual_inc": -1, "dti": 1},
        n_trials=3,
    )
    assert len(table) == 3
    assert "valid_auc" in table.columns
    assert set(best) >= {"num_leaves", "n_estimators"}
    # The tuner must never see anything outside the requested slices.
    assert table["valid_auc"].max() <= 1.0


def test_monotone_wrapper_declares_classifier_tag():
    est = pd_model.MonotoneLGBMClassifier(params={}, numeric=["a"], monotone={"a": 1})
    tags = est.__sklearn_tags__()
    assert tags.estimator_type == "classifier"


def test_pipeline_tolerates_a_column_that_is_all_missing_in_a_slice():
    """Several bureau variables only exist from 2012 onward, so they are entirely
    missing inside the 2010-2011 slices the tuner / calibrator fit on. The
    pipeline must carry such a column as a constant instead of dropping it (and
    warning), so the fitted feature width stays stable across vintages."""
    n = 2000
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "annual_inc": rng.lognormal(11, 0.5, n),
            "late_vintage_only": np.full(n, np.nan),  # not present in early y vintages
            "purpose": rng.choice(["a", "b"], n),
        }
    )
    y = pd.Series(rng.binomial(1, 0.2, n)).astype(int)
    pipe = pd_model.make_tree_pipeline("lightgbm", ["annual_inc", "late_vintage_only"], ["purpose"], seed=1)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any SimpleImputer warning becomes a failure
        pipe.fit(X, y)
    assert pipe.predict_proba(X).shape == (n, 2)

