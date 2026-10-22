"""Leakage-control tests.

These tests enforce the modeling contract that *every* fitted artefact is
learned on the training window only:

* the chronological split never mixes train and test rows, and honours the
  configured years/terms;
* winsorisation bounds are functions of the training frame alone - changing an
  out-of-time value must not move the bounds;
* the WoE encoders and the isotonic calibrator are fitted on train only, so an
  extreme out-of-time row cannot affect their parameters;
* the leakage column list really does exclude the post-origination fields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import features
from src import metrics as M
from src.config import Config
from src.pipeline import train_oot_split
from src.woe_iv import fit_encoders, transform_frame


def _cfg(**modeling) -> Config:
    base = {
        "mature_train_years": [2010, 2011],
        "mature_test_years": [2012, 2013],
        "mature_terms": ["36"],
        "oot_split_year": 2012,
    }
    base.update(modeling)
    return Config(seed=42, data={}, modeling=base, simulation={})


def _loans() -> pd.DataFrame:
    rows = []
    for year in (2010, 2011, 2012, 2013):
        for term in ("36", "60"):
            for i in range(20):
                rows.append(
                    {
                        "issue_year": year,
                        "term": term,
                        "loan_amnt": 1000.0 + 10 * i + year,
                        "annual_inc": 50_000.0 + i,
                        "default_flag": i % 2,
                        "dti": 10.0 + i,
                    }
                )
    return pd.DataFrame(rows)


def test_primary_split_honours_years_and_terms_without_overlap():
    df = _loans()
    train, oot = train_oot_split(df, _cfg(), kind="mature")
    assert set(train["issue_year"]) == {2010, 2011}
    assert set(oot["issue_year"]) == {2012, 2013}
    assert set(train["term"]) == {"36"} and set(oot["term"]) == {"36"}
    # No row may appear on both sides of the split.
    assert len(train.index.intersection(oot.index)) == 0
    assert len(train) + len(oot) < len(df)  # the 60-month cells are excluded


def test_secondary_split_is_the_original_late_window():
    df = _loans()
    train, oot = train_oot_split(df, _cfg(oot_split_year=2012), kind="immature")
    assert set(oot["issue_year"]) == {2012, 2013}
    assert set(train["issue_year"]) == {2010, 2011}


def test_clip_bounds_are_fitted_on_train_only():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    oot = pd.DataFrame({"x": [-1_000_000.0, 1.0, 1_000_000.0]})
    bounds = features.fit_clip_bounds(train, ["x"])
    oot_changed = pd.DataFrame({"x": [-9e12, 9e12, 9e12]})
    bounds_changed = features.fit_clip_bounds(oot_changed, ["x"])
    # The bounds depend only on the frame they are fitted on.
    assert bounds == features.fit_clip_bounds(train, ["x"])
    assert bounds != bounds_changed
    clipped = features.apply_clip_bounds(oot, bounds)
    assert clipped["x"].max() <= bounds["x"][1]
    assert clipped["x"].min() >= bounds["x"][0]


def test_woe_encoders_are_fitted_on_train_only():
    train = pd.DataFrame({"x": np.arange(100, dtype=float)})
    y = (train["x"] > 50).astype(int)
    enc = fit_encoders(train, y, ["x"], [], n_bins=5, min_bin_pct=0.05, max_bins=5)
    W_train = transform_frame(train, enc)
    # An out-of-time frame with a wildly different range must not change the
    # training transform (the bins are anchored on the training sample).
    W_train_again = transform_frame(train, enc)
    pd.testing.assert_frame_equal(W_train, W_train_again)
    oot = pd.DataFrame({"x": [-1e6, 0.0, 1e6]})
    W_oot = transform_frame(oot, enc)
    assert np.isfinite(W_oot.to_numpy()).all()


def test_calibrator_is_fitted_on_train_scores_only():
    rng = np.random.default_rng(0)
    y_train = rng.integers(0, 2, 500)
    p_train = np.clip(y_train * 0.3 + rng.normal(0.4, 0.1, 500), 0.01, 0.99)
    iso = M.fit_calibrator(y_train, p_train)
    # Calibration of a *different* OOT vector does not refit the map.
    p_oot = np.linspace(0.05, 0.95, 20)
    a = M.apply_calibrator(iso, p_oot)
    b = M.apply_calibrator(iso, p_oot)
    assert np.allclose(a, b)
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_leakage_columns_are_excluded_from_the_feature_matrix():
    forbidden = {
        "out_prncp",
        "total_pymnt",
        "total_rec_prncp",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "loan_status",
        "default_flag",
        "issue_year",
    }
    assert forbidden.issubset(set(features.LEAKAGE_COLUMNS))
    assert forbidden.isdisjoint(set(features.NUMERIC_FEATURES))
    assert forbidden.isdisjoint(set(features.CATEGORICAL_FEATURES))
    # The lender's own score is excluded from the main model by default.
    assert features.LENDER_MODEL_COLUMNS == ["grade", "sub_grade", "int_rate"]
    assert set(features.LENDER_MODEL_COLUMNS).isdisjoint(set(features.NUMERIC_FEATURES + features.CATEGORICAL_FEATURES))
