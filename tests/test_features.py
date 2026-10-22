"""Feature engineering and leakage control."""
from __future__ import annotations

import numpy as np

from src import features


def test_no_leakage_in_model_features():
    model_cols = set(features.NUMERIC_FEATURES) | set(features.CATEGORICAL_FEATURES)
    assert not model_cols & set(features.LEAKAGE_COLUMNS)
    # The lender's own assessment is deliberately excluded from the main model.
    assert not model_cols & set(features.LENDER_MODEL_COLUMNS)


def test_engineer_features_builds_ratios(synthetic_loans):
    eng = features.engineer_features(synthetic_loans)
    for col in ("loan_amnt_to_income", "installment_to_income", "revol_bal_to_income", "limit_utilisation"):
        assert col in eng.columns
        assert eng[col].notna().any()
    assert set(eng["term"].unique()) <= {"36", "60", "missing"}
    assert np.isfinite(eng["credit_history_months"].dropna()).all()
    assert (eng["credit_history_months"].dropna() >= 0).all()


def test_build_model_matrix_shape(synthetic_loans):
    eng = features.engineer_features(synthetic_loans)
    X, y, num, cat = features.build_model_matrix(eng)
    assert list(X.columns) == num + cat
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y) == len(eng)
    assert not X[num].isna().all().any()


def test_winsorization_bounds_are_monotone(synthetic_loans):
    eng = features.engineer_features(synthetic_loans)
    X, _, num, _ = features.build_model_matrix(eng)
    bounds = features.fit_clip_bounds(X, num)
    clipped = features.apply_clip_bounds(X, bounds)
    for col, (lo, hi) in bounds.items():
        assert lo <= hi
        assert clipped[col].max() <= hi + 1e-9
        assert clipped[col].min() >= lo - 1e-9
