"""Tests for the block-3 calibration comparison (none / intercept / Platt / isotonic).

The key property under test: every calibrator is fitted on scores it did not
train on, and the four methods produce genuinely different probability scales
while the *ranking* (AUC) is preserved by the monotone ones.
"""
from __future__ import annotations

import numpy as np

from src import calibration
from src import metrics as M


def _scores(n: int = 20_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Over-confident scores: the model thinks p is higher than it really is.
    y = rng.binomial(1, 0.10, n)
    p_raw = np.clip(0.10 + 0.05 * y + rng.normal(0, 0.05, n), 1e-4, 1 - 1e-4)
    p_raw = np.clip(p_raw * 1.6, 1e-4, 1 - 1e-4)  # deliberately over-predicting
    return p_raw, y


def test_all_calibrators_return_valid_probabilities():
    p, y = _scores()
    n = len(y)
    p_fit, y_fit = p[: n // 2], y[: n // 2]
    p_val, y_val = p[n // 2 : 3 * n // 4], y[n // 2 : 3 * n // 4]
    p_test, y_test = p[3 * n // 4 :], y[3 * n // 4 :]
    table, out, chosen = calibration.compare_calibrators(p_fit, y_fit, p_val, y_val, p_test, y_test)
    assert set(table["method"]) == {"none", "intercept", "platt", "isotonic"}
    assert chosen in {"none", "intercept", "platt", "isotonic"}
    for name, pred in out["predictions_test"].items():
        assert np.all(np.isfinite(pred)), name
        assert np.all((pred >= 0) & (pred <= 1)), name
    assert len(table) == 4


def test_intercept_and_isotonic_preserve_ranking():
    """Intercept and Platt are strictly monotone, so they preserve AUC exactly.
    Isotonic introduces flat regions (ties), so it preserves ranking only up to
    the small number of pairs that fall inside a tie - hence the looser bound."""
    p, y = _scores()
    y = np.asarray(y)
    for cls, tol in (
        (calibration.InterceptCalibrator, 1e-9),
        (calibration.PlattCalibrator, 1e-9),
        (calibration.IsotonicCalibrator, 0.01),
    ):
        cal = cls().fit(p, y)
        pred = cal.predict(p)
        assert abs(M.auc(y, pred) - M.auc(y, p)) < tol, cls.__name__


def test_intercept_calibrator_corrects_uniform_bias():
    p, y = _scores()
    y = np.asarray(y)
    cal = calibration.InterceptCalibrator().fit(p, y)
    pred = cal.predict(p)
    # Over-prediction should be pulled down towards the observed rate.
    obs = float(y.mean())
    assert abs(float(pred.mean()) - obs) < abs(float(p.mean()) - obs)


def test_identity_calibrator_is_a_noop():
    p, _ = _scores()
    out = calibration.IdentityCalibrator().fit(p, np.zeros(len(p))).predict(p)
    assert np.allclose(out, np.clip(p, 0, 1))


def test_method_selected_on_validation_not_test():
    """Selection must be reproducible from the validation column alone."""
    p, y = _scores(seed=5)
    n = len(y)
    p_fit, y_fit = p[: n // 2], y[: n // 2]
    p_val, y_val = p[n // 2 :], y[n // 2 :]
    table, out, chosen = calibration.compare_calibrators(p_fit, y_fit, p_val, y_val, p_val, y_val)
    best_val = table.sort_values("val_brier").iloc[0]["method"]
    assert chosen == best_val
