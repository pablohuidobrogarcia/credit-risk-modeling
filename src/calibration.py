"""Probability calibration comparison for the PD champion.

The model ranks loans well; the *probability scale* is what the expected-loss and
Monte-Carlo stages consume, so it has to be calibrated. This module compares four
calibration strategies **without ever touching the primary test window**:

===============  ==============================================================
method           definition
===============  ==============================================================
``none``         raw model scores (no calibration).
``intercept``    intercept-only (a.k.a. calibration-in-the-large) shift on the
                 logit scale: fixes a uniform over/under-prediction.
``platt``        Platt scaling: LogisticRegression on the logit of the score.
``isotonic``     isotonic (monotone) regression.
===============  ==============================================================

Every calibrator is fitted on **out-of-sample scores from a later temporal slice
of the training window** - a model trained on 2010-2012 is used to score 2013,
and the calibrator sees only those 2013 scores. A calibrator fitted on the same
rows the model memorised would look artificially good; this design avoids that.
The method is then *chosen* on the 2013 slice (validation data only) and reported
on the primary 2014-2015 test window, where any residual gap is distribution
shift and is reported honestly.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import metrics as M

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Calibrator objects
# ---------------------------------------------------------------------------
class IdentityCalibrator:
    """No-op calibrator (raw scores)."""

    name = "none"

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), 0.0, 1.0)


class InterceptCalibrator:
    """Intercept-only recalibration on the logit scale.

    Equivalent to re-estimating the model's intercept on the calibration slice:
    it corrects a uniform bias but leaves the slope (and ranking) exactly
    untouched. The intercept is found by solving
    ``mean(sigmoid(logit(p) + a)) == mean(y)`` so the fitted slope really is 1
    (a free-slope logistic fit would be Platt scaling, not intercept-only).
    """

    name = "intercept"

    def __init__(self) -> None:
        self.intercept = 0.0

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
        return np.log(p / (1.0 - p))

    def fit(self, p: np.ndarray, y: np.ndarray) -> "InterceptCalibrator":
        z = self._logit(p)
        target = float(np.mean(np.asarray(y, dtype=float)))

        def mean_p(a: float) -> float:
            return float(np.mean(1.0 / (1.0 + np.exp(-(z + a)))))

        lo, hi = -50.0, 50.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mean_p(mid) < target:
                lo = mid
            else:
                hi = mid
        self.intercept = float(0.5 * (lo + hi))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        z = self._logit(p) + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


class PlattCalibrator:
    """Platt scaling: logistic regression on the logit of the score."""

    name = "platt"

    def __init__(self) -> None:
        self._fit: Optional[LogisticRegression] = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
        return np.log(p / (1.0 - p))

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        z = self._logit(p).reshape(-1, 1)
        self._fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        self._fit.fit(z, np.asarray(y, dtype=int))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        assert self._fit is not None
        return self._fit.predict_proba(self._logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic (monotone) regression, wrapped for the uniform API."""

    name = "isotonic"

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(np.clip(np.asarray(p, dtype=float), 0.0, 1.0), np.asarray(y, dtype=float))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        out = self._iso.predict(np.clip(np.asarray(p, dtype=float), 0.0, 1.0))
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


CALIBRATORS = {
    "none": IdentityCalibrator,
    "intercept": InterceptCalibrator,
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_calibrators(
    p_fit: np.ndarray,
    y_fit: np.ndarray,
    p_val: np.ndarray,
    y_val: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
    methods: Sequence[str] = ("none", "intercept", "platt", "isotonic"),
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """Fit every calibrator on the fit slice and score val + test.

    Parameters
    ----------
    p_fit, y_fit:
        Out-of-sample scores and outcomes from the *calibration* slice (a later
        temporal slice of the training window).
    p_val, y_val:
        Out-of-sample scores and outcomes used to **choose** the method. Kept
        separate from the test window so selection never sees the test data.
    p_test, y_test:
        Primary out-of-time test window (reported, never used for selection).

    Returns
    -------
    (table, chosen_predictions, chosen_name)
        ``table`` has one row per method with calibration-in-the-large, slope,
        Brier and mean prediction on both the selection and test slices.
    """
    rows: list[Dict[str, Any]] = []
    preds_test: Dict[str, np.ndarray] = {}
    preds_val: Dict[str, np.ndarray] = {}
    for name in methods:
        cal = CALIBRATORS[name]()
        cal.fit(p_fit, y_fit)
        pt = cal.predict(p_test)
        pv = cal.predict(p_val)
        preds_test[name] = pt
        preds_val[name] = pv
        mt = M.calibration_intercept_slope(y_test, pt)
        mv = M.calibration_intercept_slope(y_val, pv)
        rows.append(
            {
                "method": name,
                "val_brier": round(float(mv["brier"]), 5),
                "val_citl": round(float(mv["calibration_in_the_large"]), 4),
                "val_slope": round(float(mv["slope"]), 4),
                "test_brier": round(float(mt["brier"]), 5),
                "test_citl": round(float(mt["calibration_in_the_large"]), 4),
                "test_slope": round(float(mt["slope"]), 4),
                "test_mean_predicted": round(float(mt["mean_predicted"]), 4),
                "test_observed_rate": round(float(mt["observed_rate"]), 4),
                "test_gap": round(float(mt["mean_predicted"] - mt["observed_rate"]), 4),
            }
        )
    table = pd.DataFrame(rows)
    # Choose on the validation slice by Brier (proper score, penalises both
    # miscalibration and over-confidence); ties broken by |CITL|.
    best = table.sort_values(["val_brier", "val_citl"], key=lambda s: s.abs() if s.name == "val_citl" else s)
    chosen = str(best.iloc[0]["method"])
    detail = {
        "fit_slice": "later temporal slice of the training window (out-of-sample scores)",
        "selection_metric": "Brier on the validation slice (val_brier)",
        "chosen": chosen,
        "table": table,
    }
    return table, {"chosen": chosen, "predictions_test": preds_test, "predictions_val": preds_val, "detail": detail}, chosen
