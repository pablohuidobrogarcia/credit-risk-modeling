"""Discrimination, calibration and stability metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import metrics as M


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert M.auc(y, p) == 1.0
    assert M.gini(y, p) == 1.0
    assert M.ks(y, p) == 1.0


def test_random_scores_have_no_power():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 6000)
    p = rng.random(6000)
    assert abs(M.auc(y, p) - 0.5) < 0.03


def test_gini_is_2auc_minus_1():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 3000)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, 3000), 0.001, 0.999)
    assert abs(M.gini(y, p) - (2 * M.auc(y, p) - 1)) < 1e-12


def test_psi_zero_when_identical_and_large_when_shifted():
    rng = np.random.default_rng(2)
    a = rng.normal(size=8000)
    b = rng.normal(1.2, 1.0, size=8000)
    assert M.psi(a, a) < 1e-6
    assert M.psi(a, b) > 0.25


def test_psi_report_flags_bands():
    ref = pd.DataFrame({"v": np.random.default_rng(3).normal(size=5000)})
    cmp = pd.DataFrame({"v": np.random.default_rng(4).normal(0.05, 1.0, size=5000)})
    tab = M.psi_report(ref, cmp, numeric=["v"])
    assert list(tab.columns) == ["variable", "psi", "interpretation"]
    assert tab.loc[0, "interpretation"] in {"stable", "monitor", "unstable"}


def test_calibration_table_bins_and_diff():
    rng = np.random.default_rng(5)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    tab = M.calibration_table(y, p, n_bins=10)
    assert {"n", "predicted", "observed", "diff"} <= set(tab.columns)
    assert len(tab) == 10
    assert np.allclose(tab["diff"], tab["predicted"] - tab["observed"])


def test_confusion_at_cutoff_counts():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.7, 0.2])
    out = M.confusion_at_cutoff(y, p, cutoff=0.5)
    assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (1, 1, 1, 1)
    assert 0.0 <= out["approval_rate"] <= 1.0
