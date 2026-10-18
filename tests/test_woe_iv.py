"""Weight-of-Evidence binning and Information Value."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.woe_iv import fit_categorical, fit_numeric, iv_band


@pytest.fixture()
def signal():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-1.6 * x))
    y = pd.Series((rng.random(n) < p).astype(float))
    return pd.Series(x), y


def test_iv_positive_and_monotone_woe(signal):
    x, y = signal
    enc = fit_numeric(x, y, "x", n_bins=20, min_bin_pct=0.05, max_bins=8)
    assert enc.iv > 0.10, "a real signal should have a meaningful IV"
    woe = enc.group_woe
    # Bin codes follow the numeric order, so WoE must be monotone (upward here).
    diffs = np.diff(woe)
    assert np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9)


def test_iv_near_zero_for_noise():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=20_000))
    y = pd.Series(rng.integers(0, 2, size=20_000).astype(float))
    enc = fit_numeric(x, y, "noise", n_bins=10, max_bins=6)
    assert enc.iv < 0.05
    assert iv_band(enc.iv) in {"useless", "weak"}


def test_transform_is_reproducible(signal):
    x, y = signal
    enc = fit_numeric(x, y, "x", max_bins=6)
    w1 = enc.transform(x)
    w2 = enc.transform(x)
    assert np.array_equal(w1, w2)
    # every value must come from the fitted bin table
    assert set(np.round(w1, 10)).issubset(set(np.round(np.asarray(enc.group_woe), 10)))


def test_bin_count_respected():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(size=30_000))
    y = pd.Series((rng.random(30_000) < 0.2).astype(float))
    enc = fit_numeric(x, y, "x", n_bins=20, max_bins=5)
    assert len(enc.table) <= 5


def test_unseen_category_falls_back():
    x = pd.Series(["a", "a", "b", "b", "a", "b"] * 200)
    y = pd.Series(([0, 1] * 600))
    enc = fit_categorical(x, y, "cat", max_bins=3)
    out = enc.transform(pd.Series(["never-seen", "a"]))
    assert out.shape == (2,)
    assert np.isfinite(out).all()
