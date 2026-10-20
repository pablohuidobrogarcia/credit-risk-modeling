"""Segmentation: winsorisation, safe silhouette, degenerate-cluster reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import clustering as cl


def _frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(1000.0, 500.0, size=n),
            "c": rng.choice(["x", "y"], n),
        }
    )


def test_preprocessor_clips_the_heavy_tail_before_scaling():
    X = _frame()[["a", "b"]]
    X.iloc[0, 1] = 1e9  # a borrower with an implausible balance
    Z, scaler, medians = cl.impute_and_scale(X)
    assert np.isfinite(Z).all()
    assert abs(Z[0, 1]) < 10  # without winsorisation this would be ~1e6


def test_preprocessor_transform_reuses_fit_statistics():
    X = _frame(400, seed=1)
    pp = cl.Preprocessor().fit(X[["a", "b"]])
    z_fit = pp.transform(X[["a", "b"]])
    z_new = pp.transform(X[["a", "b"]].iloc[:10])
    assert z_fit.shape == (400, 2)
    assert z_new.shape == (10, 2)
    assert np.allclose(z_new, z_fit[:10])


def test_impute_and_scale_imputes_missing_values():
    X = _frame(300, seed=2)
    X.loc[0, "b"] = np.nan
    Z, _, _ = cl.impute_and_scale(X[["a", "b"]])
    assert np.isfinite(Z).all()


def test_silhouette_safe_returns_nan_for_a_single_cluster():
    Z = np.random.default_rng(0).normal(size=(50, 3))
    labels = np.zeros(50, dtype=int)
    assert np.isnan(cl.silhouette_safe(Z, labels))


def test_silhouette_safe_handles_a_dominant_cluster():
    # 995 points in one cluster, 5 in another: a plain random subsample would
    # frequently return a single label and sklearn would raise.
    rng = np.random.default_rng(1)
    Z = np.vstack([rng.normal(size=(995, 2)), rng.normal(6.0, 0.5, size=(5, 2))])
    labels = np.array([0] * 995 + [1] * 5)
    score = cl.silhouette_safe(Z, labels, sample_size=200, seed=3)
    assert np.isfinite(score)
    assert -1.0 <= score <= 1.0


def test_stratified_sample_covers_every_cluster():
    labels = np.array([0] * 995 + [1] * 5)
    idx = cl.stratified_sample_indices(labels, sample_size=200, seed=1)
    assert set(np.unique(labels[idx])) == {0, 1}


def test_select_k_uses_silhouette_and_stability():
    scores = pd.DataFrame(
        {
            "k": [2, 3, 4],
            "silhouette": [0.10, 0.30, 0.12],
            "stability_ari": [0.80, 0.60, 0.90],
            "n_degenerate_clusters": [0, 0, 0],
        }
    )
    # Interior winner on the combined score (silhouette + stability).
    assert cl.select_k(scores) == 3


def test_select_k_avoids_a_bare_grid_boundary():
    scores = pd.DataFrame(
        {
            "k": [2, 3, 4],
            "silhouette": [0.20, 0.199, 0.10],
            "stability_ari": [0.90, 0.899, 0.80],
            "n_degenerate_clusters": [0, 0, 0],
        }
    )
    # k=2 is the grid boundary and only marginally better than k=3 -> interior k.
    assert cl.select_k(scores) == 3


def test_select_k_falls_back_when_scores_are_unusable():
    all_nan = pd.DataFrame(
        {"k": [2, 3], "silhouette": [np.nan, np.nan], "stability_ari": [np.nan, np.nan], "n_degenerate_clusters": [0, 0]}
    )
    assert cl.select_k(all_nan, default=5) == 5
    degenerate = pd.DataFrame(
        {"k": [2, 3], "silhouette": [0.3, 0.3], "stability_ari": [0.9, 0.9], "n_degenerate_clusters": [1, 1]}
    )
    assert cl.select_k(degenerate, default=7) == 7


def test_choose_k_reports_balance_stability_and_degenerate_clusters():
    scores = cl.choose_k(_frame(500, seed=4)[["a", "b"]], k_range=range(2, 5), sample_size=500, seed=4, n_boot=2)
    assert list(scores["k"]) == [2, 3, 4]
    assert {"inertia", "silhouette", "stability_ari", "largest_cluster_share", "n_degenerate_clusters"} <= set(
        scores.columns
    )
    assert (scores["silhouette"].between(-1.0, 1.0)).all()
    assert (scores["largest_cluster_share"] > 0.0).all()
    assert (scores["inertia"].diff().dropna() < 0).all()  # inertia always falls with k


def test_cluster_stability_is_high_for_well_separated_blobs():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(-5, 0.3, size=(150, 2)), rng.normal(5, 0.3, size=(150, 2))])
    ari = cl.cluster_stability(Z, k=2, seed=1, n_boot=3, sample_size=200)
    assert np.isfinite(ari) and ari > 0.9


def test_profile_segments_flags_small_segments():
    df = pd.DataFrame(
        {
            "a": np.arange(200, dtype=float),
            "default_flag": np.concatenate([np.ones(5), np.zeros(195)]),
        }
    )
    labels = np.array([0] * 199 + [1])
    prof = cl.profile_segments(df, labels, ["a"])
    assert set(prof.columns) >= {"segment", "n", "share", "degenerate", "default_rate", "a"}
    assert abs(prof["share"].sum() - 1.0) < 1e-9
    assert bool(prof.loc[prof["segment"] == 1, "degenerate"].iloc[0]) is True
    assert bool(prof.loc[prof["segment"] == 0, "degenerate"].iloc[0]) is False


def test_kmeans_segmenter_predicts_for_new_loans():
    X = _frame(400, seed=5)[["a", "b"]]
    seg = cl.KMeansSegmenter(k=3, seed=5).fit(X)
    labels = seg.predict(X)
    assert labels.shape == (400,)
    assert len(np.unique(labels)) >= 2


def test_subsample_is_deterministic_and_keeps_order():
    X = _frame(1000, seed=6)
    a = cl.subsample(X, 200, seed=7)
    b = cl.subsample(X, 200, seed=7)
    assert len(a) == 200
    assert a.index.equals(b.index)
    assert cl.subsample(X, 5000, seed=7).shape[0] == 1000  # no-op when small enough
