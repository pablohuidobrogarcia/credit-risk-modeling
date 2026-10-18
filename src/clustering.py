"""Loan/customer segmentation: k-means, hierarchical clustering and PCA."""
from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .utils import save_fig, save_table


def impute_and_scale(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, pd.Series]:
    """Median-impute then standardise a numeric feature matrix."""
    Xf = X.astype(float)
    medians = Xf.median()
    scaler = StandardScaler()
    Z = scaler.fit_transform(Xf.fillna(medians).to_numpy())
    return Z, scaler, medians


def subsample(X: pd.DataFrame, size: int, seed: int = 42) -> pd.DataFrame:
    """Random subsample with a fixed seed (no-op when already small enough)."""
    if len(X) <= size:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=size, replace=False)
    return X.iloc[np.sort(idx)]


def choose_k(
    X: pd.DataFrame,
    k_range: range = range(2, 11),
    sample_size: int = 40000,
    seed: int = 42,
) -> pd.DataFrame:
    """Elbow (inertia) and silhouette scores for a range of ``k``."""
    Z, _, _ = impute_and_scale(subsample(X, sample_size, seed))
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=int(k), n_init=10, random_state=seed)
        labels = km.fit_predict(Z)
        rows.append(
            {
                "k": int(k),
                "inertia": float(km.inertia_),
                "silhouette": float(
                    silhouette_score(Z, labels, sample_size=min(10000, len(Z)), random_state=seed)
                ),
            }
        )
    return pd.DataFrame(rows)


def elbow_silhouette_plot(scores: pd.DataFrame, chosen_k: int | None = None, name: str = "clustering_elbow.png"):
    """Inertia (elbow) and silhouette score side by side."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(scores["k"], scores["inertia"], "o-", color="steelblue", label="Inertia (elbow)")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(scores["k"], scores["silhouette"], "s--", color="crimson", label="Silhouette")
    ax2.set_ylabel("Silhouette score", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.grid(False)
    if chosen_k is not None:
        ax1.axvline(chosen_k, color="grey", ls=":", lw=1)
        ax1.annotate(f"chosen k={chosen_k}", xy=(chosen_k, float(scores["inertia"].max())), fontsize=9)
    ax1.set_title("Choosing k: elbow (inertia) and silhouette")
    return save_fig(fig, name)


class KMeansSegmenter:
    """k-means segmenter that can assign clusters to new loans."""

    def __init__(self, k: int = 4, seed: int = 42):
        self.k = k
        self.seed = seed
        self.scaler = StandardScaler()
        self.medians: pd.Series | None = None
        self.model = KMeans(n_clusters=k, n_init=10, random_state=seed)

    def fit(self, X: pd.DataFrame) -> "KMeansSegmenter":
        Xf = X.astype(float)
        self.medians = Xf.median()
        Z = self.scaler.fit_transform(Xf.fillna(self.medians).to_numpy())
        self.model.fit(Z)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.medians is not None, "fit() must be called before predict()"
        Xf = X.astype(float).fillna(self.medians)
        return self.model.predict(self.scaler.transform(Xf.to_numpy()))


def hierarchical_labels(
    X: pd.DataFrame, k: int, sample_size: int = 5000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Ward hierarchical clustering on a subsample; returns (labels, linkage)."""
    Z, _, _ = impute_and_scale(subsample(X, sample_size, seed))
    link = linkage(Z, method="ward")
    labels = fcluster(link, t=k, criterion="maxclust")
    return labels, link


def dendrogram_plot(link: np.ndarray, name: str = "clustering_dendrogram.png", truncate: int = 30):
    """Truncated dendrogram of the Ward linkage."""
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(link, truncate_mode="lastp", p=truncate, ax=ax, no_labels=True, color_threshold=0)
    ax.set_title("Hierarchical clustering (Ward) - truncated dendrogram")
    ax.set_ylabel("Distance")
    return save_fig(fig, name)


def pca_plot(X: pd.DataFrame, labels: np.ndarray, name: str = "clustering_pca.png"):
    """2-D PCA projection coloured by cluster label (scatter for readability)."""
    Xs = subsample(X, 20000) if len(X) > 20000 else X
    lab = labels[: len(Xs)] if len(labels) >= len(Xs) else labels
    Z, _, _ = impute_and_scale(Xs)
    coords = PCA(n_components=2, random_state=42).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in np.unique(lab):
        m = lab == cluster
        ax.scatter(coords[m, 0], coords[m, 1], s=4, alpha=0.4, label=f"cluster {int(cluster)}")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Loan segments - PCA projection of the standardised features")
    ax.legend(markerscale=4, fontsize=8)
    return save_fig(fig, name)


def profile_segments(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
    target: str = "default_flag",
) -> pd.DataFrame:
    """Profile each segment: size, share, feature means and realised default rate."""
    d = df.copy()
    d["segment"] = labels
    agg = {c: "mean" for c in feature_cols if c in d.columns}
    prof = d.groupby("segment").agg(n=("segment", "size"), **agg)
    prof["share"] = prof["n"] / prof["n"].sum()
    prof["default_rate"] = d.groupby("segment")[target].mean()
    cols = ["n", "share", "default_rate"] + [c for c in feature_cols if c in prof.columns]
    return prof[cols].round(4).reset_index()


def run_clustering(
    df: pd.DataFrame,
    feature_cols: List[str],
    k: int | None = None,
    k_range: range = range(2, 11),
    seed: int = 42,
    max_rows: int = 120000,
) -> Dict[str, object]:
    """End-to-end segmentation pipeline (k-means + hierarchical + PCA).

    When ``k`` is ``None`` the number of clusters is chosen automatically as the
    ``k`` that maximises the silhouette score.
    """
    work = df.dropna(subset=["default_flag"]).copy()
    X_all = work[feature_cols]
    X_fit = subsample(X_all, max_rows, seed)

    scores = choose_k(X_fit, k_range=k_range, seed=seed)
    save_table(scores, "clustering_k_selection.csv")
    if k is None:
        k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    elbow_silhouette_plot(scores, chosen_k=k)

    segmenter = KMeansSegmenter(k=k, seed=seed).fit(subsample(X_fit, 40000, seed))
    km_labels = segmenter.predict(X_all)
    kmeans_profile = profile_segments(work, km_labels, feature_cols)
    save_table(kmeans_profile, "clustering_kmeans_profile.csv")

    hier_labels, link = hierarchical_labels(X_fit, k=k, seed=seed)
    hier_profile = profile_segments(work.loc[X_fit.index], hier_labels, feature_cols)
    save_table(hier_profile, "clustering_hierarchical_profile.csv")
    dendrogram_plot(link)

    pca_plot(X_fit, segmenter.predict(X_fit))

    return {
        "k": int(k),
        "k_selection": scores,
        "kmeans_profile": kmeans_profile,
        "hierarchical_profile": hier_profile,
        "hierarchical_labels": hier_labels,
        "linkage": link,
        "segmenter": segmenter,
        "labels": km_labels,
    }
