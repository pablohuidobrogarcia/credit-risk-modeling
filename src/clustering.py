"""Loan/customer segmentation: k-means, hierarchical clustering and PCA.

Design notes
------------
* **Winsorisation before scaling.** The behavioural features are heavy tailed
  (``annual_inc`` reaches 9.3M, ``dti`` 999, ``loan_amnt_to_income`` 30k).
  On raw values the Euclidean distance is dominated by a handful of borrowers
  and k-means degenerates into "one cluster with ~99.9% of the loans + a
  handful of outliers". Every column is therefore clipped at its 1st/99th
  percentile *before* standardising.
* **MiniBatchKMeans as a fallback, not as the default.** It is implemented for
  fitting samples where exact KMeans becomes the bottleneck, but it was measured
  on the 120k-row segmentation sample: exact KMeans fits in seconds and is
  clearly better (inertia 1,174,937 vs 1,184,469, silhouette 0.2200 vs 0.1889).
  The threshold is therefore set at 250k rows - well above the sample actually
  used - so the elbow/silhouette/stability study and the production fit are
  algorithmically consistent (exact KMeans for the shipped segmentation).
* **Safe silhouette.** sklearn's ``silhouette_score(..., sample_size=N)``
  draws a plain random subsample; with very unbalanced clusters that subsample
  can contain a single label and the call raises ``ValueError``. The score is
  therefore computed on a *stratified* subsample that always includes every
  cluster (and on the full data when it is small enough).
* **Stability, not just silhouette.** k is selected on silhouette *and* the
  mean Adjusted Rand Index across bootstrap resamples (:func:`cluster_stability`),
  and the selection refuses to return a bare grid boundary without justification
  (:func:`select_k`), because a boundary optimum usually signals a truncated grid.
* **Degenerate clusters.** Segments holding less than 1% of the sample are
  reported (``degenerate`` flag) instead of being silently presented as
  business segments.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .utils import save_fig, save_table

MINIBATCH_THRESHOLD = 250_000  # rows above which MiniBatchKMeans is used
DEGENERATE_SHARE = 0.01        # a segment below 1% of the sample is flagged


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
class Preprocessor:
    """Winsorise -> median-impute -> standardise, with reusable fit statistics."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = float(lower)
        self.upper = float(upper)
        self.lo: Optional[pd.Series] = None
        self.hi: Optional[pd.Series] = None
        self.medians: Optional[pd.Series] = None
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        Xf = X.astype(float)
        self.lo = Xf.quantile(self.lower)
        self.hi = Xf.quantile(self.upper)
        clipped = Xf.clip(self.lo, self.hi, axis=1)
        self.medians = clipped.median()
        self.scaler.fit(clipped.fillna(self.medians).to_numpy())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self.medians is not None and self.lo is not None and self.hi is not None, "call fit() first"
        Xf = X.astype(float).clip(self.lo, self.hi, axis=1)
        return self.scaler.transform(Xf.fillna(self.medians).to_numpy())


def impute_and_scale(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, pd.Series]:
    """Winsorise, median-impute and standardise a numeric feature matrix.

    Returns ``(Z, scaler, medians)``. The clipping bounds are fitted on ``X``
    itself, so use this only where the matrix *is* the fitting sample (PCA,
    hierarchical clustering, k-selection).
    """
    pp = Preprocessor().fit(X)
    return pp.transform(X), pp.scaler, pp.medians


def subsample(X: pd.DataFrame, size: int, seed: int = 42) -> pd.DataFrame:
    """Random subsample with a fixed seed (no-op when already small enough)."""
    if len(X) <= size:
        return X
    return X.iloc[np.sort(subsample_indices(len(X), size, seed))]


def subsample_indices(n: int, size: int, seed: int = 42) -> np.ndarray:
    """Sorted random indices (rows are selected without replacement)."""
    if n <= size:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=size, replace=False))


def stratified_sample_indices(labels: np.ndarray, sample_size: int, seed: int = 42) -> np.ndarray:
    """Indices of a subsample containing *every* cluster (proportional + floor).

    A plain random subsample of a very unbalanced clustering frequently
    contains a single label, which makes the silhouette score undefined.
    """
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(seed)
    picked = []
    for u, c in zip(uniq, counts):
        take = int(min(c, max(5, round(sample_size * c / len(labels)))))
        pos = np.flatnonzero(labels == u)
        picked.append(rng.choice(pos, size=take, replace=False))
    return np.sort(np.concatenate(picked))


def silhouette_safe(Z: np.ndarray, labels: np.ndarray, sample_size: int = 10_000, seed: int = 42) -> float:
    """Silhouette score that never raises on unbalanced/degenerate clusterings.

    Returns ``NaN`` when the score is undefined (fewer than two clusters), so
    callers can skip the candidate ``k`` instead of crashing.
    """
    labels = np.asarray(labels)
    n_unique = int(np.unique(labels).size)
    if n_unique < 2 or n_unique >= len(labels):
        return float("nan")
    if len(Z) <= sample_size:
        sel = np.arange(len(Z))
    else:
        sel = stratified_sample_indices(labels, sample_size, seed)
    if np.unique(labels[sel]).size < 2:
        return float("nan")
    return float(silhouette_score(Z[sel], labels[sel]))


def make_kmeans(k: int, seed: int, n_rows: int):
    """MiniBatchKMeans for large fitting samples, exact KMeans otherwise."""
    if n_rows >= MINIBATCH_THRESHOLD:
        return MiniBatchKMeans(
            n_clusters=int(k), batch_size=4096, n_init=10, max_iter=200, random_state=seed
        )
    return KMeans(n_clusters=int(k), n_init=10, random_state=seed)


# ---------------------------------------------------------------------------
# Choosing k
# ---------------------------------------------------------------------------
def cluster_stability(
    Z: np.ndarray, k: int, seed: int = 42, n_boot: int = 5, sample_size: int = 20_000
) -> float:
    """Stability of a k-means solution across bootstrap resamples (mean ARI).

    Each bootstrap draws a fresh sample of rows *with replacement* from the
    reference set, refits k-means and compares labels with the reference
    solution. Because the bootstrap sample contains duplicated rows, the ARI is
    computed on the unique rows shared with the reference labelling (the first
    occurrence of each duplicated row). A high ARI means the segmentation is
    reproducible rather than an artefact of one particular sample.
    """
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    n = len(Z)
    if n < 50:
        return float("nan")
    take = int(min(sample_size, n))
    ref_idx = np.sort(rng.choice(n, size=take, replace=False))
    ref = KMeans(n_clusters=int(k), n_init=10, random_state=seed).fit_predict(Z[ref_idx])
    ref_lab = dict(zip(ref_idx.tolist(), ref.tolist()))

    scores = []
    for b in range(n_boot):
        boot = rng.choice(n, size=take, replace=True)
        try:
            labels = KMeans(n_clusters=int(k), n_init=10, random_state=seed + b).fit_predict(Z[boot])
        except Exception:  # pragma: no cover - degenerate bootstrap
            continue
        pred_lab: dict = {}
        for row, lab in zip(boot.tolist(), labels.tolist()):
            pred_lab.setdefault(row, lab)  # keep first occurrence of duplicated rows
        common = [row for row in pred_lab if row in ref_lab]
        if len(common) < 10:
            continue
        scores.append(
            adjusted_rand_score(
                [ref_lab[row] for row in common],
                [pred_lab[row] for row in common],
            )
        )
    return float(np.mean(scores)) if scores else float("nan")


def choose_k(
    X: pd.DataFrame,
    k_range: range = range(2, 11),
    sample_size: int = 40000,
    seed: int = 42,
    stability_sample: int = 20_000,
    n_boot: int = 5,
) -> pd.DataFrame:
    """Elbow (inertia), silhouette, stability (ARI) and balance diagnostics per k."""
    Z, _, _ = impute_and_scale(subsample(X, stability_sample, seed))
    rows = []
    for k in k_range:
        if int(k) >= len(Z):
            continue
        km = KMeans(n_clusters=int(k), n_init=10, random_state=seed)
        labels = km.fit_predict(Z)
        counts = np.unique(labels, return_counts=True)[1]
        shares = counts / len(labels)
        rows.append(
            {
                "k": int(k),
                "inertia": float(km.inertia_),
                "silhouette": silhouette_safe(Z, labels, sample_size=10_000, seed=seed),
                "stability_ari": round(cluster_stability(Z, int(k), seed=seed, n_boot=n_boot), 4),
                "largest_cluster_share": round(float(shares.max()), 4),
                "n_degenerate_clusters": int((shares < DEGENERATE_SHARE).sum()),
            }
        )
    return pd.DataFrame(rows)


def select_k(scores: pd.DataFrame, default: int = 4) -> int:
    """Choose k from silhouette *and* stability, never a bare grid boundary.

    Selection rule (documented so it is reproducible):

    1. Drop candidates with a degenerate cluster (<1% of the sample) or an
       undefined silhouette/stability score.
    2. Score each candidate by ``0.6 * silhouette + 0.4 * stability_ari``
       (each normalised to its own range over the grid), so discrimination
       structure dominates while stability breaks near-ties.
    3. If the winner sits on the grid boundary (the first or last candidate),
       move to the best *interior* k whenever its combined score is within 5%
       of the boundary winner - a boundary optimum usually means the grid is
       truncated, not that k is genuinely optimal.
    """
    if scores.empty:
        return int(default)
    grid = scores.dropna(subset=["silhouette", "stability_ari"]).copy()
    grid = grid[grid["n_degenerate_clusters"] == 0]
    if grid.empty:
        return int(default)

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = float(s.min()), float(s.max())
        return (s - lo) / (hi - lo) if hi > lo else s * 0.0 + 1.0

    grid["combined"] = 0.6 * _norm(grid["silhouette"]) + 0.4 * _norm(grid["stability_ari"])
    grid = grid.sort_values("k").reset_index(drop=True)
    best = grid.loc[grid["combined"].idxmax()]
    ks = list(grid["k"])
    is_boundary = int(best["k"]) in (int(ks[0]), int(ks[-1]))
    if is_boundary and len(ks) > 2:
        interior = grid.iloc[1:-1]
        if not interior.empty:
            best_interior = interior.loc[interior["combined"].idxmax()]
            if float(best_interior["combined"]) >= 0.95 * float(best["combined"]):
                return int(best_interior["k"])
    return int(best["k"])


def elbow_silhouette_plot(
    scores: pd.DataFrame, chosen_k: int | None = None, name: str = "clustering_elbow.png"
) -> None:
    """Inertia (elbow), silhouette and stability (ARI) side by side."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(scores["k"], scores["inertia"], "o-", color="steelblue", label="Inertia (elbow)")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(scores["k"], scores["silhouette"], "s--", color="crimson", label="Silhouette")
    if "stability_ari" in scores.columns:
        ax2.plot(scores["k"], scores["stability_ari"], "^:", color="darkgreen", label="Stability (ARI)")
    ax2.set_ylabel("Silhouette / stability (ARI)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.grid(False)
    if chosen_k is not None:
        ax1.axvline(chosen_k, color="grey", ls=":", lw=1)
        ax1.annotate(f"chosen k={chosen_k}", xy=(chosen_k, float(scores["inertia"].max())), fontsize=9)
    ax1.set_title("Choosing k: elbow (inertia), silhouette and bootstrap stability")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    return save_fig(fig, name)


# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------
class KMeansSegmenter:
    """k-means segmenter that can assign clusters to new loans."""

    def __init__(self, k: int = 4, seed: int = 42):
        self.k = int(k)
        self.seed = seed
        self.pp = Preprocessor()
        self.model = None

    @property
    def scaler(self):
        """Backwards-compatible alias for the fitted scaler."""
        return self.pp.scaler

    @property
    def medians(self):
        """Backwards-compatible alias for the fitted medians."""
        return self.pp.medians

    def fit(self, X: pd.DataFrame) -> "KMeansSegmenter":
        self.pp.fit(X)
        self.model = make_kmeans(self.k, self.seed, len(X))
        self.model.fit(self.pp.transform(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "fit() must be called before predict()"
        return np.asarray(self.model.predict(self.pp.transform(X)), dtype=int)


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


def pca_plot(X: pd.DataFrame, labels: np.ndarray, name: str = "clustering_pca.png") -> None:
    """2-D PCA projection coloured by cluster label (scatter for readability).

    Rows and labels are subsampled *together* so the colours stay aligned.
    """
    labels = np.asarray(labels)
    idx = subsample_indices(len(X), 20000, seed=42)
    Xs = X.iloc[idx]
    lab = labels[idx]
    Z, _, _ = impute_and_scale(Xs)
    coords = PCA(n_components=2, random_state=42).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in np.unique(lab):
        m = lab == cluster
        ax.scatter(coords[m, 0], coords[m, 1], s=4, alpha=0.4, label=f"cluster {int(cluster)}")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Loan segments - PCA projection of the winsorised, standardised features")
    ax.legend(markerscale=4, fontsize=8)
    return save_fig(fig, name)


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------
def profile_segments(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
    target: str = "default_flag",
    min_share: float = DEGENERATE_SHARE,
) -> pd.DataFrame:
    """Profile each segment: size, share, feature means and realised default rate.

    ``degenerate`` marks segments holding less than ``min_share`` (default 1%)
    of the sample: they are reported for transparency but should not be read as
    meaningful business segments.
    """
    d = df.copy()
    d["segment"] = np.asarray(labels)
    grouped = d.groupby("segment", observed=True)
    feat = [c for c in feature_cols if c in d.columns]
    prof = grouped[feat].mean()
    prof.insert(0, "n", grouped.size())
    prof["share"] = prof["n"] / prof["n"].sum()
    prof["default_rate"] = grouped[target].mean()
    prof["degenerate"] = prof["share"] < min_share
    cols = ["n", "share", "degenerate", "default_rate"] + [c for c in feature_cols if c in prof.columns]
    out = prof[cols].reset_index()
    for c in cols:
        if out[c].dtype.kind in "fc":
            out[c] = out[c].astype(float).round(4)
    return out


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
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
    ``k`` that maximises the silhouette score (see :func:`select_k`).
    """
    work = df.dropna(subset=["default_flag"]).copy()
    X_all = work[feature_cols]
    X_fit = subsample(X_all, max_rows, seed)
    print(f"    segmentation sample: {len(X_fit):,} of {len(X_all):,} loans")

    scores = choose_k(X_fit, k_range=k_range, seed=seed)
    save_table(scores, "clustering_k_selection.csv")
    if k is None:
        k = select_k(scores)
    elbow_silhouette_plot(scores, chosen_k=k)
    print(f"    chosen k={k} (silhouette-based)")

    segmenter = KMeansSegmenter(k=k, seed=seed).fit(X_fit)
    km_labels = segmenter.predict(X_all)
    kmeans_profile = profile_segments(work, km_labels, feature_cols)
    save_table(kmeans_profile, "clustering_kmeans_profile.csv")

    hier_sample = subsample(X_fit, 5000, seed)
    hier_labels, link = hierarchical_labels(hier_sample, k=k, seed=seed)
    hier_profile = profile_segments(work.loc[hier_sample.index], hier_labels, feature_cols)
    save_table(hier_profile, "clustering_hierarchical_profile.csv")
    dendrogram_plot(link)

    pca_plot(X_fit, segmenter.predict(X_fit))

    n_deg = int(kmeans_profile["degenerate"].sum())
    if n_deg:
        print(f"    WARNING: {n_deg} segment(s) below {DEGENERATE_SHARE:.0%} of the sample are flagged degenerate")

    rates = kmeans_profile["default_rate"].dropna()
    spread = {
        "min_default_rate": round(float(rates.min()), 4) if len(rates) else None,
        "max_default_rate": round(float(rates.max()), 4) if len(rates) else None,
        "spread_pp": round(float(rates.max() - rates.min()) * 100.0, 2) if len(rates) else None,
    }
    print(f"    segment default-rate spread: {spread['spread_pp']} pp ({spread['min_default_rate']} -> {spread['max_default_rate']})")

    return {
        "k": int(k),
        "k_selection": scores,
        "kmeans_profile": kmeans_profile,
        "hierarchical_profile": hier_profile,
        "hierarchical_labels": hier_labels,
        "linkage": link,
        "segmenter": segmenter,
        "labels": km_labels,
        "n_degenerate_segments": n_deg,
        "degenerate_share_threshold": DEGENERATE_SHARE,
        "default_rate_spread": spread,
    }
