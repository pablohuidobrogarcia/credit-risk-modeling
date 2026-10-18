"""Weight of Evidence (WoE) binning and Information Value (IV).

Self-contained implementation of a standard credit-scoring binning pipeline:

1. numeric variables start from quantile bins;
2. bins that are too small are merged into their nearest neighbour;
3. bins are merged until WoE is (near) monotonic and at most ``max_bins``
   remain;
4. categorical variables are WoE-encoded and rare levels pooled into a single
   ``__other__`` bucket, then merged down to ``max_bins`` levels.

Fitting returns reusable :class:`WoEEncoder` objects. They can be applied to an
out-of-time sample without re-learning anything from it, which keeps the
out-of-time validation honest (no leakage of the future into the binning).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

EPS = 1e-6
MISSING_LABEL = "__missing__"
OTHER_LABEL = "__other__"

IV_BANDS = [
    (0.0, 0.02, "useless"),
    (0.02, 0.10, "weak"),
    (0.10, 0.30, "medium"),
    (0.30, 0.50, "strong"),
    (0.50, np.inf, "suspicious (too strong)"),
]


def iv_band(iv: float) -> str:
    """Human-readable IV band (standard credit-risk convention)."""
    for lo, hi, label in IV_BANDS:
        if lo <= iv < hi:
            return label
    return "unknown"


def bin_table(codes: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Bad/good counts and WoE/IV contributions for each bin code."""
    df = pd.DataFrame({"code": np.asarray(codes), "y": np.asarray(y, dtype=float)})
    g = df.groupby("code")["y"].agg(n="size", bad="sum").reset_index()
    g["good"] = g["n"] - g["bad"]
    total_bad = max(g["bad"].sum(), EPS)
    total_good = max(g["good"].sum(), EPS)
    g["dist_bad"] = g["bad"] / total_bad
    g["dist_good"] = g["good"] / total_good
    g["woe"] = np.log(g["dist_bad"].clip(lower=EPS) / g["dist_good"].clip(lower=EPS))
    g["iv"] = (g["dist_bad"] - g["dist_good"]) * g["woe"]
    g["bad_rate"] = g["bad"] / g["n"]
    return g


# ---------------------------------------------------------------------------
# Numeric binning
# ---------------------------------------------------------------------------
def _initial_edges(x: pd.Series, n_bins: int) -> np.ndarray:
    xv = pd.to_numeric(x, errors="coerce").dropna()
    if xv.nunique() <= 1:
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.quantile(xv, np.linspace(0, 1, n_bins + 1))).astype(float)
    if len(edges) < 3:
        edges = np.unique(np.quantile(xv, [0.0, 0.5, 1.0])).astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _numeric_raw_codes(x: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Integer bin codes; NaN is given the extra final code (missing bucket)."""
    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    internal = edges[1:-1]
    safe = np.where(np.isnan(xv), -np.inf, xv)
    codes = np.digitize(safe, internal, right=True).astype(int)
    nan_mask = np.isnan(xv)
    if nan_mask.any():
        codes = np.where(nan_mask, len(edges) - 1, codes)
    return codes


def _codes_from_groups(raw_codes: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    mapping = {int(b): gi for gi, group in enumerate(groups) for b in group}
    return np.array([mapping.get(int(b), 0) for b in raw_codes], dtype=int)


def _nearest_merge_index(woe: np.ndarray, pos: int) -> int:
    """Index ``i`` such that merging groups ``i`` and ``i+1`` is closest in WoE."""
    cands = []
    if pos > 0:
        cands.append((abs(woe[pos] - woe[pos - 1]), pos - 1))
    if pos < len(woe) - 1:
        cands.append((abs(woe[pos] - woe[pos + 1]), pos))
    return min(cands)[1]


def _merge_until_stable(
    raw_codes: np.ndarray,
    y: np.ndarray,
    groups: List[List[int]],
    min_bin_pct: float,
    max_bins: int,
) -> List[List[int]]:
    n_total = len(y)
    min_n = max(int(min_bin_pct * n_total), 20)
    for _ in range(len(raw_codes) + 5):
        if len(groups) <= 1:
            break
        codes = _codes_from_groups(raw_codes, groups)
        tab = bin_table(codes, y).set_index("code").reindex(range(len(groups))).fillna(0.0)
        n = tab["n"].to_numpy()
        woe = tab["woe"].to_numpy()
        if len(groups) > 2 and (n < min_n).any():
            tiny = int(np.argmin(n))
            groups = _merge(groups, _nearest_merge_index(woe, tiny))
            continue
        diffs = np.diff(woe)
        monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
        if (not monotonic or len(groups) > max_bins) and len(diffs) > 0:
            groups = _merge(groups, int(np.argmin(np.abs(diffs))))
            continue
        break
    return groups


def _merge(groups: List[List[int]], i: int) -> List[List[int]]:
    i = max(0, min(i, len(groups) - 2))
    return groups[:i] + [groups[i] + groups[i + 1]] + groups[i + 2 :]


# ---------------------------------------------------------------------------
# Categorical binning
# ---------------------------------------------------------------------------
def _merge_categorical_groups(
    values: np.ndarray,
    y: np.ndarray,
    group_of: Dict[str, str],
    max_bins: int,
) -> Dict[str, str]:
    """Merge the closest-WoE category groups until ``max_bins`` remain."""
    for _ in range(len(group_of) + 5):
        labels = np.array([group_of[v] for v in values])
        uniq = sorted(set(labels))
        if len(uniq) <= max_bins or len(uniq) <= 1:
            break
        tab = bin_table(pd.factorize(labels)[0], y).set_index("code").reindex(range(len(uniq))).fillna(0.0)
        woe = tab["woe"].to_numpy()
        # find the globally closest pair of groups by WoE
        best = (np.inf, None)
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                d = abs(woe[i] - woe[j])
                if d < best[0]:
                    best = (d, (uniq[i], uniq[j]))
        a, b = best[1]  # type: ignore[misc]
        group_of = {k: (a if v == b else v) for k, v in group_of.items()}
    return group_of


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------
@dataclass
class WoEEncoder:
    """Fitted WoE transformation for a single variable."""

    feature: str
    kind: str  # "numeric" | "categorical"
    iv: float = 0.0
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    # numeric internals
    edges: Optional[np.ndarray] = None
    code_to_group: Dict[int, int] = field(default_factory=dict)
    group_woe: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # categorical internals
    cat_to_group: Dict[str, str] = field(default_factory=dict)
    group_woe_cat: Dict[str, float] = field(default_factory=dict)
    default_woe: float = 0.0

    def transform(self, s: pd.Series) -> np.ndarray:
        """Map raw values to WoE (unseen levels fall back to a neutral WoE)."""
        if self.kind == "numeric":
            raw = _numeric_raw_codes(s, self.edges)  # type: ignore[arg-type]
            grp = np.array([self.code_to_group.get(int(c), 0) for c in raw], dtype=int)
            return self.group_woe[grp]
        keys = _to_cat_strings(s, self.cat_to_group)
        groups = np.array([self.cat_to_group.get(k, OTHER_LABEL) for k in keys])
        return np.array([self.group_woe_cat.get(g, self.default_woe) for g in groups], dtype=float)

    @property
    def bin_labels(self) -> List[str]:
        return list(self.table["bin"].astype(str)) if "bin" in self.table.columns else []


def _to_cat_strings(s: pd.Series, known: Dict[str, str]) -> List[str]:
    out = []
    for v in s.tolist():
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            out.append(MISSING_LABEL)
        else:
            text = str(v).strip()
            out.append(MISSING_LABEL if text == "" or text.lower() in {"nan", "none", "<na>"} else text)
    return out


def fit_numeric(
    x: pd.Series,
    y: pd.Series,
    feature: str,
    n_bins: int = 20,
    min_bin_pct: float = 0.05,
    max_bins: int = 8,
) -> WoEEncoder:
    """Fit quantile + monotonic-merge binning for a numeric variable."""
    yv = np.asarray(y, dtype=float)
    xv = pd.to_numeric(x, errors="coerce")
    has_missing = bool(xv.isna().any())
    edges = _initial_edges(xv, n_bins)
    raw = _numeric_raw_codes(xv, edges)
    n_interval = len(edges) - 1
    n_codes = n_interval + (1 if has_missing else 0)
    groups = [[i] for i in range(n_codes)]
    groups = _merge_until_stable(raw, yv, groups, min_bin_pct, max_bins)

    def label(code: int) -> str:
        if has_missing and code == n_interval:
            return MISSING_LABEL
        return f"({edges[code]:.4g}, {edges[code + 1]:.4g}]"

    group_labels = [" & ".join(label(b) for b in g) for g in groups]

    codes = _codes_from_groups(raw, groups)
    tab = bin_table(codes, yv).sort_values("code").reset_index(drop=True)
    tab["bin"] = [group_labels[int(c)] for c in tab["code"]]
    code_to_group = {int(b): gi for gi, g in enumerate(groups) for b in g}
    group_woe = np.zeros(len(groups))
    for _, row in tab.iterrows():
        group_woe[int(row["code"])] = float(row["woe"])
    return WoEEncoder(
        feature=feature,
        kind="numeric",
        iv=float(tab["iv"].sum()),
        table=tab[["code", "bin", "n", "bad", "good", "bad_rate", "woe", "iv"]],
        edges=edges,
        code_to_group=code_to_group,
        group_woe=group_woe,
    )


def fit_categorical(
    x: pd.Series,
    y: pd.Series,
    feature: str,
    min_bin_pct: float = 0.02,
    max_bins: int = 8,
) -> WoEEncoder:
    """Fit WoE encoding for a categorical variable with rare-level pooling."""
    yv = np.asarray(y, dtype=float)
    values = _to_cat_strings(x, {})
    min_n = max(int(min_bin_pct * len(values)), 50)
    counts = pd.Series(values).value_counts()
    group_of = {v: v for v in counts.index}
    for v in counts.index:
        if counts[v] < min_n and v != MISSING_LABEL:
            group_of[v] = OTHER_LABEL
    group_of = _merge_categorical_groups(np.array(values), yv, group_of, max_bins)

    labels = np.array([group_of[v] for v in values])
    codes, uniques = pd.factorize(labels)
    tab = bin_table(codes, yv)
    tab["bin"] = [uniques[int(c)] for c in tab["code"]]
    group_woe_cat = dict(zip(tab["bin"], tab["woe"]))
    default = group_woe_cat.get(OTHER_LABEL, 0.0)
    return WoEEncoder(
        feature=feature,
        kind="categorical",
        iv=float(tab["iv"].sum()),
        table=tab[["code", "bin", "n", "bad", "good", "bad_rate", "woe", "iv"]].sort_values("bin").reset_index(drop=True),
        cat_to_group=group_of,
        group_woe_cat={k: float(v) for k, v in group_woe_cat.items()},
        default_woe=float(default),
    )


def fit_encoders(
    X: pd.DataFrame,
    y: pd.Series,
    numeric: Sequence[str],
    categorical: Sequence[str],
    n_bins: int = 20,
    min_bin_pct: float = 0.05,
    max_bins: int = 8,
) -> Dict[str, WoEEncoder]:
    """Fit one encoder per variable."""
    encoders: Dict[str, WoEEncoder] = {}
    for col in numeric:
        encoders[col] = fit_numeric(X[col], y, col, n_bins=n_bins, min_bin_pct=min_bin_pct, max_bins=max_bins)
    for col in categorical:
        encoders[col] = fit_categorical(X[col], y, col, min_bin_pct=0.02, max_bins=max_bins)
    return encoders


def transform_frame(X: pd.DataFrame, encoders: Dict[str, WoEEncoder]) -> pd.DataFrame:
    """Apply fitted encoders, returning a WoE matrix (one column per variable)."""
    return pd.DataFrame({f: enc.transform(X[f]) for f, enc in encoders.items()}, index=X.index)


def iv_summary(encoders: Dict[str, WoEEncoder]) -> pd.DataFrame:
    """Information Value table sorted by strength of the variable."""
    rows = [
        {"variable": f, "iv": enc.iv, "n_bins": len(enc.table), "band": iv_band(enc.iv)}
        for f, enc in encoders.items()
    ]
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
