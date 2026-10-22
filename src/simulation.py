"""Monte Carlo portfolio-loss simulation.

Model
-----
A single-factor (Vasicek / Merton) credit-risk model. Borrower ``i`` defaults
when its standardised asset value falls below the default threshold:

.. math::
    A_i = \\sqrt{\\rho}\\,Z + \\sqrt{1-\\rho}\\,\\varepsilon_i,
    \\qquad A_i < \\Phi^{-1}(PD_i)

where :math:`Z \\sim N(0,1)` is the systematic factor, :math:`\\varepsilon_i`
the idiosyncratic shock and :math:`\\rho` the asset correlation. Conditional on
:math:`Z`, defaults are independent Bernoulli draws with probability

.. math::
    p_i(Z) = \\Phi\\!\\left(\\frac{\\Phi^{-1}(PD_i) - \\sqrt{\\rho}\\,Z}
    {\\sqrt{1-\\rho}}\\right).

Losses are :math:`L = \\sum_i EAD_i \\cdot LGD_i \\cdot D_i`. The portfolio is
simulated at the level of homogeneous buckets (``grade`` x PD decile), which
keeps the simulation exact while remaining fast for hundreds of thousands of
loans.

Reported statistics: Expected Loss (EL), Unexpected Loss (UL), Value-at-Risk at
99% and 99.9%, Expected Shortfall (ES), a convergence study and a stress-test
grid. Two variance-reduction techniques are studied: antithetic variates on the
systematic factor, and importance sampling on that factor for the tail VaR. The
importance-sampling estimator is validated against the closed-form Vasicek VaR
of an infinitely granular homogeneous portfolio (:func:`vasicek_var_closed_form`)
and its spread is measured with a replication-based RMSE study
(:func:`variance_reduction_comparison`) rather than the (invalid for quantiles)
``se / sqrt(ESS)`` scaling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .utils import save_fig


# ---------------------------------------------------------------------------
# Portfolio bucketing
# ---------------------------------------------------------------------------
@dataclass
class BucketedPortfolio:
    """Homogeneous buckets used by the simulation."""

    ead: np.ndarray          # total EAD per bucket
    lgd: np.ndarray          # EAD-weighted average LGD per bucket (0-1)
    pd: np.ndarray           # EAD-weighted average PD per bucket (0-1)
    n: np.ndarray            # number of loans per bucket
    labels: List[str] = field(default_factory=list)

    @property
    def total_ead(self) -> float:
        return float(self.ead.sum())

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "bucket": self.labels,
                "n_loans": self.n.astype(int),
                "pd": self.pd,
                "lgd": self.lgd,
                "ead": self.ead,
                "expected_loss": self.pd * self.lgd * self.ead,
            }
        )


def build_portfolio(
    df: pd.DataFrame,
    pd_hat: np.ndarray,
    lgd_hat: np.ndarray,
    ead_col: str = "funded_amnt",
    n_detile: int = 10,
) -> BucketedPortfolio:
    """Group loans into ``grade`` x PD-decile buckets with EAD weights."""
    d = pd.DataFrame(
        {
            "grade": df["grade"].astype(str).values,
            "pd": np.clip(np.asarray(pd_hat, dtype=float), 1e-6, 1 - 1e-6),
            "lgd": np.clip(np.asarray(lgd_hat, dtype=float), 0.0, 1.0),
            "ead": pd.to_numeric(df[ead_col], errors="coerce").fillna(0.0).to_numpy(),
        }
    )
    d["pd_decile"] = pd.qcut(d["pd"], q=n_detile, duplicates="drop").astype(str)

    grouped = (
        d.groupby(["grade", "pd_decile"], observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "n": float(len(g)),
                    "pd": float(np.average(g["pd"], weights=np.maximum(g["ead"], 1e-9))),
                    "lgd": float(np.average(g["lgd"], weights=np.maximum(g["ead"], 1e-9))),
                    "ead": float(g["ead"].sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    labels = [f"{row.grade}|{row.pd_decile}" for row in grouped.itertuples()]
    return BucketedPortfolio(
        ead=grouped["ead"].to_numpy(dtype=float),
        lgd=grouped["lgd"].to_numpy(dtype=float),
        pd=grouped["pd"].to_numpy(dtype=float),
        n=grouped["n"].to_numpy(dtype=float),
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------
def conditional_pd(pd_draws: np.ndarray, z: np.ndarray, rho: float) -> np.ndarray:
    """Vasicek conditional default probability p_i(Z), broadcastable arrays."""
    a = norm.ppf(np.clip(pd_draws, 1e-12, 1 - 1e-12))
    return norm.cdf((a - np.sqrt(rho) * z) / np.sqrt(1.0 - rho))


def _systematic_draws(
    rng: np.random.Generator, n_simulations: int, antithetic: bool, stratified: bool = False
) -> np.ndarray:
    """Draws of the systematic factor Z.

    Three schemes:

    * ``stratified=True`` - *stratified sampling* of Z. The unit interval is cut
      into ``n`` equal-probability strata, one uniform draw is placed in each,
      and the draws are mapped through ``norm.ppf``. The empirical distribution
      of Z then matches the theoretical N(0,1) quantiles exactly, so the tail of
      the factor is sampled in proportion rather than by luck. This is the
      variance-reduction technique the report measures against plain sampling.
    * ``antithetic=True`` - antithetic variates (kept as a reported *negative*
      result: on a 100k-path portfolio it barely moves the estimate).
    * otherwise plain i.i.d. standard normal draws.
    """
    if stratified:
        # One uniform per stratum, then a random permutation so the strata do not
        # line up with any ordering of the loss computation.
        u = (np.arange(n_simulations) + rng.random(n_simulations)) / n_simulations
        rng.shuffle(u)
        return norm.ppf(u)
    if antithetic:
        half = n_simulations // 2
        z_half = rng.standard_normal(half)
        return np.concatenate([z_half, -z_half])
    return rng.standard_normal(n_simulations)


def _loss_paths(
    portfolio: BucketedPortfolio, rng: np.random.Generator, z: np.ndarray, rho: float
) -> np.ndarray:
    """One loss number per path, vectorised over (paths x buckets)."""
    p = conditional_pd(portfolio.pd, z[:, None], rho)
    defaults = rng.binomial(portfolio.n.astype(int)[None, :], np.clip(p, 0.0, 1.0))
    return ((defaults / np.maximum(portfolio.n, 1e-9)) * portfolio.ead * portfolio.lgd).sum(axis=1)


def simulate_losses(
    portfolio: BucketedPortfolio,
    rho: float = 0.12,
    n_simulations: int = 100_000,
    seed: int = 42,
    antithetic: bool = True,
    stratified: bool = False,
) -> Dict[str, object]:
    """Simulate the portfolio-loss distribution and summarise it."""
    rng = np.random.default_rng(seed)
    z = _systematic_draws(rng, n_simulations, antithetic, stratified=stratified)
    losses = _loss_paths(portfolio, rng, z, rho)
    stats = loss_statistics(losses, portfolio.total_ead)
    stats["n_simulations"] = int(len(z))
    stats["asset_correlation"] = float(rho)
    stats["antithetic"] = bool(antithetic)
    stats["stratified"] = bool(stratified)
    return {
        "losses": losses,
        "z": z,
        "stats": stats,
        "convergence": np.cumsum(losses) / np.arange(1, len(losses) + 1),
        "portfolio": portfolio,
        "rho": rho,
        "n_simulations": len(z),
        "antithetic": antithetic,
        "stratified": stratified,
    }


def loss_statistics(losses: np.ndarray, total_ead: float, levels: Sequence[float] = (0.99, 0.999)) -> Dict[str, float]:
    """EL, UL, VaR and Expected Shortfall, as amounts and as a share of EAD."""
    losses = np.asarray(losses, dtype=float)
    el = float(losses.mean())
    ul = float(losses.std(ddof=1))
    out: Dict[str, float] = {
        "expected_loss": el,
        "expected_loss_pct_ead": el / total_ead if total_ead else float("nan"),
        "unexpected_loss": ul,
        "unexpected_loss_pct_ead": ul / total_ead if total_ead else float("nan"),
        "loss_mean_over_std": el / ul if ul else float("nan"),
        "total_ead": float(total_ead),
    }
    for level in levels:
        var = float(np.quantile(losses, level))
        tail = losses[losses >= var]
        es = float(tail.mean()) if tail.size else var
        key = f"{level:.4f}".rstrip("0").rstrip(".")
        out[f"var_{key}"] = var
        out[f"var_{key}_pct_ead"] = var / total_ead if total_ead else float("nan")
        out[f"es_{key}"] = es
        out[f"es_{key}_pct_ead"] = es / total_ead if total_ead else float("nan")
        out[f"economic_capital_{key}"] = var - el
    return out


# ---------------------------------------------------------------------------
# Convergence and variance reduction
# ---------------------------------------------------------------------------
def convergence_study(
    portfolio: BucketedPortfolio,
    rho: float = 0.12,
    n_max: int = 100_000,
    seed: int = 42,
    antithetic: bool = True,
    stratified: bool = False,
) -> pd.DataFrame:
    """Running mean and standard error of the loss estimate vs number of paths.

    Plain Monte Carlo error decays as ``1/sqrt(N)``; both antithetic variates and
    stratified sampling of the systematic factor reduce the constant for the same
    ``N``. Run both at the same seed and compare ``running_se``.
    """
    rng = np.random.default_rng(seed)
    z = _systematic_draws(rng, n_max, antithetic, stratified=stratified)
    losses = _loss_paths(portfolio, rng, z, rho)

    n = np.arange(1, len(losses) + 1)
    running_mean = np.cumsum(losses) / n
    running_var = np.maximum(np.cumsum(losses**2) / n - running_mean**2, 0.0)
    running_se = np.sqrt(running_var / n)

    checkpoints = [100, 500, 1000, 5000, 10000, 25000, 50000, 100000]
    checkpoints = [c for c in checkpoints if c <= len(losses)]
    idx = np.array(checkpoints) - 1
    return pd.DataFrame(
        {
            "n_paths": checkpoints,
            "running_mean": running_mean[idx],
            "running_se": running_se[np.array(checkpoints) - 1],
        }
    )


def variance_reduction_report(
    portfolio: BucketedPortfolio, rho: float = 0.12, n_max: int = 100_000, seed: int = 42
) -> pd.DataFrame:
    """Compare plain, antithetic and stratified-fold sampling of the factor.

    Both variance-reduction arms use the SAME seed as the plain arm, so the
    ``running_se`` columns are directly comparable. Antithetic variates are kept
    as an honest negative result (they barely move a 100k-path estimate);
    stratified sampling of Z is the technique that does reduce the error.
    """
    plain = convergence_study(portfolio, rho=rho, n_max=n_max, seed=seed, antithetic=False, stratified=False)
    anti = convergence_study(portfolio, rho=rho, n_max=n_max, seed=seed, antithetic=True, stratified=False)
    strat = convergence_study(portfolio, rho=rho, n_max=n_max, seed=seed, antithetic=False, stratified=True)
    out = plain[["n_paths", "running_mean", "running_se"]].rename(
        columns={"running_mean": "mean_plain", "running_se": "se_plain"}
    )
    out["mean_antithetic"] = anti["running_mean"].values
    out["se_antithetic"] = anti["running_se"].values
    out["mean_stratified"] = strat["running_mean"].values
    out["se_stratified"] = strat["running_se"].values
    out["se_reduction_antithetic_pct"] = (1 - out["se_antithetic"] / out["se_plain"]) * 100
    out["se_reduction_stratified_pct"] = (1 - out["se_stratified"] / out["se_plain"]) * 100
    return out


# ---------------------------------------------------------------------------
# Importance sampling on the systematic factor (tail / VaR estimation)
# ---------------------------------------------------------------------------
def _weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Weighted empirical ``level``-quantile (weights need not be normalised)."""
    order = np.argsort(values)
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    total = w.sum()
    if total <= 0:
        return float("nan")
    cum = np.cumsum(w) / total
    idx = int(np.searchsorted(cum, float(level)))
    return float(v[min(idx, len(v) - 1)])


def _batch_quantile_se(
    losses: np.ndarray, z: np.ndarray, shift: float, level: float, n_batches: int = 20
) -> float:
    """Standard error of the IS quantile via the batched-means (independent groups) method.

    The naive formula ``se_plain / sqrt(ESS)`` is valid for weighted *means* but
    **not** for a weighted empirical quantile: the quantile is an order statistic
    driven by the location of the loss distribution, so inflating it by the weight
    ESS overstates the IS error badly. Instead the reweighted sample is split into
    ``n_batches`` disjoint groups, the weighted quantile is recomputed in each, and
    the standard error is the usual ``sd / sqrt(n_batches)`` of those batch
    estimates - a genuine, replication-free measure of the estimator's spread.
    """
    losses = np.asarray(losses, dtype=float)
    n = len(losses)
    if n < n_batches * 2:
        return float("nan")
    shift = float(shift)
    log_w = shift * z - 0.5 * shift**2
    bounds = np.linspace(0, n, n_batches + 1).astype(int)
    batch = np.empty(n_batches)
    for b in range(n_batches):
        sl = slice(bounds[b], bounds[b + 1])
        lw = log_w[sl]
        w = np.exp(lw - lw.max())
        batch[b] = _weighted_quantile(losses[sl], w, level)
    if not np.all(np.isfinite(batch)):
        return float("nan")
    return float(batch.std(ddof=1) / np.sqrt(n_batches))


def importance_sampling_var(
    portfolio: BucketedPortfolio,
    rho: float = 0.12,
    level: float = 0.999,
    n_paths: int = 20_000,
    shift: float = 1.5,
    seed: int = 42,
) -> Dict[str, float]:
    """Estimate the tail VaR with importance sampling on the systematic factor Z.

    Plain Monte Carlo needs very many paths to populate a 99.9% tail: the loss
    quantile sits far out in the ``Z`` distribution, so only a tiny fraction of
    draws explore it. Losses rise when the systematic factor is *low* (a bad
    macro state), so the sampling distribution is shifted towards that region,
    ``Z ~ N(-shift, 1)`` instead of ``N(0, 1)``. Because the loss is monotone
    *decreasing* in ``Z``, the level-quantile of the loss corresponds to the
    ``1 - level`` quantile of ``Z`` (``z = Phi^{-1}(1 - level)``, i.e. about
    ``-3.09`` at 99.9%), which is why the shift is centered near that value.

    Each path is reweighted by the likelihood ratio

    .. math::
        w(Z) = \\frac{\\phi(Z)}{\\phi(Z + \\text{shift})}
             = \\exp\\!\\left(\\text{shift}\\,Z - \\tfrac{1}{2}\\text{shift}^2\\right),

    where ``phi`` is the standard normal density, and the loss quantile is read
    off the *weighted empirical* CDF. The estimator is unbiased by construction;
    the effective sample size :math:`\\mathrm{ESS} = 1/\\sum w_i^2` (weights
    normalised to sum to one) controls how much the shift can buy, and it is
    reported alongside a **batched** standard error that does not rely on the
    (invalid here) ``se/sqrt(ESS)`` scaling. A shift much larger than
    ``|Phi^{-1}(1-level)|`` collapses the ESS and biases the estimate upward.
    """
    rng = np.random.default_rng(seed)
    n_paths = int(n_paths)
    shift = float(shift)
    z = rng.normal(loc=-shift, scale=1.0, size=n_paths)
    losses = _loss_paths(portfolio, rng, z, rho)
    # Likelihood ratio phi(z)/phi(z + shift).
    log_w = shift * z - 0.5 * shift**2
    w = np.exp(log_w - log_w.max())
    w = w / w.sum()

    var_is = _weighted_quantile(losses, w, level)
    # ES under the same (reweighted) measure: weights of the paths at/above VaR.
    tail_mask = losses >= var_is
    if tail_mask.any():
        w_tail = w[tail_mask]
        es_is = float(np.sum(w_tail * losses[tail_mask]) / np.sum(w_tail))
    else:
        es_is = var_is

    ess = float(1.0 / np.sum(w**2)) if np.sum(w**2) > 0 else float("nan")
    se_is = _batch_quantile_se(losses, z, shift, level)

    # Naive estimate on the same number of paths for an apples-to-apples comparison.
    z_naive = rng.standard_normal(n_paths)
    losses_naive = _loss_paths(portfolio, rng, z_naive, rho)
    var_naive = float(np.quantile(losses_naive, float(level)))
    se_naive = _quantile_se(losses_naive, float(level))
    return {
        "level": float(level),
        "shift": shift,
        "shift_target": float(-abs(norm.ppf(1.0 - float(level)))),
        "n_paths": n_paths,
        "var_importance_sampling": var_is,
        "es_importance_sampling": es_is,
        "var_naive_same_paths": var_naive,
        "effective_sample_size": ess,
        "effective_sample_size_share": ess / n_paths if n_paths else float("nan"),
        "es_naive_same_paths": float(np.mean(losses_naive[losses_naive >= var_naive])) if np.any(losses_naive >= var_naive) else var_naive,
        "se_importance_sampling": se_is,
        "se_naive": se_naive,
        "se_ratio_is_over_naive": (se_is / se_naive if se_naive else float("nan")),
        "var_pct_ead_is": var_is / portfolio.total_ead if portfolio.total_ead else float("nan"),
        "var_pct_ead_naive": var_naive / portfolio.total_ead if portfolio.total_ead else float("nan"),
    }


def _quantile_se(losses: np.ndarray, level: float) -> float:
    """Standard error of the naive sample quantile via the batched-means method.

    Splitting the sample into ``n_batches`` disjoint groups, recomputing the
    quantile in each and taking ``sd / sqrt(n_batches)`` of the group estimates
    gives a consistent, distribution-free measure of the estimator's spread that
    can be compared directly with :func:`_batch_quantile_se`.
    """
    losses = np.asarray(losses, dtype=float)
    n = len(losses)
    n_batches = 20
    if n < n_batches * 2:
        return float("nan")
    bounds = np.linspace(0, n, n_batches + 1).astype(int)
    batch = np.array([float(np.quantile(losses[bounds[b] : bounds[b + 1]], level)) for b in range(n_batches)])
    if not np.all(np.isfinite(batch)):
        return float("nan")
    return float(batch.std(ddof=1) / np.sqrt(n_batches))


def importance_sampling_sensitivity(
    portfolio: BucketedPortfolio,
    rhos_shift: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5),
    level: float = 0.999,
    n_paths: int = 20_000,
    rho: float = 0.12,
    seed: int = 42,
) -> pd.DataFrame:
    """How the IS tail estimate, its effective sample size and its real SE move with the shift.

    A larger shift explores the tail more aggressively, pushing more paths into
    the region that matters, but it also concentrates the likelihood-ratio
    weights on fewer paths: the effective sample size falls and the weighted
    quantile eventually drifts *upward*. The shift should therefore be centered
    near the factor quantile that maps to the target level
    (``|Phi^{-1}(1 - level)|``, about ``3.09`` at 99.9%) and tuned by the
    measured standard error, not a single arbitrary constant. The whole curve is
    reported so the trade-off is visible; ``se_ratio_is_over_naive`` below 1
    means importance sampling genuinely beats plain Monte Carlo.
    """
    rows = []
    for s in rhos_shift:
        out = importance_sampling_var(
            portfolio, rho=rho, level=level, n_paths=n_paths, shift=float(s), seed=seed
        )
        rows.append(
            {
                "shift": float(s),
                "shift_target": out["shift_target"],
                "var_is": out["var_importance_sampling"],
                "var_naive": out["var_naive_same_paths"],
                "var_diff_is_minus_naive": out["var_importance_sampling"] - out["var_naive_same_paths"],
                "effective_sample_size": out["effective_sample_size"],
                "ess_share": out["effective_sample_size_share"],
                "se_is": out["se_importance_sampling"],
                "se_naive": out["se_naive"],
                "se_ratio_is_over_naive": out["se_ratio_is_over_naive"],
            }
        )
    return pd.DataFrame(rows)


def variance_reduction_comparison(
    portfolio: BucketedPortfolio,
    rho: float = 0.12,
    level: float = 0.999,
    n_paths: int = 20_000,
    shift: Optional[float] = None,
    seed: int = 42,
    n_replications: int = 40,
) -> Dict[str, float]:
    """Real RMSE comparison of IS vs naive on the tail VaR, by replication.

    Independently of the single-run standard errors, the estimator's true
    spread is measured by repeating both the importance-sampling and the plain
    estimator many times and taking the root-mean-square deviation from their
    common mean. This is the honest, replication-based evidence for how much the
    variance reduction actually buys (used as a unit test and reported in the
    validation report).
    """
    if shift is None:
        shift = abs(float(norm.ppf(1.0 - float(level))))
    reps_is = np.empty(int(n_replications))
    reps_naive = np.empty(int(n_replications))
    for r in range(int(n_replications)):
        out = importance_sampling_var(
            portfolio, rho=rho, level=level, n_paths=n_paths, shift=shift, seed=seed + r
        )
        reps_is[r] = out["var_importance_sampling"]
        reps_naive[r] = out["var_naive_same_paths"]
    target = float(reps_is.mean())
    rmse_is = float(np.sqrt(np.mean((reps_is - target) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((reps_naive - target) ** 2)))
    return {
        "level": float(level),
        "shift": float(shift),
        "n_paths": int(n_paths),
        "n_replications": int(n_replications),
        "mean_var_importance_sampling": float(reps_is.mean()),
        "mean_var_naive": float(reps_naive.mean()),
        "rmse_importance_sampling": rmse_is,
        "rmse_naive": rmse_naive,
        "rmse_reduction_factor": (rmse_naive / rmse_is if rmse_is else float("nan")),
    }


def vasicek_var_closed_form(pd_value: float, rho: float, level: float) -> float:
    """Closed-form ``level``-VaR of an infinitely granular homogeneous portfolio.

    For a homogeneous pool with default probability ``pd``, unit LGD and unit
    EAD, the portfolio loss rate is exactly the conditional PD ``p(Z)`` and the
    ``level``-quantile of the loss is ``p(z_level)`` with
    ``z_level = Phi^{-1}(1 - level)``:

    .. math::
        \\mathrm{VaR}_{\\text{level}} = \\Phi\\!\\left(
        \\frac{\\Phi^{-1}(pd) - \\sqrt{\\rho}\\,\\Phi^{-1}(1 - level)}
        {\\sqrt{1 - \\rho}}\\right).

    Used as the analytic benchmark for the Monte Carlo and importance-sampling
    estimators (unit tests and the validation report).
    """
    z_level = float(norm.ppf(1.0 - float(level)))
    return float(norm.cdf((norm.ppf(float(pd_value)) - np.sqrt(rho) * z_level) / np.sqrt(1.0 - rho)))


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------
def stress_test(
    portfolio: BucketedPortfolio,
    pd_shocks: Sequence[float] = (0.30, 0.50),
    lgd_shocks: Sequence[float] = (0.10,),
    rho: float = 0.12,
    n_simulations: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Re-run the simulation under PD/LGD shocks and tabulate the impact."""
    scenarios: List[Tuple[str, float, float]] = [("baseline", 0.0, 0.0)]
    for s in pd_shocks:
        scenarios.append((f"PD +{int(round(s * 100))}%", s, 0.0))
    for s in lgd_shocks:
        scenarios.append((f"LGD +{int(round(s * 100))}%", 0.0, s))
    if pd_shocks and lgd_shocks:
        scenarios.append(
            (
                f"PD +{int(round(pd_shocks[-1] * 100))}% & LGD +{int(round(lgd_shocks[-1] * 100))}%",
                pd_shocks[-1],
                lgd_shocks[-1],
            )
        )

    rows = []
    for name, pd_shock, lgd_shock in scenarios:
        stressed = BucketedPortfolio(
            ead=portfolio.ead,
            lgd=np.clip(portfolio.lgd * (1 + lgd_shock), 0.0, 1.0),
            pd=np.clip(portfolio.pd * (1 + pd_shock), 1e-9, 1 - 1e-9),
            n=portfolio.n,
            labels=portfolio.labels,
        )
        out = simulate_losses(stressed, rho=rho, n_simulations=n_simulations, seed=seed)
        row = {
            "scenario": name,
            "pd_shock": pd_shock,
            "lgd_shock": lgd_shock,
            "mean_ead_weighted_pd": float(np.average(stressed.pd, weights=stressed.ead)),
            "mean_lgd": float(np.average(stressed.lgd, weights=stressed.ead)),
        }
        row.update(out["stats"])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Asset-correlation calibration and sensitivity
# ---------------------------------------------------------------------------
def basel_retail_correlation(pd_value: float, kind: str = "other") -> float:
    """PD-dependent asset correlation of the Basel IRB retail framework.

    ``other retail``: ``rho = 0.03 + 0.13 * exp(-35 * PD)`` capped to
    ``[0.03, 0.16]``; ``revolving``: ``rho = 0.04 + 0.16 * exp(-35 * PD)``
    capped to ``[0.04, 0.20]`` (Basel II/III, CRE22.5/24). The framework
    therefore *implies* a very low correlation for high-PD portfolios: at the
    observed portfolio PD the implied rho sits near the bottom of the range.
    """
    pd_clipped = float(np.clip(pd_value, 1e-6, 1.0))
    if kind == "revolving":
        rho = 0.04 + 0.16 * np.exp(-35.0 * pd_clipped)
        return float(np.clip(rho, 0.04, 0.20))
    rho = 0.03 + 0.13 * np.exp(-35.0 * pd_clipped)
    return float(np.clip(rho, 0.03, 0.16))


def rho_sensitivity(
    portfolio: BucketedPortfolio,
    rhos: Sequence[float],
    names: Optional[Sequence[str]] = None,
    n_simulations: int = 100_000,
    seed: int = 42,
    antithetic: bool = False,
    stratified: bool = True,
) -> pd.DataFrame:
    """Re-run the simulation over a grid of asset correlations.

    The loss distribution is extremely sensitive to ``rho`` (systematic risk),
    so every capital figure in the report is quoted together with this grid.
    """
    rows = []
    for i, r in enumerate(rhos):
        label = names[i] if names is not None else f"rho={float(r):.2f}"
        out = simulate_losses(
            portfolio,
            rho=float(r),
            n_simulations=n_simulations,
            seed=seed,
            antithetic=antithetic,
            stratified=stratified,
        )
        row = {"label": label, "rho": float(r)}
        row.update(out["stats"])
        rows.append(row)
    return pd.DataFrame(rows)


def rho_sensitivity_plot(
    sens: pd.DataFrame, base_rho: Optional[float] = None, scale: float = 1_000_000.0,
    unit: str = "$m", name: str = "mc_rho_sensitivity.png",
):
    """EL, VaR 99% and VaR 99.9% as a function of the asset correlation."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sens))
    w = 0.27
    ax.bar(x - w, sens["expected_loss"] / scale, width=w, label="Expected loss", color="steelblue")
    ax.bar(x, sens["var_0.99"] / scale, width=w, label="VaR 99%", color="orange")
    ax.bar(x + w, sens["var_0.999"] / scale, width=w, label="VaR 99.9%", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels(sens["label"], rotation=15, ha="right")
    ax.set_ylabel(f"Loss ({unit})")
    ax.set_title("Sensitivity of the loss distribution to the asset correlation")
    ax.legend(fontsize=8)
    if base_rho is not None:
        ax.annotate(f"base case rho={base_rho:.3f}", xy=(0.98, 0.95), xycoords="axes fraction", ha="right", fontsize=8)
    return save_fig(fig, name)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def loss_distribution_plot(
    losses: np.ndarray,
    stats: Dict[str, float],
    scale: float = 1_000_000.0,
    unit: str = "$m",
    name: str = "mc_loss_distribution.png",
):
    """Histogram of simulated losses with EL / VaR / ES markers."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(losses / scale, bins=80, color="steelblue", alpha=0.8, density=True)
    ax.axvline(stats["expected_loss"] / scale, color="navy", lw=2, label=f"EL = {stats['expected_loss']/scale:,.1f}{unit}")
    ax.axvline(stats["var_0.99"] / scale, color="orange", ls="--", lw=2, label=f"VaR 99% = {stats['var_0.99']/scale:,.1f}{unit}")
    ax.axvline(stats["var_0.999"] / scale, color="crimson", ls="--", lw=2, label=f"VaR 99.9% = {stats['var_0.999']/scale:,.1f}{unit}")
    ax.axvline(stats["es_0.99"] / scale, color="black", ls=":", lw=2, label=f"ES 99% = {stats['es_0.99']/scale:,.1f}{unit}")
    ax.set_xlabel(f"Portfolio loss ({unit})")
    ax.set_ylabel("Density")
    ax.set_title("Simulated portfolio loss distribution (single-factor Vasicek)")
    ax.legend(fontsize=8)
    return save_fig(fig, name)


def convergence_plot(studies: Dict[str, pd.DataFrame], name: str = "mc_convergence.png"):
    """Standard error vs number of simulations on a log-log scale."""
    fig, ax = plt.subplots(figsize=(9, 5))
    reference_n = None
    for label, df in studies.items():
        ax.loglog(df["n_paths"], df["running_se"], "o-", lw=1.8, ms=4, label=label)
        reference_n = df["n_paths"].to_numpy()
    if reference_n is not None and len(reference_n) > 1:
        base = reference_n[0]
        ref = studies[list(studies)[0]]["running_se"].iloc[0] * (reference_n / base) ** -0.5
        ax.loglog(reference_n, ref, "k--", lw=1, label=r"$N^{-1/2}$ reference")
    ax.set_xlabel("Number of simulations (N)")
    ax.set_ylabel("Standard error of EL (USD)")
    ax.set_title("Monte Carlo convergence of the expected loss")
    ax.legend(fontsize=8)
    return save_fig(fig, name)


def stress_plot(stress_df: pd.DataFrame, scale: float = 1_000_000.0, unit: str = "$m", name: str = "mc_stress_test.png"):
    """EL and VaR 99.9% across stress scenarios."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(stress_df))
    w = 0.38
    ax.bar(x - w / 2, stress_df["expected_loss"] / scale, width=w, label="Expected loss", color="steelblue")
    ax.bar(x + w / 2, stress_df["var_0.999"] / scale, width=w, label="VaR 99.9%", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels(stress_df["scenario"], rotation=20, ha="right")
    ax.set_ylabel(f"Loss ({unit})")
    ax.set_title("Stress testing: PD and LGD shocks")
    ax.legend()
    return save_fig(fig, name)
