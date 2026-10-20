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
grid. Antithetic variates are used as a variance-reduction technique.
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


def _systematic_draws(rng: np.random.Generator, n_simulations: int, antithetic: bool) -> np.ndarray:
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
) -> Dict[str, object]:
    """Simulate the portfolio-loss distribution and summarise it."""
    rng = np.random.default_rng(seed)
    z = _systematic_draws(rng, n_simulations, antithetic)
    losses = _loss_paths(portfolio, rng, z, rho)
    stats = loss_statistics(losses, portfolio.total_ead)
    stats["n_simulations"] = int(len(z))
    stats["asset_correlation"] = float(rho)
    stats["antithetic"] = bool(antithetic)
    return {
        "losses": losses,
        "z": z,
        "stats": stats,
        "convergence": np.cumsum(losses) / np.arange(1, len(losses) + 1),
        "portfolio": portfolio,
        "rho": rho,
        "n_simulations": len(z),
        "antithetic": antithetic,
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
) -> pd.DataFrame:
    """Running mean and standard error of the loss estimate vs number of paths.

    Plain Monte Carlo error decays as ``1/sqrt(N)``; antithetic variates reduce
    the constant for the same ``N``.
    """
    rng = np.random.default_rng(seed)
    z = _systematic_draws(rng, n_max, antithetic)
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
    """Compare the standard error with and without antithetic variates."""
    plain = convergence_study(portfolio, rho=rho, n_max=n_max, seed=seed, antithetic=False)
    anti = convergence_study(portfolio, rho=rho, n_max=n_max, seed=seed, antithetic=True)
    out = plain[["n_paths", "running_mean", "running_se"]].rename(
        columns={"running_mean": "mean_plain", "running_se": "se_plain"}
    )
    out["mean_antithetic"] = anti["running_mean"].values
    out["se_antithetic"] = anti["running_se"].values
    out["se_reduction_pct"] = (1 - out["se_antithetic"] / out["se_plain"]) * 100
    return out


# ---------------------------------------------------------------------------
# Importance sampling on the systematic factor (tail / VaR estimation)
# ---------------------------------------------------------------------------
def importance_sampling_var(
    portfolio: BucketedPortfolio,
    rho: float = 0.12,
    level: float = 0.999,
    n_paths: int = 20_000,
    shift: float = 2.5,
    seed: int = 42,
) -> Dict[str, float]:
    """Estimate the tail VaR with importance sampling on the systematic factor Z.

    Plain Monte Carlo needs very many paths to populate a 99.9% tail: with a
    Gaussian systematic factor the loss quantile sits far out in the ``Z``
    distribution, so only a tiny fraction of draws explore it. Losses rise when
    the systematic factor is *low* (a bad macro state), so the sampling
    distribution is shifted towards that region,
    ``Z ~ N(-shift, 1)`` instead of ``N(0, 1)``, and every path is reweighted by
    the likelihood ratio

    .. math::
        w(Z) = \\frac{\\phi(Z)}{\\phi(Z + \\text{shift})}
             = \\exp\\!\\left(\\text{shift}\\,Z - \\tfrac{1}{2}\\text{shift}^2\\right),

    where ``phi`` is the standard normal density. The loss quantile is then read
    off the *weighted empirical* CDF.

    Returns the IS estimate together with the naive estimate and the effective
    sample size of the reweighted sample, so the variance reduction is verifiable.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(loc=-float(shift), scale=1.0, size=int(n_paths))
    losses = _loss_paths(portfolio, rng, z, rho)
    # Likelihood ratio phi(z)/phi(z + shift).
    log_w = float(shift) * z - 0.5 * float(shift) ** 2
    w = np.exp(log_w - log_w.max())
    w = w / w.sum()

    order = np.argsort(losses)
    losses_sorted = losses[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    idx = int(np.searchsorted(cum_w, float(level)))
    idx = min(idx, len(losses_sorted) - 1)
    var_is = float(losses_sorted[idx])
    tail_mask = losses >= var_is
    if tail_mask.any():
        w_tail = w[tail_mask]
        es_is = float(np.sum(w_tail * losses[tail_mask]) / np.sum(w_tail))
    else:
        es_is = var_is

    # Effective sample size and a crude standard error of the IS quantile.
    ess = float(1.0 / np.sum(w**2)) if np.sum(w**2) > 0 else float("nan")
    # Naive estimate on the same number of paths for an apples-to-apples comparison.
    z_naive = rng.standard_normal(int(n_paths))
    losses_naive = _loss_paths(portfolio, rng, z_naive, rho)
    var_naive = float(np.quantile(losses_naive, float(level)))
    se_naive = _quantile_se(losses_naive, float(level))
    return {
        "level": float(level),
        "shift": float(shift),
        "n_paths": int(n_paths),
        "var_importance_sampling": var_is,
        "es_importance_sampling": es_is,
        "var_naive_same_paths": var_naive,
        "effective_sample_size": ess,
        "es_naive_same_paths": float(np.mean(losses_naive[losses_naive >= var_naive])) if np.any(losses_naive >= var_naive) else var_naive,
        "se_importance_sampling": se_naive / np.sqrt(max(ess, 1.0)) * np.sqrt(len(losses_naive)),
        "se_naive": se_naive,
        "var_pct_ead_is": var_is / portfolio.total_ead if portfolio.total_ead else float("nan"),
        "var_pct_ead_naive": var_naive / portfolio.total_ead if portfolio.total_ead else float("nan"),
    }


def _quantile_se(losses: np.ndarray, level: float) -> float:
    """Asymptotic standard error of a sample quantile (kernel-free approximation)."""
    losses = np.asarray(losses, dtype=float)
    n = len(losses)
    if n < 10:
        return float("nan")
    q = float(np.quantile(losses, level))
    # Density estimate via a small bandwidth around the quantile.
    h = 1.06 * losses.std(ddof=1) * n ** (-1 / 5)
    if h <= 0:
        return float("nan")
    density = np.mean(np.abs(losses - q) <= h) / (2 * h)
    if density <= 0:
        return float("nan")
    return float(np.sqrt(level * (1 - level) / n) / density)


def importance_sampling_sensitivity(
    portfolio: BucketedPortfolio,
    rhos_shift: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5),
    level: float = 0.999,
    n_paths: int = 20_000,
    rho: float = 0.12,
    seed: int = 42,
) -> pd.DataFrame:
    """How the IS tail estimate and its effective sample size move with the shift.

    A larger shift explores the tail more aggressively **but** concentrates the
    likelihood-ratio weights on fewer paths (the effective sample size falls),
    which can make the estimator *noisier* than plain Monte Carlo. Reporting the
    whole curve - instead of a single tuned shift - makes that trade-off visible.
    """
    rows = []
    for s in rhos_shift:
        out = importance_sampling_var(
            portfolio, rho=rho, level=level, n_paths=n_paths, shift=float(s), seed=seed
        )
        rows.append(
            {
                "shift": float(s),
                "var_is": out["var_importance_sampling"],
                "var_naive": out["var_naive_same_paths"],
                "effective_sample_size": out["effective_sample_size"],
                "ess_share": out["effective_sample_size"] / float(n_paths),
                "se_is": out["se_importance_sampling"],
                "se_naive": out["se_naive"],
                "se_ratio_is_over_naive": (
                    out["se_importance_sampling"] / out["se_naive"] if out["se_naive"] else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


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
    antithetic: bool = True,
) -> pd.DataFrame:
    """Re-run the simulation over a grid of asset correlations.

    The loss distribution is extremely sensitive to ``rho`` (systematic risk),
    so every capital figure in the report is quoted together with this grid.
    """
    rows = []
    for i, r in enumerate(rhos):
        label = names[i] if names is not None else f"rho={float(r):.2f}"
        out = simulate_losses(
            portfolio, rho=float(r), n_simulations=n_simulations, seed=seed, antithetic=antithetic
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
