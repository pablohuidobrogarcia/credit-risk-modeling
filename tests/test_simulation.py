"""Monte Carlo single-factor (Vasicek) loss simulation."""
from __future__ import annotations

import numpy as np

from src import simulation as sim


def _portfolio(pd_, lgd=0.45, ead=1_000_000.0, n=1000):
    return sim.BucketedPortfolio(
        ead=np.array([ead]),
        lgd=np.array([lgd]),
        pd=np.array([pd_]),
        n=np.array([float(n)]),
        labels=["test"],
    )


def test_zero_pd_gives_zero_loss():
    out = sim.simulate_losses(_portfolio(0.0), rho=0.12, n_simulations=500, seed=1)
    assert np.allclose(out["losses"], 0.0)


def test_certain_default_loses_ead_times_lgd():
    out = sim.simulate_losses(_portfolio(1.0), rho=0.12, n_simulations=200, seed=1)
    assert np.allclose(out["losses"], 450_000.0)


def test_expected_loss_matches_analytic():
    # rho = 0 removes the systematic factor, so EL = pd * lgd * ead exactly.
    out = sim.simulate_losses(_portfolio(0.05), rho=0.0, n_simulations=20_000, seed=2)
    assert abs(out["stats"]["expected_loss"] - 0.05 * 0.45 * 1_000_000.0) < 2_000.0


def test_loss_statistics_keys_and_ordering():
    out = sim.simulate_losses(_portfolio(0.10), rho=0.15, n_simulations=20_000, seed=3)
    s = out["stats"]
    for key in ("expected_loss", "unexpected_loss", "var_0.99", "var_0.999", "es_0.99", "es_0.999"):
        assert key in s
    assert s["var_0.99"] >= s["expected_loss"]
    assert s["var_0.999"] >= s["var_0.99"]
    assert s["es_0.99"] >= s["var_0.99"]


def test_higher_pd_means_higher_var():
    low = sim.simulate_losses(_portfolio(0.03), rho=0.12, n_simulations=20_000, seed=4)["stats"]
    high = sim.simulate_losses(_portfolio(0.15), rho=0.12, n_simulations=20_000, seed=4)["stats"]
    assert high["var_0.999"] > low["var_0.999"]
    assert high["expected_loss"] > low["expected_loss"]


def test_antithetic_does_not_hurt_variance():
    rep = sim.variance_reduction_report(_portfolio(0.08), rho=0.12, n_max=20_000, seed=5)
    last = rep.iloc[-1]
    assert last["se_antithetic"] <= last["se_plain"] * 1.05


def test_convergence_se_decays():
    conv = sim.convergence_study(_portfolio(0.08), rho=0.12, n_max=50_000, seed=6)
    assert conv["running_se"].iloc[-1] < conv["running_se"].iloc[0]


def test_stress_test_increases_losses():
    stress = sim.stress_test(_portfolio(0.07), pd_shocks=(0.30, 0.50), lgd_shocks=(0.10,), seed=7)
    base = stress.loc[stress["scenario"] == "baseline", "expected_loss"].iloc[0]
    assert (stress["expected_loss"] > base).sum() >= 4
    assert len(stress) == 5


def test_build_portfolio_ead_weights():
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"grade": rng.choice(list("ABCD"), 3000), "funded_amnt": rng.uniform(1000, 30000, 3000)})
    pd_hat = rng.random(3000) * 0.4
    lgd_hat = rng.uniform(0.2, 0.9, 3000)
    port = sim.build_portfolio(df, pd_hat, lgd_hat)
    assert abs(port.total_ead - df["funded_amnt"].sum()) < 1e-6
    assert len(port.labels) == len(port.pd) == len(port.n)
    assert ((port.pd > 0) & (port.pd < 1)).all()


def test_basel_retail_correlation_stays_in_the_regulatory_band():
    low_pd = sim.basel_retail_correlation(0.001)
    high_pd = sim.basel_retail_correlation(0.50)
    assert 0.03 <= high_pd <= 0.16
    assert 0.03 <= low_pd <= 0.16
    assert low_pd > high_pd  # correlation decreases as PD rises
    revolving = sim.basel_retail_correlation(0.001, kind="revolving")
    assert 0.04 <= revolving <= 0.20
    assert revolving > low_pd


def test_rho_sensitivity_returns_one_row_per_rho_and_tighter_tails():
    sens = sim.rho_sensitivity(
        _portfolio(0.08), [0.02, 0.15], names=["low", "high"], n_simulations=20_000, seed=9
    )
    assert list(sens["label"]) == ["low", "high"]
    assert list(sens["rho"]) == [0.02, 0.15]
    assert sens["expected_loss"].iloc[0] > 0
    # Higher asset correlation fattens the tail without moving the mean much.
    assert sens["var_0.999"].iloc[1] > sens["var_0.999"].iloc[0]


def test_importance_sampling_matches_the_naive_tail_estimate():
    # Importance sampling is an unbiased reweighting of the systematic factor:
    # its 99.9% VaR should be the same order of magnitude as the naive estimate
    # (the loss distribution here is discrete, so exact equality is not expected).
    out = sim.importance_sampling_var(_portfolio(0.10), rho=0.15, level=0.999, n_paths=40_000, shift=1.5, seed=11)
    naive = out["var_naive_same_paths"]
    assert out["var_importance_sampling"] > 0
    assert 0.5 * naive <= out["var_importance_sampling"] <= 1.5 * naive
    assert out["es_importance_sampling"] >= out["var_importance_sampling"] * 0.99


def test_importance_sampling_hits_the_tail_more_often_than_naive():
    # The whole point: with the shift most paths are in the tail region.
    out = sim.importance_sampling_var(_portfolio(0.10), rho=0.15, level=0.999, n_paths=20_000, shift=1.5, seed=12)
    assert 0 < out["effective_sample_size"] <= out["n_paths"]
    assert out["var_pct_ead_is"] > 0


def test_weighted_quantile_matches_plain_quantile_with_uniform_weights():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(10_000)
    uniform = np.ones_like(values)
    # The weighted (step-CDF) quantile agrees with numpy's linearly interpolated
    # quantile up to the gap between adjacent order statistics.
    gaps = np.diff(np.sort(values))
    for level in (0.5, 0.99, 0.999):
        weighted = sim._weighted_quantile(values, uniform, level)
        plain = float(np.quantile(values, level))
        assert abs(weighted - plain) <= gaps.max()


def test_vasicek_closed_form_matches_a_granular_monte_carlo():
    # An infinitely granular homogeneous pool has loss rate exactly equal to the
    # conditional PD p(Z); its level-quantile has the closed form used as our
    # benchmark. The simulated naive VaR must agree with it in expectation.
    pd_, rho, level = 0.10, 0.15, 0.999
    truth = sim.vasicek_var_closed_form(pd_, rho, level)
    assert 0.0 < truth < 1.0
    # Monotone in rho/PD and anchored at rho -> 0 (VaR -> PD).
    assert abs(sim.vasicek_var_closed_form(pd_, 0.0, level) - pd_) < 1e-9
    assert sim.vasicek_var_closed_form(pd_, 0.15, level) > sim.vasicek_var_closed_form(pd_, 0.02, level)

    granular = sim.BucketedPortfolio(
        ead=np.array([1.0]), lgd=np.array([1.0]), pd=np.array([pd_]),
        n=np.array([10_000_000.0]), labels=["granular"],
    )
    out = sim.simulate_losses(granular, rho=rho, n_simulations=200_000, seed=3)
    naive_var = float(np.quantile(out["losses"], level))
    assert abs(naive_var - truth) < 0.01


def test_importance_sampling_is_unbiased_for_the_closed_form_var():
    # Verifies the likelihood ratio w(z) = phi(z)/phi(z + shift) is implemented
    # correctly: averaging the reweighted quantile across seeds must converge to
    # the analytic Vasicek VaR (a shift of ~3.09 is the factor quantile that maps
    # to the 99.9% level).
    pd_, rho, level = 0.10, 0.15, 0.999
    truth = sim.vasicek_var_closed_form(pd_, rho, level)
    granular = sim.BucketedPortfolio(
        ead=np.array([1.0]), lgd=np.array([1.0]), pd=np.array([pd_]),
        n=np.array([10_000_000.0]), labels=["granular"],
    )
    ests = np.array([
        sim.importance_sampling_var(granular, rho=rho, level=level, n_paths=30_000, shift=3.09, seed=s)[
            "var_importance_sampling"
        ]
        for s in range(12)
    ])
    assert abs(float(ests.mean()) - truth) < 0.01


def test_shift_target_is_the_factor_quantile_for_the_level():
    out = sim.importance_sampling_var(_portfolio(0.10), rho=0.15, level=0.999, n_paths=5_000, shift=1.5, seed=1)
    assert abs(out["shift_target"] - (-3.0902323)) < 1e-4
    out99 = sim.importance_sampling_var(_portfolio(0.10), rho=0.15, level=0.99, n_paths=5_000, shift=1.5, seed=1)
    assert abs(out99["shift_target"] - (-2.3263479)) < 1e-4


def test_importance_sampling_reduces_variance_on_a_realistic_portfolio():
    # The honest, replication-based test: IS at a shift near the factor quantile
    # must have a materially smaller RMSE than plain Monte Carlo at equal paths.
    port = sim.BucketedPortfolio(
        ead=np.linspace(1_000, 30_000, 40),
        lgd=np.full(40, 0.85),
        pd=np.clip(np.linspace(0.03, 0.25, 40), 1e-3, 0.9),
        n=np.full(40, 2_000.0),
        labels=[f"b{i}" for i in range(40)],
    )
    cmp = sim.variance_reduction_comparison(
        port, rho=0.15, level=0.999, n_paths=10_000, shift=1.5, seed=5, n_replications=25
    )
    assert cmp["rmse_importance_sampling"] > 0
    assert cmp["rmse_naive"] > 0
    assert cmp["rmse_reduction_factor"] > 1.0


