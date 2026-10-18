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
