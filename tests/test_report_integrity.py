"""Number-integrity tests: documentation must match the produced artifacts.

These tests run offline (no dataset needed) because both ``summary.json`` and the
Markdown documents are committed. They enforce the repository rule that every
headline figure in ``README.md`` and ``reports/model_validation_report.md`` is
read from ``reports/results/summary.json``:

* :func:`test_readme_results_block_matches_summary` regenerates the marked block
  and fails on any drift (this is what makes CI catch a hand-edited number).
* :func:`test_headline_scalars_reappear_in_docs` spot-checks that the concrete
  headline values actually appear in the documents, so a number cannot be quietly
  dropped from the prose while the generated block stays in sync.
"""
from __future__ import annotations

import re

import pytest

from src import report_tables
from src.config import project_root

README = project_root() / "README.md"
REPORT = project_root() / "reports" / "model_validation_report.md"
SUMMARY = project_root() / "reports" / "results" / "summary.json"


def _summary() -> dict:
    if not SUMMARY.exists():
        pytest.skip("summary.json not present")
    data = report_tables.load_summary(SUMMARY)
    if not data.get("pd"):
        pytest.skip("summary.json has no PD stage (partial run)")
    return data


# ---------------------------------------------------------------------------
# 1. The generated README block must equal the block regenerated from JSON.
# ---------------------------------------------------------------------------
def test_readme_results_block_matches_summary():
    summary = _summary()
    # check_docs raises AssertionError with a diff when the text has drifted.
    report_tables.check_docs(summary, path=README)


def test_readme_markers_present_and_unique():
    text = README.read_text(encoding="utf-8")
    assert text.count(report_tables.START_MARKER) == 1
    assert text.count(report_tables.END_MARKER) == 1


# ---------------------------------------------------------------------------
# 2. Headline scalars must literally appear in the documentation.
# ---------------------------------------------------------------------------
def _fmt_pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f} %"


def _fmt_num(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def test_headline_scalars_appear_in_docs():
    summary = _summary()
    readme = README.read_text(encoding="utf-8")
    report = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
    haystack = readme + "\n" + report

    pd_ = summary["pd"]
    checks = [
        _fmt_num(pd_["champion_auc"]),
        _fmt_num(pd_["champion_gini"]),
        _fmt_num(pd_["champion_ks"]),
        _fmt_num(pd_["benchmark_auc_lender_grade"]),
        _fmt_num(pd_["benchmark_gini_lender_grade"]),
        _fmt_pct(pd_["default_rate_oot"]),
    ]
    lgd = summary.get("lgd", {})
    if lgd:
        checks.append(_fmt_num(lgd["el_rate_oot"]))
        checks.append(_fmt_pct(lgd["el_rate_oot"]))
        best = next(
            (m for m in lgd["model_comparison"] if m["model"] == lgd["best_lgd_model"]),
            lgd["model_comparison"][0],
        )
        checks.append(_fmt_num(best["mae"]))
    sim = summary.get("simulation", {})
    stats = sim.get("statistics", {})
    if stats:
        for key in ("expected_loss_pct_ead", "var_0.99_pct_ead", "var_0.999_pct_ead"):
            if stats.get(key) is not None:
                checks.append(_fmt_pct(stats[key]))

    missing = [c for c in checks if c not in haystack]
    assert not missing, f"Headline values absent from the docs: {missing}"


def test_censoring_gap_is_reported_consistently():
    """The primary/secondary AUC gap quoted in the docs equals summary.json."""
    summary = _summary()
    wc = {row["window"]: row for row in summary["pd"]["window_comparison"]}
    mature = next(v for k, v in wc.items() if "mature" in k)
    immature = next(v for k, v in wc.items() if "immature" in k)
    gap = immature["auc"] - mature["auc"]
    readme = README.read_text(encoding="utf-8")
    # The docs state the gap as "+0.0xxx" with three or four decimals.
    assert f"{gap:+.4f}" in readme or f"{gap:+.3f}" in readme, (
        f"AUC censoring gap {gap:+.4f} not found in README"
    )


def test_summary_has_no_non_finite_numbers():
    """Guards against NaN/inf leaking into a headline artifact."""
    import math

    summary = _summary()
    bad: list[str] = []

    def walk(prefix: str, obj) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}", v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(f"{prefix}[{i}]", v)
        elif isinstance(obj, float) and not math.isfinite(obj):
            bad.append(prefix)

    walk("", summary)
    assert not bad, f"Non-finite values in summary.json: {bad}"


def test_no_stale_repo_slugs_in_docs():
    """The old owner/repo names must not reappear anywhere in the docs."""
    pattern = re.compile(r"Pablo-dot1|credit-risk-lending-club")
    for path in (README, REPORT):
        if path.exists():
            hit = pattern.search(path.read_text(encoding="utf-8"))
            assert hit is None, f"Stale repo slug {hit.group(0)!r} in {path.name}"


# ---------------------------------------------------------------------------
# 3. The new (block 2-8) headline figures must also be present and consistent.
# ---------------------------------------------------------------------------
def test_calibration_method_and_comparison_are_reported():
    summary = _summary()
    pd_ = summary["pd"]
    method = pd_.get("calibration_method")
    if not method:
        pytest.skip("calibration comparison not present (partial run)")
    rows = pd_.get("calibration_comparison") or []
    assert rows, "calibration_comparison is empty"
    assert method in {r["method"] for r in rows}
    readme = README.read_text(encoding="utf-8")
    # The chosen method name is stated in the generated block.
    assert f"`{method}`" in readme


def test_challenger_uses_a_wider_feature_set_than_the_scorecard():
    """Block 2's whole point: the challengers must not inherit the scorecard's
    parsimonious IV selection."""
    summary = _summary()
    pd_ = summary["pd"]
    n_scorecard = pd_.get("n_features_used")
    n_chal = pd_.get("n_features_challenger")
    if n_chal is None:
        pytest.skip("challenger feature count not present")
    assert pd_.get("challenger_uses_full_features") is True
    assert n_chal >= n_scorecard


def test_forecast_benchmark_contains_sarimax_and_naive():
    summary = _summary()
    bench = (summary.get("eda") or {}).get("forecast_benchmark") or []
    if not bench:
        pytest.skip("forecast benchmark not present (partial run)")
    methods = {row["method"] for row in bench}
    assert {"sarimax", "seasonal_naive", "ets"} <= methods
    for row in bench:
        assert row["rmse"] >= 0


def test_monte_carlo_variance_reduction_is_reported():
    summary = _summary()
    sim = summary.get("simulation") or {}
    if not sim:
        pytest.skip("simulation not present")
    assert sim.get("se_reduction_stratified_pct") is not None
    # The antithetic arm is kept as a reported negative result, so its number
    # must exist too (even though it does not help).
    assert "se_reduction_antithetic_pct" in sim


def test_lgd_two_part_boundary_rate_is_reported():
    summary = _summary()
    lgd = summary.get("lgd") or {}
    if not lgd:
        pytest.skip("lgd stage not present")
    if "boundary_rate_oot" not in lgd:
        pytest.skip("two-part LGD not present (partial run)")
    assert 0.0 <= lgd["boundary_rate_train"] <= 1.0
    assert 0.0 <= lgd["boundary_rate_oot"] <= 1.0
    assert "two_part" in {m["model"] for m in lgd["model_comparison"]}
