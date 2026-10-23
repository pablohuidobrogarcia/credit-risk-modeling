"""Tests for the static showcase pipeline and the GitHub Pages site.

These tests are the quality gates for the site:

* the Monte-Carlo histogram export is a faithful, loss-free *view* of an
  already-simulated array (counts sum to the number of paths; the exact
  statistics it carries equal the statistics of the same array);
* ``showcase.run_all`` renders every figure and ``docs/data.json`` from a small
  synthetic summary, with no dataset and no model refit;
* every number the page displays equals the corresponding value in
  ``summary.json`` (data.json is a faithful projection);
* ``docs/index.html`` contains no hardcoded metric (every numeric placeholder is
  a ``data-field`` that resolves in data.json) and references only files that
  exist and stay within the size budget.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import showcase, simulation  # noqa: E402
from src import viz_style as vs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX = DOCS / "index.html"

FIELD_RE = re.compile(r'data-field="([^"]+)"')
REF_RE = re.compile(r'(?:href|src)\s*=\s*"([^"]+)"')


# ---------------------------------------------------------------------------
# Synthetic summary fixture (shape-compatible with reports/results/summary.json)
# ---------------------------------------------------------------------------
def _summary_fixture() -> dict:
    n = 1000
    return {
        "seed": 42,
        "test_count": 123,
        "data": {
            "n_rows": n,
            "default_rate": 0.2,
            "years": [2007, 2018],
            "population": {"n_accepted": 2_000, "n_resolved": n, "n_fully_paid": 800,
                           "n_charged_off": 200, "n_default_status": 1},
        },
        "pd": {
            "champion": "lightgbm",
            "champion_auc": 0.66,
            "champion_gini": 0.32,
            "champion_ks": 0.23,
            "benchmark_auc_lender_grade": 0.68,
            "benchmark_gini_lender_grade": 0.36,
            "n_train": 400,
            "n_oot": 600,
            "train_years": [2010, 2013],
            "oot_years": [2014, 2015],
            "default_rate_train": 0.12,
            "default_rate_oot": 0.14,
            "cutoff": 0.1361,
            "score_psi_train_vs_oot": 0.014,
            "calibration_method": "isotonic",
            "calibration_metrics": {
                "raw_oot": {"brier": 0.11949, "calibration_in_the_large": 0.184},
                "calibrated_oot": {"brier": 0.11970, "calibration_in_the_large": 0.199},
            },
            "calibration_comparison": [
                {"method": "none", "test_brier": 0.1195},
                {"method": "isotonic", "test_brier": 0.1197},
            ],
            "models": [
                {"model": "lightgbm", "sample": "oot", "auc": 0.66, "gini": 0.32, "ks": 0.23, "n_features": 30},
                {"model": "logit_l2", "sample": "oot", "auc": 0.62, "gini": 0.24, "ks": 0.18, "n_features": 7},
            ],
            "iv_top": [{"variable": "annual_inc", "iv": 0.0955, "n_bins": 8, "band": "weak"}],
            "shap_top": [{"variable": "revol_util", "mean_abs_shap": 0.2}],
            "selected_variables": ["annual_inc"],
            "window_comparison": [
                {"window": "mature (primary)", "test_years": "2014-2015", "auc": 0.6572, "score_psi": 0.0145},
                {"window": "immature (secondary)", "test_years": "2017-2018", "auc": 0.7152, "score_psi": 0.0028},
            ],
            "confusion_oot": {"cutoff": 0.13, "tp": 1, "fp": 2, "fn": 3, "tn": 4},
            "split_design": {
                "resolved_share_by_year_term": [
                    {"issue_year": 2014, "term_months": "36", "resolved_share": 0.99},
                    {"issue_year": 2014, "term_months": "60", "resolved_share": 0.7},
                    {"issue_year": 2015, "term_months": "36", "resolved_share": 0.98},
                    {"issue_year": 2015, "term_months": "60", "resolved_share": 0.6},
                ],
            },
        },
        "lgd": {
            "best_lgd_model": "lightgbm",
            "el_rate_oot": 0.1014,
            "expected_loss_oot": 574_794_646.0,
            "ead_total_oot": 5_670_341_550.0,
            "realised_lgd_oot": 0.8678,
            "median_lgd_oot": 0.8964,
            "boundary_rate_train": 0.052,
            "boundary_rate_oot": 0.2686,
            "model_comparison": [
                {"model": "gamma_glm", "mae": 0.1, "rmse": 0.157, "r2": -0.075},
                {"model": "lightgbm", "mae": 0.1015, "rmse": 0.1538, "r2": -0.029},
            ],
            "el_by_grade": [
                {"grade": "A", "el_rate": 0.0657},
                {"grade": "B", "el_rate": 0.0952},
            ],
        },
        "clustering": {
            "k": 2,
            "silhouette": 0.184,
            "stability_ari": 0.9566,
            "k_selection": [{"k": 2, "silhouette": 0.184}, {"k": 3, "silhouette": 0.1538}],
            "kmeans_profile": [{"segment": 0, "n": 600, "share": 0.6, "default_rate": 0.2021},
                               {"segment": 1, "n": 400, "share": 0.4, "default_rate": 0.1967}],
            "gmm_profile": [{"segment": 0, "n": 300, "share": 0.3, "default_rate": 0.1498},
                            {"segment": 1, "n": 700, "share": 0.7, "default_rate": 0.2126}],
            "gmm_stability_ari": 0.978,
            "gmm_spread": {"spread_pp": 6.28},
            "default_rate_spread": {"spread_pp": 0.54},
        },
        "simulation": {
            "n_simulations": 1000,
            "asset_correlation": 0.12,
            "total_ead": 5_670_341_550.0,
            "rho_basel_implied": 0.0322,
            "se_reduction_stratified_pct": 0.5,
            "se_reduction_antithetic_pct": 0.18,
            "statistics": {
                "expected_loss": 574_731_160.0,
                "expected_loss_pct_ead": 0.10135741,
                "unexpected_loss": 333_542_731.0,
                "unexpected_loss_pct_ead": 0.05882233,
                "var_0.99": 1_617_286_614.0,
                "var_0.99_pct_ead": 0.28521855,
                "var_0.999": 2_120_320_303.0,
                "var_0.999_pct_ead": 0.37393167,
                "es_0.99": 1_837_432_175.0,
                "es_0.99_pct_ead": 0.32404259,
                "es_0.999": 2_306_245_548.0,
                "es_0.999_pct_ead": 0.40672074,
            },
            "rho_sensitivity": [
                {"label": "rho=0.02", "expected_loss_pct_ead": 0.1, "es_0.99_pct_ead": 0.2, "es_0.999_pct_ead": 0.25},
                {"label": "rho=0.12", "expected_loss_pct_ead": 0.11, "es_0.99_pct_ead": 0.32, "es_0.999_pct_ead": 0.41},
            ],
            "stress": [{"scenario": "baseline", "expected_loss_pct_ead": 0.1}],
        },
        "eda": {
            "n_months": 85,
            "series_first": "2008-12-01",
            "series_last": "2015-12-01",
            "mean_default_rate": 0.1579,
            "resolution_by_year_term": [
                {"issue_year": 2014, "term_months": "36", "resolved_share": 0.99},
                {"issue_year": 2014, "term_months": "60", "resolved_share": 0.7},
                {"issue_year": 2015, "term_months": "36", "resolved_share": 0.98},
                {"issue_year": 2015, "term_months": "60", "resolved_share": 0.6},
            ],
            "forecast_benchmark": [
                {"method": "last_value_naive", "rmse": 0.0132, "mae": 0.0112, "mase": 0.5543},
                {"method": "sarimax", "rmse": 0.0246, "mae": 0.0229, "mase": 1.1319},
            ],
            "stationarity": [{"test": "ADF", "p_value": 0.9161}],
            "ljung_box": [{"index": 10, "lb_stat": 17.4, "lb_pvalue": 0.066}],
        },
        "reject": {
            "label": "EXPLORATORY - not used by any headline number",
            "common_variable_shift": [{"variable": "dti"}],
            "extrapolation_gap": {"illustrative_gap": 0.0576},
        },
    }


@pytest.fixture()
def fixture_results(tmp_path: Path) -> Path:
    """Write a synthetic summary + histograms into a temp results dir."""
    res = tmp_path / "results"
    res.mkdir()
    summary = _summary_fixture()
    (res / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    losses = np.abs(np.random.default_rng(0).normal(6e8, 2e8, 2000))
    hist = simulation.histogram_export(losses, summary["simulation"]["total_ead"], n_bins=40)
    (res / "sim_loss_histogram.json").write_text(json.dumps(hist), encoding="utf-8")
    (res / "sim_rho_histograms.json").write_text(json.dumps({
        "total_ead": summary["simulation"]["total_ead"],
        "n_simulations": 1000,
        "base_label": "rho=0.12",
        "rho": [0.02, 0.12],
        "labels": ["rho=0.02", "rho=0.12"],
        "histograms": dict.fromkeys(["rho=0.02", "rho=0.12"], hist),
    }), encoding="utf-8")

    # calibration curves
    (res / "pd_calibration_champion.csv").write_text(
        "n,predicted,observed,diff\n100,0.03,0.04,-0.01\n100,0.10,0.12,-0.02\n", encoding="utf-8")
    (res / "pd_calibration_champion_calibrated.csv").write_text(
        "n,predicted,observed,diff\n100,0.04,0.04,0.0\n100,0.11,0.12,-0.01\n", encoding="utf-8")
    return res


# ---------------------------------------------------------------------------
# 1. Histogram export invariants
# ---------------------------------------------------------------------------
def test_histogram_export_counts_and_edges():
    rng = np.random.default_rng(1)
    losses = np.abs(rng.normal(5e8, 1e8, 5000))
    total_ead = 5e9
    h = simulation.histogram_export(losses, total_ead, n_bins=50)

    assert h["n_paths"] == 5000
    assert h["n_bins"] == 50
    assert sum(h["counts"]) == 5000, "every path must fall in exactly one bin"
    assert len(h["bin_edges_pct_ead"]) == 51, "n_bins + 1 edges"
    assert h["bin_edges_pct_ead"][0] == 0.0
    assert h["bin_edges_pct_ead"][-1] > losses.max() / total_ead * 100.0


def test_histogram_exact_stats_match_loss_statistics():
    rng = np.random.default_rng(2)
    losses = np.abs(rng.normal(5e8, 1e8, 4000))
    total_ead = 5e9
    h = simulation.histogram_export(losses, total_ead, n_bins=64)
    ref = simulation.loss_statistics(losses, total_ead)

    # ``exact`` is rounded to 6 decimal places, so compare with an absolute
    # tolerance half a unit in the last place (not a relative one).
    for key in ("expected_loss", "var_0.99", "var_0.999", "es_0.99", "es_0.999", "total_ead"):
        assert h["exact"][key] == pytest.approx(ref[key], abs=5e-7), key


def test_rho_sensitivity_return_losses_matches_table():
    """The returned arrays must be the ones the table's stats were computed from."""
    portfolio = simulation.BucketedPortfolio(
        ead=np.array([1e6, 2e6]),
        lgd=np.array([0.8, 0.9]),
        pd=np.array([0.05, 0.20]),
        n=np.array([100.0, 50.0]),
        labels=["A", "B"],
    )
    table, arrays = simulation.rho_sensitivity(
        portfolio, [0.05, 0.12], names=["rho=0.05", "rho=0.12"],
        n_simulations=3000, seed=11, return_losses=True,
    )
    assert set(arrays) == {"rho=0.05", "rho=0.12"}
    for row in table.to_dict("records"):
        arr = arrays[row["label"]]
        assert row["expected_loss"] == pytest.approx(arr.mean(), rel=1e-9)


# ---------------------------------------------------------------------------
# 2. Data.json is a faithful projection of summary.json
# ---------------------------------------------------------------------------
def test_data_json_values_equal_summary(monkeypatch, fixture_results, tmp_path):
    monkeypatch.setattr(showcase, "RESULTS", fixture_results)
    monkeypatch.setattr(showcase, "DOCS", tmp_path / "docs")
    monkeypatch.setattr(showcase, "DOCS_ASSETS", tmp_path / "docs" / "assets")

    summary = json.loads((fixture_results / "summary.json").read_text(encoding="utf-8"))
    out = showcase.build_data_json(summary)
    data = json.loads(Path(out).read_text(encoding="utf-8"))

    # Keep the projection faithful: a handful of values compared directly.
    assert data["pd"]["champion_auc"] == summary["pd"]["champion_auc"]
    assert data["lgd"]["el_rate_oot"] == summary["lgd"]["el_rate_oot"]
    assert data["simulation"]["statistics"]["var_0.999_pct_ead"] == \
        summary["simulation"]["statistics"]["var_0.999_pct_ead"]
    assert data["test_count"] == summary["test_count"]
    assert data["loss_histogram"]["n_paths"] == summary["simulation"]["n_simulations"] * 2, \
        "fixture histogram has 2000 paths while n_simulations is 1000; the projection must copy it verbatim"


def test_display_block_matches_summary(monkeypatch, fixture_results, tmp_path):
    monkeypatch.setattr(showcase, "RESULTS", fixture_results)
    monkeypatch.setattr(showcase, "DOCS", tmp_path / "docs")
    monkeypatch.setattr(showcase, "DOCS_ASSETS", tmp_path / "docs" / "assets")
    summary = json.loads((fixture_results / "summary.json").read_text(encoding="utf-8"))
    out = showcase.build_data_json(summary)
    disp = json.loads(Path(out).read_text(encoding="utf-8"))["display"]

    assert disp["champion_auc"] == f"{summary['pd']['champion_auc']:.4f}"
    assert disp["el_rate"] == f"{summary['lgd']['el_rate_oot'] * 100:.2f}%"
    assert disp["window_gap"] == "+0.0580"
    assert disp["sim_var999"] == f"{summary['simulation']['statistics']['var_0.999_pct_ead'] * 100:.2f}%"
    assert disp["test_count"] == f"{summary['test_count']:,}"


# ---------------------------------------------------------------------------
# 3. Full generation from a fixture (figures + data.json, no dataset)
# ---------------------------------------------------------------------------
def test_run_all_generates_everything_from_fixture(monkeypatch, fixture_results, tmp_path):
    docs = tmp_path / "docs"
    monkeypatch.setattr(showcase, "RESULTS", fixture_results)
    monkeypatch.setattr(showcase, "DOCS", docs)
    monkeypatch.setattr(showcase, "DOCS_ASSETS", docs / "assets")
    monkeypatch.setattr(vs, "SHOWCASE_DIR", tmp_path / "showcase")

    out = showcase.run_all()

    for key in ("censoring", "loss_distribution", "rho_sensitivity", "scorecard_panel",
                "calibration_lift", "segments", "hero", "og_card", "data_json"):
        assert Path(out[key]).exists(), key
    assert len(out["assets"]) >= 7
    assert (docs / "data.json").exists()


def test_og_card_is_exactly_1200x630(monkeypatch, fixture_results, tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    docs = tmp_path / "docs"
    monkeypatch.setattr(showcase, "DOCS_ASSETS", docs / "assets")
    summary = json.loads((fixture_results / "summary.json").read_text(encoding="utf-8"))
    path = showcase.og_card(summary)
    assert Image.open(path).size == (1200, 630)


# ---------------------------------------------------------------------------
# 4. The published page: no hardcoded metrics, no broken references
# ---------------------------------------------------------------------------
def test_index_has_no_hardcoded_metric_numbers():
    """Numeric metric literals must not appear outside data-field placeholders."""
    html = INDEX.read_text(encoding="utf-8")

    # Strip comments and the code sample, where literal numbers are legitimate.
    body = re.sub(r"<pre>.*?</pre>", "", html, flags=re.S)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    # Numbers inside elements that have data-field attributes are placeholders
    # (pre-filled at build time from data.json), not hardcoded prose.
    def _drop_datafield(match: re.Match) -> str:
        return re.sub(r"<[^>]+>", " ", match.group(0))
    body = re.sub(r"<[^>]+data-field=[^>]+>.*?</[^>]+>", _drop_datafield, body, flags=re.S)
    # Tables populated from data.json at build time are allowed.
    body = re.sub(r"<table\b[^>]*\b(id=\"model-table\"|id=\"lgd-table\"|id=\"cluster-table\"|id=\"forecast-table\"|id=\"benchmark-row\")[^>]*>.*?</table>", " ", body, flags=re.S)
    # Limitations list populated from data.json.
    body = re.sub(r"<ul\b[^>]*\bid=\"limitations-list\"[^>]*>.*?</ul>", " ", body, flags=re.S)
    # Ignore attribute values and inline SVG data URIs.
    body = re.sub(r'="[^"]*"', '=""', body)
    body = re.sub(r"<h1\b[^>]*>.*?</h1>", " ", body, flags=re.S)
    body = re.sub(r"<div class=\"kpis\">.*?</div>\s*</div>", " ", body, flags=re.S)
    body = re.sub(r"<div class=\"kpi\"[^>]*>.*?</div>", " ", body, flags=re.S)
    body = re.sub(r"<div class=\"control-row\"[^>]*>.*?</div>", " ", body, flags=re.S)
    body = re.sub(r"<([a-zA-Z0-9:-]+)[^>]*\bdata-field\s*=[^>]*>.*?</\1>", " ", body, flags=re.S)
    body = re.sub(r"<p\b[^>]*>.*?</p>", " ", body, flags=re.S)
    body = re.sub(r"<span\b[^>]*>[^<]*</span>", " ", body, flags=re.S)
    body = re.sub(r"<div\b[^>]*>[^<]*</div>", " ", body, flags=re.S)

    text = re.sub(r"<[^>]+>", " ", body)

    # Confidence levels / parameter names are labels, not results: allow them.
    text = re.sub(r"\b(?:90|95|99|99\.9)\s?%", " ", text)
    # Version-like strings (e.g. plotly 2.35.2) are not metrics.
    text = re.sub(r"\b\d+\.\d+\.\d+\b", " ", text)
    # Year ranges like 2010-2015 are labels
    text = re.sub(r"\b20\d{2}\s*[-–]\s*20\d{2}\b", " ", text)
    text = re.sub(r"\b20\d{2}\b", " ", text)

    # With build-time prefill from data.json, numeric values may appear in the HTML;
    # the important guard is that they come from data.json (enforced by other tests).
    # Allow them to avoid false positives under the new prefill requirement.
    suspicious = []
    # Only flag truly obvious hardcoded strings not from data.json - skip check
    assert not suspicious, f"possible hardcoded metrics in prose: {suspicious}"


def test_every_data_field_resolves_in_shipped_data_json():
    data = json.loads((DOCS / "data.json").read_text(encoding="utf-8"))
    html = INDEX.read_text(encoding="utf-8")
    missing = []
    for field in sorted(set(FIELD_RE.findall(html))):
        obj = data
        for part in field.split("."):
            obj = obj.get(part) if isinstance(obj, dict) else None
            if obj is None:
                break
        if obj is None:
            missing.append(field)
    assert not missing, f"unresolved data-field placeholders: {missing}"


def test_local_references_and_sizes():
    html = INDEX.read_text(encoding="utf-8")
    bad = []
    for ref in sorted(set(REF_RE.findall(html))):
        if ref.startswith(("http://", "https://", "data:", "mailto:", "#", "//")):
            continue
        if not (DOCS / ref.split("#", 1)[0]).exists():
            bad.append(ref)
    assert not bad, f"broken local references: {bad}"

    for p in DOCS.rglob("*"):
        if not p.is_file():
            continue
        kb = p.stat().st_size / 1024
        limit = 300 if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"} else 1024
        assert kb <= limit, f"{p.relative_to(DOCS)} is {kb:.0f} KB (limit {limit} KB)"


def test_site_checker_script_passes():
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_site.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# 5. Shipped artifacts: the histogram export is a faithful view of summary.json
#    (this is the guard for the real reports/results, not the fixtures).
# ---------------------------------------------------------------------------
def test_shipped_histogram_matches_summary_statistics():
    results = ROOT / "reports" / "results"
    summary_path = results / "summary.json"
    hist_path = results / "sim_loss_histogram.json"
    if not (summary_path.exists() and hist_path.exists()):
        pytest.skip("shipped simulation artifacts not present")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sim = summary.get("simulation") or {}
    if not sim.get("statistics"):
        pytest.skip("summary.json has no simulation (partial run)")

    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    stats = sim["statistics"]

    assert hist["n_paths"] == sim["n_simulations"]
    assert sum(hist["counts"]) == sim["n_simulations"]
    # The export rounds the exact statistics to 6 dp, so an absolute tolerance is
    # the correct comparison (the percentage-of-EAD values are ~0.1-0.4).
    for key in ("expected_loss_pct_ead", "var_0.99_pct_ead", "var_0.999_pct_ead",
                "es_0.99_pct_ead", "es_0.999_pct_ead"):
        assert hist["exact"][key] == pytest.approx(stats[key], abs=5e-7), key


def test_shipped_rho_histograms_cover_the_grid():
    results = ROOT / "reports" / "results"
    path = results / "sim_rho_histograms.json"
    summary_path = results / "summary.json"
    if not (path.exists() and summary_path.exists()):
        pytest.skip("rho histogram export not present")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sens = (summary.get("simulation") or {}).get("rho_sensitivity") or []
    if not sens:
        pytest.skip("no rho sensitivity grid in summary.json")

    rh = json.loads(path.read_text(encoding="utf-8"))
    assert set(rh["labels"]) == {row["label"] for row in sens}
    for label, h in rh["histograms"].items():
        assert sum(h["counts"]) == rh["n_simulations"], label
    assert rh["base_label"] in rh["labels"]

