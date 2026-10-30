"""Contracts for the portfolio-explorer aggregates and the shipped dashboard.

These tests are deliberately independent of the browser: they prove that the
JSON the dashboard reads is internally consistent, reconciles with the headline
summary, and never leaks a NaN, an empty segment or a score histogram whose
counts do not match its segment.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
DOCS = ROOT / "docs"


def _load(name: str):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not generated")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pe():
    return _load("portfolio_explorer.json")


@pytest.fixture(scope="module")
def summary():
    return _load("summary.json")


def test_dimensions_reconcile_with_window_total(pe):
    meta = pe["meta"]
    required = [
        "total", "state", "grade", "purpose", "vintage", "term", "home_ownership",
    ]
    for dim in required:
        assert dim in pe["dimensions"], dim
        items = pe["dimensions"][dim]
        assert items, f"{dim} has no segments"
        assert sum(item["n"] for item in items) == meta["n_total"], dim
        assert sum(item["ead"] for item in items) == pytest.approx(meta["ead_total"], rel=1e-9)
        for item in items:
            assert item["n"] > 0
            assert item["ead"] >= 0
            assert 0.0 <= item["default_rate"] <= 1.0
            assert 0.0 <= item["ci_low"] <= item["default_rate"] <= item["ci_high"] <= 1.0
            assert 0.0 <= item["pred_pd"] <= 1.0
            assert 0.0 <= item["el_pct_ead"] <= 1.0
            assert item["ead_share"] >= 0.0
            for value in (item["n"], item["default_rate"], item["ci_low"], item["ci_high"],
                          item["pred_pd"], item["mean_lgd"], item["ead"], item["ead_share"],
                          item["el_pct_ead"], item["el_amount"]):
                assert math.isfinite(float(value)), (dim, item["id"], value)


def test_total_reconciles_with_headline_summary(pe, summary):
    total = pe["dimensions"]["total"][0]
    assert total["n"] == summary["pd"]["n_oot"] == pe["meta"]["n_total"]
    assert total["ead"] == pytest.approx(summary["lgd"]["ead_total_oot"], rel=1e-9)
    assert total["default_rate"] == pytest.approx(summary["pd"]["default_rate_oot"], abs=5e-5)
    assert total["el_pct_ead"] == pytest.approx(summary["lgd"]["el_rate_oot"], abs=5e-5)
    assert total["el_amount"] == pytest.approx(summary["lgd"]["expected_loss_oot"], rel=1e-4)


def test_tilemap_covers_every_state_and_flags_small_n(pe):
    states = pe["tilemap"]["states"]
    state_items = pe["dimensions"]["state"]
    assert set(states) == {item["id"] for item in state_items}
    for item in state_items:
        assert states[item["id"]]["insufficient"] == (item["n"] < pe["meta"]["min_n"])


def test_score_histograms_match_segment_counts(pe):
    bins = pe["meta"]["score_bins"]
    assert len(bins) == 21
    assert bins == sorted(bins)
    hist = pe["score_histograms"]
    for dim, items in pe["dimensions"].items():
        for item in items:
            key = f"{dim}:{item['id']}"
            assert key in hist, key
            counts = hist[key]
            assert len(counts) == len(bins) - 1, key
            assert sum(counts) == item["n"], (key, sum(counts), item["n"])
            assert all(isinstance(c, int) and c >= 0 for c in counts), key


def test_shipped_data_json_is_small_and_embeds_the_aggregates(pe):
    path = DOCS / "data.json"
    assert path.exists()
    assert path.stat().st_size < 400 * 1024, f"data.json is {path.stat().st_size / 1024:.0f} KB"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["portfolio_explorer"]["meta"]["n_total"] == pe["meta"]["n_total"]
    assert data["portfolio_explorer"]["dimensions"]["state"] == pe["dimensions"]["state"]
    assert data["interactive"]["cutoff_explorer"]["histograms"]["goods"], "cutoff histogram is empty"
    assert data["interactive"]["cutoff_explorer"]["histograms"]["bads"], "cutoff histogram is empty"
    assert data["interactive"]["applicant_simulator"].get("test_score_hist"), "simulator score histogram missing"
