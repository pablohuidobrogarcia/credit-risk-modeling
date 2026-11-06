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

import numpy as np
import pandas as pd
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
        leaf_items = items
        if dim == "vintage" and any(item.get("segment_type") == "month" for item in items):
            leaf_items = [item for item in items if item.get("segment_type") == "month"]
            year_items = [item for item in items if item.get("segment_type") == "year"]
            assert len(year_items) == 2
            assert sum(item["n"] for item in year_items) == meta["n_total"]
            assert sum(item["ead"] for item in year_items) == pytest.approx(meta["ead_total"], rel=1e-9)
        assert sum(item["n"] for item in leaf_items) == meta["n_total"], dim
        assert sum(item["ead"] for item in leaf_items) == pytest.approx(meta["ead_total"], rel=1e-9)
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


def test_vintage_months_reconcile_with_year_aggregates(pe):
    items = pe["dimensions"]["vintage"]
    years = {item["id"]: item for item in items if item.get("segment_type") == "year"}
    months = {item["id"]: item for item in items if item.get("segment_type") == "month"}
    assert set(years) == {"2014", "2015"}
    expected_months = {
        f"{year}-{month:02d}" for year in (2014, 2015) for month in range(1, 13)
    }
    assert set(months) == expected_months
    assert len(months) == 24
    assert sum(item["n"] for item in months.values()) == pe["meta"]["n_total"]
    assert sum(item["ead"] for item in months.values()) == pytest.approx(pe["meta"]["ead_total"], rel=1e-9)
    for month_id, item in months.items():
        assert item["name"] == pd.Timestamp(month_id + "-01").strftime("%b %Y")
        assert item["insufficient"] is False
        assert 0.0 <= item["resolved_share"] <= 1.0
        assert item["immature"] is bool(item["resolved_share"] < 0.90)
    for year, year_item in years.items():
        subset = [item for month_id, item in months.items() if month_id.startswith(year)]
        assert sum(item["n"] for item in subset) == year_item["n"]
        assert sum(item["n"] * item["default_rate"] for item in subset) / year_item["n"] == pytest.approx(year_item["default_rate"], abs=1e-8)
        assert sum(item["n"] * item["pred_pd"] for item in subset) / year_item["n"] == pytest.approx(year_item["pred_pd"], abs=1e-8)
        assert sum(item["ead"] * item["el_pct_ead"] for item in subset) / year_item["ead"] == pytest.approx(year_item["el_pct_ead"], abs=1e-8)


def test_total_reconciles_with_headline_summary(pe, summary):
    total = pe["dimensions"]["total"][0]
    assert total["n"] == summary["pd"]["n_oot"] == pe["meta"]["n_total"]
    assert total["ead"] == pytest.approx(summary["lgd"]["ead_total_oot"], rel=1e-9)
    assert total["default_rate"] == pytest.approx(summary["pd"]["default_rate_oot"], abs=5e-5)
    assert total["el_pct_ead"] == pytest.approx(summary["lgd"]["el_rate_oot"], abs=5e-5)
    assert total["el_amount"] == pytest.approx(summary["lgd"]["expected_loss_oot"], rel=1e-4)


def test_meta_mean_pd_keeps_the_full_precision_from_summary(pe, summary):
    full_pd = summary["pd"]["calibration_metrics"]["calibrated_oot"]["mean_predicted"]
    assert pe["meta"]["mean_pd"] == pytest.approx(full_pd, rel=0, abs=1e-15)
    assert pe["dimensions"]["total"][0]["pred_pd"] == pytest.approx(full_pd, rel=0, abs=1e-15)
    gap_pp = (pe["meta"]["default_rate_total"] - pe["meta"]["mean_pd"]) * 100
    assert round(gap_pp, 2) == 2.29


def test_monthly_drift_matches_vintage_rows(pe, summary):
    rows = sorted(
        (item for item in pe["dimensions"]["vintage"] if item.get("segment_type") == "month"),
        key=lambda item: item["id"],
    )
    assert len(rows) == 24
    gaps = np.array([(row["default_rate"] - row["pred_pd"]) * 100 for row in rows])
    x = np.arange(len(rows), dtype=float)
    slope, intercept = np.polyfit(x, gaps, 1)
    fitted = slope * x + intercept
    r2 = 1 - ((gaps - fitted) ** 2).sum() / ((gaps - gaps.mean()) ** 2).sum()
    drift = summary["pd"]["monthly_drift"]
    assert drift["n_months"] == 24
    assert drift["first_month"] == rows[0]["id"] == "2014-01"
    assert drift["last_month"] == rows[-1]["id"] == "2015-12"
    assert drift["first_gap_pp"] == pytest.approx(gaps[0], abs=1e-12)
    assert drift["last_gap_pp"] == pytest.approx(gaps[-1], abs=1e-12)
    assert drift["min_gap_month"] == rows[int(np.argmin(gaps))]["id"]
    assert drift["max_gap_month"] == rows[int(np.argmax(gaps))]["id"]
    assert drift["slope_pp_per_month"] == pytest.approx(slope, abs=1e-12)
    assert drift["r_squared"] == pytest.approx(r2, abs=1e-12)
    assert drift["first_observed"] == pytest.approx(rows[0]["default_rate"], abs=1e-12)
    assert drift["last_predicted_pd"] == pytest.approx(rows[-1]["pred_pd"], abs=1e-12)


def test_ead_at_default_reconciliation(summary):
    lgd = summary["lgd"]
    assert lgd["ead_total_oot"] < lgd["funded_total_oot"]
    assert lgd["expected_loss_oot"] == pytest.approx(
        lgd["el_rate_oot"] * lgd["ead_total_oot"], rel=5e-4
    )
    assert lgd["expected_loss_oot"] == pytest.approx(
        lgd["el_rate_funded_oot"] * lgd["funded_total_oot"], rel=5e-4
    )
    assert lgd["old_expected_loss_oot"] > lgd["expected_loss_oot"]
    assert lgd["el_to_realised_ratio"] == pytest.approx(
        lgd["expected_loss_oot"] / lgd["realised_loss_oot"], rel=1e-4
    )
    rates = lgd["ead_rate_by_grade"]
    assert {row["grade"] for row in rates} == set("ABCDEFG")
    for row in rates:
        assert 0.0 < row["ead_rate_train"] <= 1.0
        assert 0.0 < row["ead_rate_oot"] <= 1.0
    assert lgd["n_defaults_oot"] + lgd["n_defaults_oot_zero_ead"] == lgd["n_defaults_oot_pd"]


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
