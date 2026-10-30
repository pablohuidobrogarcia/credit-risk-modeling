"""Tests for the friendly scorecard-bin formatter and simulator metadata."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src import bin_labels

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
POINTS = RESULTS / "pd_scorecard_points.csv"
WOE = RESULTS / "pd_woe_oot.csv"


def _points():
    return pd.read_csv(POINTS)


def test_adjacent_merged_intervals_collapse():
    rows = _points()
    annual = rows[rows["variable"] == "annual_inc"]["bin"].tolist()
    dti = rows[rows["variable"] == "dti"]["bin"].tolist()
    assert len(annual) == 8
    assert len(dti) == 8
    assert bin_labels.format_bin("annual_inc", annual[0]) == "Up to $26,000"
    assert bin_labels.format_bin("annual_inc", annual[-1]) == "More than $90,600"
    assert bin_labels.format_bin("dti", dti[0]) == "Up to 7.9%"
    assert bin_labels.format_bin("dti", dti[-1]) == "More than 26.7%"


def test_inquiries_and_categories_are_readable():
    rows = _points()
    inq = rows[rows["variable"] == "inq_last_6mths"]["bin"].tolist()
    assert [bin_labels.format_bin("inq_last_6mths", b) for b in inq] == ["0 or 1", "2", "3 or more"]
    home = rows[rows["variable"] == "home_ownership"]["bin"].tolist()
    labels = [bin_labels.format_bin("home_ownership", b) for b in home]
    assert "Mortgage" in labels and "Not reported" in labels and "Other / rare categories" in labels
    assert not any("__" in label for label in labels)


def test_no_formatted_option_contains_raw_bin_syntax():
    rows = _points()
    for _, row in rows.iterrows():
        label = bin_labels.format_bin(str(row["variable"]), str(row["bin"]))
        for forbidden in ("inf", "e+", "__", "(", "]", "["):
            assert forbidden not in label, (row["variable"], row["bin"], label)
        assert label.strip()


def test_every_label_maps_back_to_the_published_points():
    rows = _points()
    lookup = {(str(r["variable"]), str(r["bin"])): float(r["points"]) for _, r in rows.iterrows()}
    meta = bin_labels.build_var_meta(rows.to_dict(orient="records"))
    for variable, entry in meta["variables"].items():
        labels = [option["label"] for option in entry["options"]]
        assert len(labels) == len(set(labels)), f"duplicate labels for {variable}"
        for option in entry["options"]:
            key = option["key"]
            assert (variable, key) in lookup
            assert option["points"] == pytest.approx(lookup[(variable, key)], abs=1e-3)
            assert math.isfinite(option["points"])


def test_build_var_meta_orders_and_defaults():
    rows = _points()
    meta = bin_labels.build_var_meta(rows.to_dict(orient="records"))
    assert set(meta["variables"]) == {
        "annual_inc", "dti", "inq_last_6mths", "revol_util",
        "installment_to_income", "home_ownership", "purpose",
    }
    for variable, entry in meta["variables"].items():
        assert entry["options"], variable
        assert all(opt["label"] for opt in entry["options"])
        assert entry["default"] in [opt["key"] for opt in entry["options"]]
        assert entry["lower_risk"] in [opt["key"] for opt in entry["options"]]
        assert entry["higher_risk"] in [opt["key"] for opt in entry["options"]]
    assert meta["variables"]["purpose"]["options"][0]["label"] == "Debt consolidation"
    assert meta["variables"]["annual_inc"]["options"][0]["label"] == "Up to $26,000"
    assert set(meta["presets"]) == {"typical", "lower", "higher"}


@pytest.mark.skipif(not WOE.exists(), reason="WoE matrix is a regenerated local artifact")
def test_test_window_shares_sum_to_window():
    rows = _points()
    woe = pd.read_csv(WOE)
    counts = bin_labels.test_counts_from_woe(woe.to_dict(orient="records"), rows.to_dict(orient="records"))
    for variable, mapping in counts.items():
        assert sum(mapping.values()) == len(woe), variable


def test_shipped_var_meta_contract():
    data = json.loads((ROOT / "docs" / "data.json").read_text(encoding="utf-8"))
    app = data["interactive"]["applicant_simulator"]
    assert app["var_meta"]
    assert app["presets"]["typical"]
    for variable, meta in app["var_meta"].items():
        keys = {option["key"] for option in meta["options"]}
        assert app["presets"]["typical"][variable] in keys
        for option in meta["options"]:
            assert option["label"]
            assert math.isfinite(float(option["points"]))
            assert 0.0 <= float(option["share"]) <= 1.0

