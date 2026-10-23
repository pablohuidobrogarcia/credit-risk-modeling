"""Static showcase figures + ``docs/data.json`` for the GitHub Pages site.

This module is the only thing the ``figures`` pipeline stage runs. It reads the
artifacts already written by the modeling stages (``reports/results/*.json`` and
``*.csv``) and turns them into:

* seven presentation-grade PNGs under ``reports/figures/showcase`` (also copied
  into ``docs/assets``), and
* ``docs/data.json`` — every number the website displays, taken verbatim from
  ``summary.json`` (plus the Monte-Carlo histogram exports).

Nothing here fits, tunes or recomputes a model. If a value is not already in the
artifacts it is not shown. The website reads ``data.json`` at load time and
contains no hardcoded metric.
"""
from __future__ import annotations

import csv
import html as html_lib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from . import viz_style as vs
from .config import project_root
from .utils import RESULTS_DIR as _RESULTS_DIR

SHOWCASE = vs.SHOWCASE_DIR
DOCS = project_root() / "docs"
DOCS_ASSETS = DOCS / "assets"

# Indirection so tests can point the generator at small fixtures instead of the
# real reports/results directory. Not meant to be changed in production.
RESULTS = _RESULTS_DIR


def _load_json(name: str) -> Dict[str, Any]:
    with open(RESULTS / name, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------
def _read_csv(name: str) -> List[Dict[str, str]]:
    path = RESULTS / name
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _pct(x: float, nd: int = 1) -> str:
    return "n/a" if x != x else f"{x * 100:.{nd}f}%"


def _usd_m(x: float) -> str:
    return "n/a" if x != x else f"${x / 1e6:,.1f}M"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_censoring(summary: Dict[str, Any]) -> Path:
    """Figure 1 — resolved share by vintage (the reason the split exists)."""
    vs.apply_style()
    rows = summary["eda"]["resolution_by_year_term"]
    by_year: Dict[int, Dict[str, float]] = {}
    for r in rows:
        by_year.setdefault(int(r["issue_year"]), {})[str(r["term_months"])] = _f(r["resolved_share"])
    years = sorted(by_year)
    terms = ["36", "60"]

    secondary = (summary.get("pd", {}).get("split_design", {}) or {}).get("secondary", {}) or {}
    incomplete = _f((secondary.get("test_resolution_all_terms") or {}).get("resolved_share"))

    fig, ax = plt.subplots(figsize=vs.FIGSIZE_WIDE)
    width = 0.4
    x = np.arange(len(years))
    colors = {"36": vs.ROLE["primary"], "60": vs.ROLE["secondary"]}
    for i, term in enumerate(terms):
        vals = [by_year[y].get(term, np.nan) for y in years]
        offs = (i - 0.5) * width
        ax.bar(x + offs, vals, width=width, label=f"{term}-month term", color=colors[term])
    ax.axhline(0.90, color=vs.ROLE["series"], ls="--", lw=1.8,
               label="90% resolution threshold (primary split)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of the vintage's loans with a final outcome")
    vs.pct_axis(ax)
    ax.legend(loc="upper right", ncol=1)
    subtitle = (
        f"By the 2018 snapshot only {_pct(incomplete)} of 2017-2018 loans had a final "
        "outcome — that window is incomplete."
        if incomplete == incomplete
        else "Only loans with a final outcome can be measured; recent vintages are incomplete."
    )
    vs.takeaway(
        ax,
        "Late vintages are right-censored — only resolved loans can be measured",
        subtitle,
    )
    vs.annotate_source(ax, "Source: summary.json → eda.resolution_by_year_term; pd.split_design")
    return vs.save_showcase(fig, "censoring.png")


def figure_auc_window(summary: Dict[str, Any]) -> Path:
    """Figure 1b — mature vs censored AUC as a dot plot."""
    vs.apply_style()
    wc = summary["pd"]["window_comparison"]
    labels = [f"{r['window'].split(' ')[0].capitalize()} ({r['test_years']})" for r in wc]
    aucs = [r["auc"] for r in wc]
    colors = [vs.ROLE["primary"], vs.ROLE["danger"]]
    gap = aucs[1] - aucs[0]

    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    y = np.arange(len(aucs))[::-1]
    ax.scatter(aucs, y, s=150, color=colors, zorder=3)
    ax.axvline(0.5, color=vs.ROLE["reference"], ls=":", lw=1.4, label="Chance (AUC = 0.50)")
    for yi, value in zip(y, aucs):
        ax.text(value + 0.006, yi, f"{value:.4f}", va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.5, 0.78)
    ax.set_ylim(-0.7, len(aucs) - 0.3)
    ax.set_xlabel("Out-of-time AUC (axis 0.50–0.78; chance = 0.50)")
    ax.legend(loc="lower right")
    vs.takeaway(
        ax,
        f"Censoring inflates AUC by +{gap:.4f}",
        "Same model recipe, but the censored window is incomplete and has a different "
        "loan composition, so the two AUCs are not comparable.",
        fontsize=12,
    )
    vs.annotate_source(ax, "Source: summary.json → pd.window_comparison")
    return vs.save_showcase(fig, "auc_window.png")


def figure_loss_distribution(summary: Dict[str, Any]) -> Path:
    """Figure 2 — MC loss distribution with EL / VaR / ES markers."""
    vs.apply_style()
    hist = _load_json("sim_loss_histogram.json")
    edges = np.asarray(hist["bin_edges_pct_ead"], dtype=float)
    counts = np.asarray(hist["counts"], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    stats = summary["simulation"]["statistics"]

    fig, ax = plt.subplots(figsize=vs.FIGSIZE_WIDE)
    ax.bar(centers, counts, width=np.diff(edges), align="center",
           color=vs.ROLE["primary"], alpha=0.75)
    markers = [
        (stats["expected_loss_pct_ead"], "EL", vs.ROLE["reference"], "-"),
        (stats["var_0.99_pct_ead"], "VaR 99%", vs.ROLE["secondary"], "--"),
        (stats["var_0.999_pct_ead"], "VaR 99.9%", vs.ROLE["danger"], "--"),
        (stats["es_0.99_pct_ead"], "ES 99%", vs.ROLE["tertiary"], ":"),
    ]
    for value, label, color, ls in markers:
        ax.axvline(value * 100.0, color=color, ls=ls, lw=1.8,
                   label=f"{label} = {value * 100:.1f}% of EAD")
    ax.set_xlabel("Portfolio loss (% of EAD)")
    ax.set_ylabel("Simulated paths (count)")
    ax.legend(loc="upper right")
    vs.takeaway(
        ax,
        "The tail is the story — losses are dominated by systematic risk",
        f"{summary['simulation']['n_simulations']:,} paths, single-factor Vasicek, "
        f"rho = {summary['simulation']['asset_correlation']}.",
    )
    vs.annotate_source(ax, "Source: summary.json → simulation.statistics; sim_loss_histogram.json")
    return vs.save_showcase(fig, "loss_distribution.png")


def figure_rho_sensitivity(summary: Dict[str, Any]) -> Path:
    """Figure 3 — capital metrics as a function of the asset correlation."""
    vs.apply_style()
    sens = summary["simulation"]["rho_sensitivity"]
    labels = [r["label"] for r in sens]
    el = [r["expected_loss_pct_ead"] for r in sens]
    var999 = [r.get("var_0.999_pct_ead") for r in sens]
    es99 = [r["es_0.99_pct_ead"] for r in sens]
    es999 = [r["es_0.999_pct_ead"] for r in sens]

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    x = np.arange(len(labels))
    w = 0.2
    ax.bar(x - 1.5 * w, el, width=w, label="Expected loss", color=vs.ROLE["primary"])
    if all(v is not None and v == v for v in var999):
        ax.bar(x - 0.5 * w, var999, width=w, label="VaR 99.9%", color=vs.ROLE["accent"])
        ax.bar(x + 0.5 * w, es99, width=w, label="ES 99%", color=vs.ROLE["secondary"])
        ax.bar(x + 1.5 * w, es999, width=w, label="ES 99.9%", color=vs.ROLE["danger"])
        ax.set_ylim(0, max([v for v in es999 if v is not None]) * 1.15)
    else:
        ax.bar(x, es99, width=w, label="ES 99%", color=vs.ROLE["secondary"])
        ax.bar(x + w, es999, width=w, label="ES 99.9%", color=vs.ROLE["danger"])
        ax.set_ylim(0, max([v for v in es999 if v is not None]) * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Loss (% of EAD)")
    vs.pct_axis(ax)
    ax.legend(loc="upper left", ncol=2)
    vs.takeaway(
        ax,
        "Asset correlation rho drives the tail — EL barely moves, tail metrics rise",
        "Capital figures are quoted together with this grid because rho is the key assumption.",
        fontsize=11.5,
    )
    fig.subplots_adjust(bottom=0.30)
    vs.annotate_source(fig, "Source: summary.json → simulation.rho_sensitivity")
    return vs.save_showcase(fig, "rho_sensitivity.png")


def figure_scorecard_panel(summary: Dict[str, Any]) -> Path:
    """Figure 4 — AUC dot plot, IV ranking and champion-vs-benchmark metrics."""
    vs.apply_style()
    pd_ = summary["pd"]
    models = pd_["models"]
    fig = plt.figure(figsize=(12.0, 4.8))
    gs = fig.add_gridspec(1, 3, wspace=0.38, left=0.07, right=0.985, top=0.84, bottom=0.17)

    # (a) AUC by model: dot plot, axis range stated, chance marked.
    ax0 = fig.add_subplot(gs[0, 0])
    names = [m["model"] for m in models][::-1]
    aucs = [m["auc"] for m in models][::-1]
    colors = [vs.ROLE["tertiary"] if n == pd_["champion"] else vs.ROLE["muted"] for n in names]
    y = np.arange(len(names))
    ax0.scatter(aucs, y, s=70, color=colors, zorder=3)
    ax0.axvline(0.5, color=vs.ROLE["reference"], ls=":", lw=1.3, label="Chance (0.50)")
    ax0.set_yticks(y)
    ax0.set_yticklabels(names)
    ax0.set_xlim(0.5, max(aucs) + 0.05)
    ax0.set_xlabel("Out-of-time AUC (axis 0.50–{:.2f})".format(max(aucs) + 0.05))
    for yi, v in zip(y, aucs):
        ax0.text(v + 0.006, yi, f"{v:.4f}", va="center", fontsize=8.5)
    ax0.legend(loc="lower right")
    vs.takeaway(ax0, "Champion: LightGBM",
                f"Mature-window AUC {pd_['champion_auc']:.4f}", fontsize=11.5)

    # (b) IV ranking of the scorecard variables: bars from zero.
    ax1 = fig.add_subplot(gs[0, 1])
    iv = sorted(pd_["iv_top"], key=lambda r: r["iv"])[-8:]
    labels = [r["variable"] for r in iv]
    vals = [r["iv"] for r in iv]
    vs.horizontal_barh(ax1, labels, vals, colors=[vs.ROLE["primary"]] * len(labels),
                       value_fmt="{:.3f}")
    ax1.set_xlim(0, max(vals) * 1.18)
    ax1.set_xlabel("Information value (IV), axis from zero")
    vs.takeaway(ax1, "What drives the score", "Top variables by IV", fontsize=11.5)

    # (c) AUC / Gini / KS vs benchmark: bars from zero; KS is not available.
    ax2 = fig.add_subplot(gs[0, 2])
    metrics = ["AUC", "Gini", "KS"]
    champ = [pd_["champion_auc"], pd_["champion_gini"], pd_["champion_ks"]]
    bench = [pd_.get("benchmark_auc_lender_grade"),
             pd_.get("benchmark_gini_lender_grade"), None]
    x = np.arange(len(metrics))
    w = 0.36
    ax2.bar(x - w / 2, champ, width=w, label="Champion (leakage-free)",
            color=vs.ROLE["primary"])
    bench_vals = [b if b is not None else 0.0 for b in bench]
    ax2.bar(x + w / 2, bench_vals, width=w, label="Benchmark (+ grade/int_rate)",
            color=vs.ROLE["secondary"])
    ax2.text(x[2] + w / 2, 0.015, "n/a", ha="center", va="bottom", fontsize=9,
             color=vs.ROLE["muted"], fontweight="bold")
    for xi, v in zip(x, champ):
        ax2.text(xi - w / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x[:2], bench_vals[:2]):
        ax2.text(xi + w / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 0.82)
    ax2.set_ylabel("Metric value (axis from zero)")
    ax2.legend(loc="upper right", fontsize=8)
    vs.takeaway(ax2, "The lender's own grade still wins",
                "Benchmark KS is not reported, hence n/a", fontsize=11.5)

    vs.annotate_source(fig, "Source: summary.json → pd (models, iv_top, chunk benchmark metrics)")
    return vs.save_showcase(fig, "scorecard_panel.png")


def figure_calibration_lift(summary: Dict[str, Any]) -> Path:
    """Figure 5 — calibration curve (raw vs calibrated) and method comparison."""
    vs.apply_style()
    raw = _read_csv("pd_calibration_champion.csv")
    cal = _read_csv("pd_calibration_champion_calibrated.csv")
    comp = summary["pd"]["calibration_comparison"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=vs.FIGSIZE_PANEL,
                                   gridspec_kw={"wspace": 0.32})

    rp = [_f(r["predicted"]) for r in raw]
    ro = [_f(r["observed"]) for r in raw]
    cp = [_f(r["predicted"]) for r in cal]
    co = [_f(r["observed"]) for r in cal]
    hi = max(max(ro), max(co)) * 1.08
    ax0.plot([0.0, hi], [0.0, hi], color=vs.ROLE["reference"], ls="--", lw=1.3,
             label="Perfect calibration")
    ax0.plot(rp, ro, "o-", color=vs.ROLE["danger"], lw=1.6, ms=5, label="Raw scores")
    ax0.plot(cp, co, "s-", color=vs.ROLE["primary"], lw=1.6, ms=5,
             label="Isotonic-calibrated")
    ax0.set_xlim(0.0, hi)
    ax0.set_ylim(0.0, hi)
    ax0.set_xlabel("Mean predicted PD (decile)")
    ax0.set_ylabel("Observed default rate")
    ax0.legend(loc="upper left")
    vs.pct_axis(ax0)
    vs.pct_axis(ax0, axis="x")
    vs.takeaway(ax0, "Every decile sits above the diagonal",
                "Observed default exceeds predicted: the score under-predicts.")

    methods = [r["method"] for r in comp]
    test_brier = [r["test_brier"] for r in comp]
    colors = [vs.ROLE["tertiary"] if m == summary["pd"]["calibration_method"] else vs.ROLE["muted"]
              for m in methods]
    x = np.arange(len(methods))
    lo_b = min(test_brier) - 0.0002
    hi_b = max(test_brier) + 0.0002
    ax1.scatter(x, test_brier, s=80, color=colors, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha="right")
    ax1.set_ylim(lo_b, hi_b)
    ax1.set_ylabel("Test-window Brier score")
    for xi, v in zip(x, test_brier):
        ax1.text(xi, v + 0.00004, f"{v:.5f}", ha="center", va="bottom", fontsize=8.5)
    vs.takeaway(
        ax1,
        "Recalibration does not improve Brier",
        f"Dot plot, axis {lo_b:.4f}–{hi_b:.4f}: differences are in the 4th decimal.",
        fontsize=12,
    )

    vs.annotate_source(fig, "Source: summary.json → pd.calibration_comparison; pd_calibration_champion*.csv")
    return vs.save_showcase(fig, "calibration_lift.png")


def figure_segments(summary: Dict[str, Any]) -> Path:
    """Figure 6 — clustering k-selection and segment default rates."""
    vs.apply_style()
    clu = summary["clustering"]
    ksel = clu["k_selection"]
    kmeans = clu["kmeans_profile"]
    gmm = clu["gmm_profile"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=vs.FIGSIZE_PANEL, gridspec_kw={"wspace": 0.32})

    ks = [r["k"] for r in ksel]
    sil = [r["silhouette"] for r in ksel]
    colors = [vs.ROLE["tertiary"] if k == clu["k"] else vs.ROLE["muted"] for k in ks]
    ax0.bar(range(len(ks)), sil, color=colors, width=0.6)
    ax0.set_xticks(range(len(ks)))
    ax0.set_xticklabels([str(k) for k in ks])
    ax0.set_xlabel("Number of clusters (k)")
    ax0.set_ylabel("Silhouette score")
    vs.takeaway(ax0, f"k = {clu['k']} chosen", "Best silhouette / stability trade-off")

    # Default rate by segment (k-means and GMM), from the profiles.
    seg_labels = ["k-means-" + str(r["segment"]) for r in kmeans] + \
                 ["GMM-" + str(r["segment"]) for r in gmm]
    drs = [r["default_rate"] for r in kmeans] + [r["default_rate"] for r in gmm]
    cols = ([vs.ROLE["primary"]] * len(kmeans)) + ([vs.ROLE["series"]] * len(gmm))
    ax1.bar(range(len(seg_labels)), drs, color=cols, width=0.6)
    ax1.axhline(summary["data"]["default_rate"], color=vs.ROLE["reference"], ls="--", lw=1.2,
                label=f"Portfolio {_pct(summary['data']['default_rate'])}")
    ax1.set_xticks(range(len(seg_labels)))
    ax1.set_xticklabels(seg_labels, rotation=15, ha="right")
    ax1.set_ylabel("Observed default rate")
    vs.pct_axis(ax1)
    ax1.legend(loc="upper right")
    vs.takeaway(ax1, "Segments are descriptive, not predictive",
                "GMM separates default rate more than k-means; both are weak", fontsize=11.5)

    vs.annotate_source(fig, "Source: summary.json → clustering")
    return vs.save_showcase(fig, "segments.png")


def figure_hero(summary: Dict[str, Any]) -> Path:
    """Figure 7 — hero summary card for the top of the page / social card."""
    vs.apply_style()
    pd_ = summary["pd"]
    lgd = summary["lgd"]
    sim = summary["simulation"]["statistics"]
    el_by_grade = lgd["el_by_grade"]

    fig = plt.figure(figsize=vs.FIGSIZE_HERO)
    gs = fig.add_gridspec(2, 3, hspace=0.95, wspace=0.55,
                          left=0.07, right=0.985, top=0.88, bottom=0.12)

    # EL by grade (the credit-risk story in one chart).
    ax0 = fig.add_subplot(gs[:, 0])
    grades = [r["grade"] for r in el_by_grade]
    elr = [r["el_rate"] for r in el_by_grade]
    ax0.barh(range(len(grades)), elr, color=vs.ROLE["primary"], height=0.68)
    ax0.set_yticks(range(len(grades)))
    ax0.set_yticklabels(grades)
    ax0.invert_yaxis()
    ax0.set_xlim(0, max(elr) * 1.30)
    ax0.set_xlabel("Expected loss (% of EAD)")
    vs.pct_axis(ax0, axis="x")
    for i, v in enumerate(elr):
        ax0.text(v + max(elr) * 0.03, i, _pct(v), va="center", fontsize=8.5)
    vs.takeaway(ax0, "EL by grade", fontsize=10.5, pad=8)

    # AUC dot plots with the axis range stated and chance marked.
    def auc_dots(ax, labels, vals, colors, xlim_hi, title):
        y = np.arange(len(vals))[::-1]
        ax.scatter(vals, y, s=80, color=colors, zorder=3)
        ax.axvline(0.5, color=vs.ROLE["reference"], ls=":", lw=1.2)
        for yi, v in zip(y, vals):
            ax.text(v + 0.006, yi, f"{v:.4f}", va="center", fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0.5, xlim_hi)
        ax.set_ylim(-0.7, len(vals) - 0.3)
        ax.set_xlabel("AUC (axis 0.50–{:.2f}; chance 0.50)".format(xlim_hi))
        vs.takeaway(ax, title, fontsize=10.5, pad=8)

    ax1 = fig.add_subplot(gs[0, 1])
    auc_dots(
        ax1,
        ["Champion", "Benchmark"],
        [pd_["champion_auc"], pd_["benchmark_auc_lender_grade"]],
        [vs.ROLE["primary"], vs.ROLE["secondary"]],
        0.75,
        "Discrimination (mature)",
    )

    ax2 = fig.add_subplot(gs[0, 2])
    wc = pd_["window_comparison"]
    auc_dots(
        ax2,
        ["Mature", "Censored"],
        [r["auc"] for r in wc],
        [vs.ROLE["primary"], vs.ROLE["danger"]],
        0.78,
        "Censoring inflation",
    )

    # Tail metrics: bars from zero.
    ax3 = fig.add_subplot(gs[1, 1:])
    names = ["EL", "VaR 99%", "VaR 99.9%", "ES 99%", "ES 99.9%"]
    values = [sim["expected_loss_pct_ead"], sim["var_0.99_pct_ead"],
              sim["var_0.999_pct_ead"], sim["es_0.99_pct_ead"], sim["es_0.999_pct_ead"]]
    ax3.bar(range(len(names)), values, color=vs.ROLE["primary"], width=0.6)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, fontsize=8.5)
    ax3.set_ylim(0, max(values) * 1.18)
    ax3.set_ylabel("Loss (% of EAD)", labelpad=8)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax3.tick_params(axis="y", pad=2)
    vs.takeaway(ax3, "Portfolio risk (Monte Carlo, rho = 0.12)", fontsize=10.5, pad=8)

    vs.annotate_source(fig, "Source: summary.json (pd, lgd.el_by_grade, simulation.statistics)")
    return vs.save_showcase(fig, "hero.png")


# ---------------------------------------------------------------------------
# OG social card
# ---------------------------------------------------------------------------
def og_card(summary: Dict[str, Any]) -> Path:
    """1200x630 Open Graph card built from the hero numbers."""
    vs.apply_style()
    pd_ = summary["pd"]
    sim = summary["simulation"]["statistics"]
    lgd = summary["lgd"]

    fig = plt.figure(figsize=(12.0, 6.3), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#0B2545"))
    ax.add_patch(plt.Rectangle((0, 0), 1, 0.09, color=vs.OKABE_ITO["orange"]))
    ax.text(0.055, 0.86, "Credit Risk Modeling", color="white", fontsize=30, fontweight="bold")
    ax.text(0.055, 0.79, "Lending Club · PD · LGD · EL · Monte Carlo capital",
            color="#BFD3E6", fontsize=15)
    ax.plot([0.055, 0.945], [0.72, 0.72], color="#2E5A88", lw=1.5)

    cards = [
        ("Champion AUC", f"{pd_['champion_auc']:.4f}", "Gini " + f"{pd_['champion_gini']:.4f}"),
        ("Expected loss", f"{lgd['el_rate_oot']*100:.2f}%", "of EAD"),
        ("VaR 99.9%", f"{sim['var_0.999_pct_ead']*100:.1f}%", "of EAD"),
        ("Censoring gap", f"+{(pd_['window_comparison'][1]['auc']-pd_['window_comparison'][0]['auc']):.4f}", "AUC inflation"),
    ]
    x0, dx = 0.055, 0.225
    for i, (label, value, sub) in enumerate(cards):
        x = x0 + i * dx
        ax.text(x, 0.56, label, color="#9FC0E0", fontsize=14)
        ax.text(x, 0.40, value, color="white", fontsize=27, fontweight="bold")
        ax.text(x, 0.33, sub, color="#9FC0E0", fontsize=12)

    ax.text(0.055, 0.19,
            "Fully reproducible · mature "
            f"{pd_.get('train_years', [2010])[0]}-{pd_.get('oot_years', [2015])[-1]} validation · "
            "every number from summary.json",
            color="#BFD3E6", fontsize=12.5)
    ax.text(0.055, 0.125, "github.com/pablohuidobrogarcia/credit-risk-modeling",
            color="#9FC0E0", fontsize=12.5)
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
    out = DOCS_ASSETS / "og-card.png"
    # Fixed 1200x630 output: no tight bbox, so the social card is exactly the
    # size Open Graph expects.
    fig.savefig(out, dpi=100, facecolor="#0B2545", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# data.json
# ---------------------------------------------------------------------------
def build_data_json(summary: Dict[str, Any]) -> Path:
    """Assemble ``docs/data.json`` — the site's only data source."""
    pd_ = summary["pd"]
    lgd = summary["lgd"]
    sim = summary["simulation"]
    clu = summary["clustering"]
    eda = summary["eda"]
    data = summary["data"]
    reject = summary.get("reject", {})

    hist = _load_json("sim_loss_histogram.json")
    rho_hists = _load_json("sim_rho_histograms.json")

    # ------------------------------------------------------------------
    # display: every string the page renders, preformatted here in Python.
    # All keys are flat (no dots) so the site can address them with a simple
    # data-field="display.<key>" placeholder, and the checker can prove that
    # every placeholder resolves. Nothing is formatted in the browser and no
    # metric literal exists in index.html.
    # ------------------------------------------------------------------
    stats = sim.get("statistics", {}) or {}
    cal_m = pd_.get("calibration_metrics", {}) or {}
    raw_cal = cal_m.get("raw_oot", {}) or {}
    cal_cal = cal_m.get("calibrated_oot", {}) or {}
    wc = pd_.get("window_comparison", []) or []
    mature = wc[0] if len(wc) > 0 else {}
    immature = wc[1] if len(wc) > 1 else {}
    bench_auc = pd_.get("benchmark_auc_lender_grade")
    champ_auc = pd_.get("champion_auc")
    auc_gap = (bench_auc - champ_auc) if (bench_auc is not None and champ_auc is not None) else None
    auc_inflation = (
        (immature.get("auc") - mature.get("auc"))
        if (mature.get("auc") is not None and immature.get("auc") is not None)
        else None
    )
    fb = eda.get("forecast_benchmark", []) or []
    naive = next((r for r in fb if r.get("method") == "last_value_naive"), {})
    sarimax = next((r for r in fb if r.get("method") == "sarimax"), {})
    ets = next((r for r in fb if r.get("method") == "ets"), {})
    lb = eda.get("ljung_box", []) or []
    lb10 = next((r for r in lb if r.get("index") == 10), {})
    pop = data.get("population", {}) or {}
    ead = sim.get("total_ead")
    try:
        n_deciles = len(_read_csv("pd_calibration_champion.csv"))
    except OSError:
        n_deciles = 0
    forecast_n_test = naive.get("n_test") or (fb[0].get("n_test") if fb else None)

    def usdm(x):
        return "n/a" if x is None else f"${x / 1e6:,.1f}M"

    def pct(x, nd=1):
        return "n/a" if x is None else f"{x * 100:.{nd}f}%"

    def f4(x):
        return "n/a" if x is None else f"{x:.4f}"

    display: Dict[str, str] = {
        "n_rows": f"{int(data.get('n_rows', 0)):,}" if data.get("n_rows") else "n/a",
        "n_accepted": f"{int(pop.get('n_accepted', 0)):,}" if pop.get("n_accepted") else "n/a",
        "n_resolved": f"{int(pop.get('n_resolved', 0)):,}" if pop.get("n_resolved") else "n/a",
        "default_rate": pct(data.get("default_rate"), 2),
        "data_years": "–".join(str(y) for y in (data.get("years") or [])),
        "train_years": "–".join(str(y) for y in (pd_.get("train_years") or [])),
        "oot_years": "–".join(str(y) for y in (pd_.get("oot_years") or [])),
        "default_rate_train": pct(pd_.get("default_rate_train")),
        "default_rate_oot": pct(pd_.get("default_rate_oot")),
        "champion": str(pd_.get("champion", "n/a")),
        "champion_auc": f4(champ_auc),
        "champion_gini": f4(pd_.get("champion_gini")),
        "champion_ks": f4(pd_.get("champion_ks")),
        "n_train": f"{int(pd_.get('n_train', 0)):,}" if pd_.get("n_train") else "n/a",
        "n_oot": f"{int(pd_.get('n_oot', 0)):,}" if pd_.get("n_oot") else "n/a",
        "benchmark_auc": f4(bench_auc),
        "benchmark_gini": f4(pd_.get("benchmark_gini_lender_grade")),
        "auc_gap_vs_benchmark": f4(auc_gap),
        "score_psi": f4(pd_.get("score_psi_train_vs_oot")),
        "cutoff": f4(pd_.get("cutoff")),
        "window_mature_auc": f4(mature.get("auc")),
        "window_mature_psi": f4(mature.get("score_psi")),
        "window_mature_dr": pct(mature.get("default_rate_test"), 2),
        "window_immature_auc": f4(immature.get("auc")),
        "window_immature_psi": f4(immature.get("score_psi")),
        "window_immature_dr": pct(immature.get("default_rate_test"), 2),
        "window_gap": ("+" if (auc_inflation or 0) >= 0 else "") + f4(auc_inflation),
        "cal_raw_brier": f"{raw_cal.get('brier'):.5f}" if raw_cal.get("brier") is not None else "n/a",
        "cal_brier": f"{cal_cal.get('brier'):.5f}" if cal_cal.get("brier") is not None else "n/a",
        "cal_raw_citl": f4(raw_cal.get("calibration_in_the_large")),
        "cal_citl": f4(cal_cal.get("calibration_in_the_large")),
        "cal_method": str(pd_.get("calibration_method", "n/a")),
        "cal_underprediction_pp": (
            f"{abs(mature.get('calibration_gap') or 0) * 100:.1f}"
            if mature.get("calibration_gap") is not None else "n/a"
        ),
        "cal_n_deciles": str(n_deciles) if n_deciles else "n/a",
        "el_rate": pct(lgd.get("el_rate_oot"), 2),
        "el_amount": usdm(lgd.get("expected_loss_oot")),
        "portfolio_ead": usdm(ead),
        "realized_lgd": f4(lgd.get("realised_lgd_oot")),
        "boundary_train": pct(lgd.get("boundary_rate_train")),
        "boundary_oot": pct(lgd.get("boundary_rate_oot")),
        "sim_n": f"{int(sim.get('n_simulations', 0)):,}" if sim.get("n_simulations") else "n/a",
        "rho": f"{sim.get('asset_correlation'):.2f}" if sim.get("asset_correlation") is not None else "n/a",
        "rho_basel": f4(sim.get("rho_basel_implied")),
        "sim_el": pct(stats.get("expected_loss_pct_ead"), 2),
        "sim_el_usd": usdm(stats.get("expected_loss")),
        "sim_ul": pct(stats.get("unexpected_loss_pct_ead"), 2),
        "sim_ul_usd": usdm(stats.get("unexpected_loss")),
        "sim_var99": pct(stats.get("var_0.99_pct_ead"), 2),
        "sim_var99_usd": usdm(stats.get("var_0.99")),
        "sim_var999": pct(stats.get("var_0.999_pct_ead"), 2),
        "sim_var999_usd": usdm(stats.get("var_0.999")),
        "sim_es99": pct(stats.get("es_0.99_pct_ead"), 2),
        "sim_es99_usd": usdm(stats.get("es_0.99")),
        "sim_es999": pct(stats.get("es_0.999_pct_ead"), 2),
        "sim_es999_usd": usdm(stats.get("es_0.999")),
        "se_stratified": f"{sim.get('se_reduction_stratified_pct', 0):.2f}%",
        "se_antithetic": f"{sim.get('se_reduction_antithetic_pct', 0):.2f}%",
        "cluster_k": str(clu.get("k", "n/a")),
        "cluster_silhouette": f4(clu.get("silhouette")),
        "cluster_stability_ari": f4(clu.get("stability_ari")),
        "cluster_gmm_ari": f4(clu.get("gmm_stability_ari")),
        "cluster_spread_pp": f"{(clu.get('default_rate_spread') or {}).get('spread_pp', 'n/a')} pp",
        "cluster_gmm_spread_pp": f"{(clu.get('gmm_spread') or {}).get('spread_pp', 'n/a')} pp",
        "reject_gap": f4((reject.get("extrapolation_gap") or {}).get("illustrative_gap")),
        "forecast_ljung_box_p": f"{lb10.get('lb_pvalue'):.3f}" if lb10.get("lb_pvalue") is not None else "n/a",
        "forecast_naive_mase": f4(naive.get("mase")),
        "forecast_sarimax_mase": f4(sarimax.get("mase")),
        "forecast_ets_mase": f4(ets.get("mase")),
        "forecast_n_test": str(forecast_n_test) if forecast_n_test is not None else "n/a",
        "forecast_n_months": str(eda.get("n_months")) if eda.get("n_months") is not None else "n/a",
        "test_count": f"{int(summary.get('test_count', 0)):,}" if summary.get("test_count") else "n/a",
    }

    # Reader-facing explanations that quote numbers are generated here, from the
    # same values, so they cannot drift away from summary.json.
    display["censoring_note"] = (
        "Only resolved loans enter the sample, so early defaults and early prepayments "
        "are over-represented relative to loans that are still open. The censored "
        f"2017-2018 window has a higher observed default rate ({display['window_immature_dr']}) "
        f"than the mature 2014-2015 window ({display['window_mature_dr']}), and its loan "
        "composition differs, so its AUC is not comparable. The "
        f"{display['window_gap']} AUC gap is a measurement artifact of the sampling; "
        "the censored window is shown only to quantify it and feeds no headline number."
    )
    display["calibration_caption"] = (
        "Reliability by predicted-PD decile. Every point lies above the diagonal: "
        "observed default exceeds predicted, so the score under-predicts. On the test "
        f"window the calibrated score is {display['cal_underprediction_pp']} pp below "
        "the observed rate."
    )
    display["calibration_note"] = (
        f"Recalibration barely changes the Brier score (raw {display['cal_raw_brier']} vs "
        f"calibrated {display['cal_brier']}); calibration-in-the-large moves from "
        f"{display['cal_raw_citl']} to {display['cal_citl']}. The selected method is "
        f"{display['cal_method']}, and no method beats the raw score. Every decile lies "
        "above the diagonal, so the score under-predicts default: on the test window the "
        f"calibrated score is {display['cal_underprediction_pp']} pp below the observed rate."
    )
    display["segments_note"] = (
        f"Silhouette {display['cluster_silhouette']}, stability ARI "
        f"{display['cluster_stability_ari']}. k-means separates the default rate by only "
        f"{display['cluster_spread_pp']}; a Gaussian mixture separates more "
        f"({display['cluster_gmm_spread_pp']}) and is at least as stable (bootstrap ARI "
        f"{display['cluster_gmm_ari']} vs {display['cluster_stability_ari']}). Neither is "
        "used as a model feature."
    )
    display["forecast_caption"] = (
        f"{display['forecast_n_test']}-month holdout: an honest seasonal model is beaten "
        "by the naive benchmark"
    )
    display["forecast_note"] = (
        f"A last-value naive forecast wins outright (MASE {display['forecast_naive_mase']} "
        f"vs {display['forecast_sarimax_mase']} for SARIMAX and "
        f"{display['forecast_ets_mase']} for ETS) on the "
        f"{display['forecast_n_test']}-month holdout. Residual autocorrelation is only "
        f"marginal (Ljung-Box at lag 10, p = {display['forecast_ljung_box_p']}), consistent "
        "with a series that is close to a random walk. This is reported as a limitation, "
        "not dressed up."
    )

    # Limitations are assembled from the same formatted values, so a reader never
    # sees a raw 17-digit float in a sentence.
    limitations = [
        "Discrimination is moderate (champion Gini "
        f"{display['champion_gini']}); the lender's own grade still adds about "
        f"{display['auc_gap_vs_benchmark']} AUC.",
        "Loan-level LGD is not usable (every model has a negative out-of-time R-squared).",
        "Probability calibration is a null result: isotonic did not improve the test Brier "
        f"({display['cal_raw_brier']} raw vs {display['cal_brier']} calibrated).",
        "The monthly default-rate forecast is beaten by a naive benchmark "
        f"(last-value naive MASE {display['forecast_naive_mase']} vs SARIMAX "
        f"{display['forecast_sarimax_mase']}).",
        "Antithetic variates give a negligible variance reduction (reported, not hidden).",
        "No macroeconomic variables; the data ends in 2018, so only one credit cycle is observed.",
        "Reject inference is exploratory only and feeds no headline number. The data contains "
        "approved loans only, so the selection bias cannot be corrected.",
    ]

    payload: Dict[str, Any] = {
        "generated_from": "reports/results/summary.json",
        "seed": summary.get("seed"),
        "display": display,
        "data": {
            "n_rows": data.get("n_rows"),
            "default_rate": data.get("default_rate"),
            "years": data.get("years"),
            "population": data.get("population"),
        },
        "pd": {
            "champion": pd_.get("champion"),
            "champion_auc": pd_.get("champion_auc"),
            "champion_gini": pd_.get("champion_gini"),
            "champion_ks": pd_.get("champion_ks"),
            "benchmark_auc": pd_.get("benchmark_auc_lender_grade"),
            "benchmark_gini": pd_.get("benchmark_gini_lender_grade"),
            "auc_gap_vs_benchmark": round(
                (pd_.get("benchmark_auc_lender_grade") or 0) - (pd_.get("champion_auc") or 0), 4
            ),
            "n_train": pd_.get("n_train"),
            "n_oot": pd_.get("n_oot"),
            "train_years": pd_.get("train_years"),
            "oot_years": pd_.get("oot_years"),
            "default_rate_train": pd_.get("default_rate_train"),
            "default_rate_oot": pd_.get("default_rate_oot"),
            "models": pd_.get("models"),
            "iv_top": pd_.get("iv_top"),
            "shap_top": pd_.get("shap_top"),
            "selected_variables": pd_.get("selected_variables"),
            "window_comparison": pd_.get("window_comparison"),
            "confusion_oot": pd_.get("confusion_oot"),
            "cutoff": pd_.get("cutoff"),
            "calibration_method": pd_.get("calibration_method"),
            "calibration_comparison": pd_.get("calibration_comparison"),
            "calibration_metrics": pd_.get("calibration_metrics"),
            "score_psi_train_vs_oot": pd_.get("score_psi_train_vs_oot"),
            "split_design": pd_.get("split_design"),
        },
        "calibration_curve": {
            "raw": _read_csv("pd_calibration_champion.csv"),
            "calibrated": _read_csv("pd_calibration_champion_calibrated.csv"),
        },
        "lgd": {
            "best_model": lgd.get("best_lgd_model"),
            "el_rate_oot": lgd.get("el_rate_oot"),
            "expected_loss_oot": lgd.get("expected_loss_oot"),
            "ead_total_oot": lgd.get("ead_total_oot"),
            "realised_lgd_oot": lgd.get("realised_lgd_oot"),
            "median_lgd_oot": lgd.get("median_lgd_oot"),
            "boundary_rate_train": lgd.get("boundary_rate_train"),
            "boundary_rate_oot": lgd.get("boundary_rate_oot"),
            "model_comparison": lgd.get("model_comparison"),
            "el_by_grade": lgd.get("el_by_grade"),
        },
        "clustering": {
            "k": clu.get("k"),
            "silhouette": clu.get("silhouette"),
            "stability_ari": clu.get("stability_ari"),
            "k_selection": clu.get("k_selection"),
            "kmeans_profile": clu.get("kmeans_profile"),
            "gmm_profile": clu.get("gmm_profile"),
            "gmm_stability_ari": clu.get("gmm_stability_ari"),
            "gmm_spread": clu.get("gmm_spread"),
            "default_rate_spread": clu.get("default_rate_spread"),
        },
        "simulation": {
            "n_simulations": sim.get("n_simulations"),
            "asset_correlation": sim.get("asset_correlation"),
            "total_ead": sim.get("total_ead"),
            "statistics": sim.get("statistics"),
            "rho_sensitivity": sim.get("rho_sensitivity"),
            "stress": sim.get("stress"),
            "se_reduction_stratified_pct": sim.get("se_reduction_stratified_pct"),
            "se_reduction_antithetic_pct": sim.get("se_reduction_antithetic_pct"),
            "rho_basel_implied": sim.get("rho_basel_implied"),
        },
        "loss_histogram": hist,
        "rho_histograms": rho_hists,
        "forecast": {
            "benchmark": eda.get("forecast_benchmark"),
            "sarimax": eda.get("sarimax"),
            "stationarity": eda.get("stationarity"),
            "ljung_box": eda.get("ljung_box"),
            "n_months": eda.get("n_months"),
            "series_first": eda.get("series_first"),
            "series_last": eda.get("series_last"),
            "mean_default_rate": eda.get("mean_default_rate"),
        },
        "reject": {
            "label": reject.get("label"),
            "illustrative_gap": (reject.get("extrapolation_gap") or {}).get("illustrative_gap"),
            "common_variable_shift": reject.get("common_variable_shift"),
        },
        "test_count": summary.get("test_count"),
        "limitations": limitations,
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    with open(DOCS / "data.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=1, default=str)
        f.write("\n")
    bake_index_html(payload)
    return DOCS / "data.json"


# ---------------------------------------------------------------------------
# Build-time rendering of index.html
# ---------------------------------------------------------------------------
_DISPLAY_ELEMENT_RE = re.compile(
    r'(<(?P<tag>span|div|td|th|em|strong|p|a|li)\b'
    r'(?P<attrs>[^>]*\bdata-field="display\.(?P<key>[A-Za-z0-9_]+)"[^>]*)>)'
    r"(?P<text>.*?)(</(?P=tag)>)",
    re.S,
)


def _num4(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "—"


def _pct1(x: Any) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _int(x: Any) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except (TypeError, ValueError):
        return "—"


def _table_rows(payload: Dict[str, Any]) -> Dict[str, str]:
    """HTML rows for the four data-driven tables, identical to site.js output."""
    pd_ = payload.get("pd") or {}
    lgd = payload.get("lgd") or {}
    clu = payload.get("clustering") or {}
    forecast = payload.get("forecast") or {}

    models = [m for m in (pd_.get("models") or []) if (m.get("sample") or "oot") == "oot"]
    model_rows = "".join(
        "<tr><td>{name}{star}</td><td>{auc}</td><td>{gini}</td><td>{ks}</td>"
        "<td>{n}</td></tr>".format(
            name=html_lib.escape(str(m.get("model", ""))),
            star=" ★" if m.get("model") == pd_.get("champion") else "",
            auc=_num4(m.get("auc")),
            gini=_num4(m.get("gini")),
            ks=_num4(m.get("ks")),
            n=m.get("n_features") if m.get("n_features") is not None else "—",
        )
        for m in models
    )

    lgd_rows = "".join(
        "<tr><td>{name}{star}</td><td>{mae}</td><td>{rmse}</td><td>{r2}</td></tr>".format(
            name=html_lib.escape(str(m.get("model", ""))),
            star=" ★" if m.get("model") == lgd.get("best_model") else "",
            mae=_num4(m.get("mae")),
            rmse=_num4(m.get("rmse")),
            r2=_num4(m.get("r2")),
        )
        for m in (lgd.get("model_comparison") or [])
    )

    cluster_rows = "".join(
        "<tr><td>k-means {seg}</td><td>{n}</td><td>{share}</td><td>{dr}</td></tr>".format(
            seg=s.get("segment"),
            n=_int(s.get("n")),
            share=_pct1(s.get("share")),
            dr=_pct1(s.get("default_rate")),
        )
        for s in (clu.get("kmeans_profile") or [])
    )

    forecast_rows = "".join(
        "<tr><td>{method}</td><td>{rmse}</td><td>{mae}</td><td>{mase}</td></tr>".format(
            method=html_lib.escape(str(r.get("method", "")).replace("_", " ")),
            rmse=_num4(r.get("rmse")),
            mae=_num4(r.get("mae")),
            mase=_num4(r.get("mase")),
        )
        for r in (forecast.get("benchmark") or [])
    )

    return {
        "model-table": model_rows,
        "lgd-table": lgd_rows,
        "cluster-table": cluster_rows,
        "forecast-table": forecast_rows,
    }


def bake_index_html(payload: Dict[str, Any]) -> Path | None:
    """Write every displayed number into ``docs/index.html`` at build time.

    The page still hydrates from ``data.json`` when JavaScript runs, but the
    shipped HTML is already complete: if JS is blocked the KPIs and tables show
    real values instead of em-dash placeholders. Regenerating is idempotent and
    keeps the HTML and data.json in lock-step.
    """
    index = DOCS / "index.html"
    if not index.exists():
        return None
    text = index.read_text(encoding="utf-8").replace("\r\n", "\n")
    display = payload.get("display") or {}

    def fill(match: re.Match) -> str:
        key = match.group("key")
        value = display.get(key)
        if value is None:
            return match.group(0)
        return match.group(1) + html_lib.escape(str(value)) + match.group(6)

    text = _DISPLAY_ELEMENT_RE.sub(fill, text)

    for table_id, rows_html in _table_rows(payload).items():
        pattern = re.compile(
            r'(<table\b[^>]*\bid="' + re.escape(table_id) + r'"[^>]*>.*?<tbody>)'
            r"(.*?)(</tbody>)",
            re.S,
        )
        def _replace_table(match: re.Match, replacement: str = rows_html) -> str:
            return match.group(1) + replacement + match.group(3)

        text = pattern.sub(_replace_table, text, count=1)

    limitations = "".join(
        f"<li>{html_lib.escape(str(t))}</li>" for t in (payload.get("limitations") or [])
    )
    text = re.sub(
        r'(<ul\b[^>]*\bid="limitations-list"[^>]*>)(.*?)(</ul>)',
        lambda m: m.group(1) + limitations + m.group(3),
        text,
        count=1,
        flags=re.S,
    )

    with open(index, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return index


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def copy_assets_to_docs() -> List[Path]:
    """Copy the showcase PNGs into ``docs/assets`` with web-friendly names."""
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for src in sorted(SHOWCASE.glob("*.png")):
        if src.name == "og-card.png":
            continue
        dst = DOCS_ASSETS / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def run_all() -> Dict[str, Any]:
    """Generate every showcase figure, the OG card and ``docs/data.json``."""
    summary = _load_json("summary.json")
    out: Dict[str, Any] = {}
    out["censoring"] = str(figure_censoring(summary))
    out["auc_window"] = str(figure_auc_window(summary))
    out["loss_distribution"] = str(figure_loss_distribution(summary))
    out["rho_sensitivity"] = str(figure_rho_sensitivity(summary))
    out["scorecard_panel"] = str(figure_scorecard_panel(summary))
    out["calibration_lift"] = str(figure_calibration_lift(summary))
    out["segments"] = str(figure_segments(summary))
    out["hero"] = str(figure_hero(summary))
    out["og_card"] = str(og_card(summary))
    out["data_json"] = str(build_data_json(summary))
    out["assets"] = [str(p) for p in copy_assets_to_docs()]
    return out
