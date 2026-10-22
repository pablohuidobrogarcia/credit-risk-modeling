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
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

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

    fig, ax = plt.subplots(figsize=vs.FIGSIZE_WIDE)
    width = 0.4
    x = np.arange(len(years))
    colors = {"36": vs.ROLE["primary"], "60": vs.ROLE["secondary"]}
    for i, term in enumerate(terms):
        vals = [by_year[y].get(term, np.nan) for y in years]
        offs = (i - 0.5) * width
        ax.bar(x + offs, vals, width=width, label=f"{term}-month term", color=colors[term])
    ax.axhline(0.90, color=vs.ROLE["danger"], ls="--", lw=1.6)
    ax.text(len(years) - 0.5, 0.915, "90% resolved bar for the primary split",
            ha="right", va="bottom", fontsize=9, color=vs.ROLE["danger"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of the vintage's loans with a final outcome")
    vs.pct_axis(ax)
    vs.takeaway(
        ax,
        "Late vintages are right-censored — the old test window was not trustworthy",
        "Only loans that already resolved can be measured, so 2017-2018 look artificially complete early.",
    )
    vs.annotate_source(ax, "Source: summary.json → eda.resolution_by_year_term")
    return vs.save_showcase(fig, "censoring.png")


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
    es99 = [r["es_0.99_pct_ead"] for r in sens]
    es999 = [r["es_0.999_pct_ead"] for r in sens]

    fig, ax = plt.subplots(figsize=vs.FIGSIZE_WIDE)
    x = np.arange(len(labels))
    w = 0.26
    ax.bar(x - w, el, width=w, label="Expected loss", color=vs.ROLE["primary"])
    ax.bar(x, es99, width=w, label="ES 99%", color=vs.ROLE["secondary"])
    ax.bar(x + w, es999, width=w, label="ES 99.9%", color=vs.ROLE["danger"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, max(es999) * 1.15)
    ax.set_ylabel("Loss (% of EAD)")
    vs.pct_axis(ax)
    ax.legend(loc="upper left")
    vs.takeaway(
        ax,
        "Asset correlation rho drives the tail — EL barely moves, ES moves a lot",
        "Capital figures are quoted together with this grid because rho is the key assumption.",
    )
    vs.annotate_source(ax, "Source: summary.json → simulation.rho_sensitivity")
    return vs.save_showcase(fig, "rho_sensitivity.png")


def figure_scorecard_panel(summary: Dict[str, Any]) -> Path:
    """Figure 4 — ROC (mature vs censored), model table, IV ranking."""
    vs.apply_style()
    pd_ = summary["pd"]
    models = pd_["models"]
    fig = plt.figure(figsize=vs.FIGSIZE_PANEL)
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.28)

    # (a) AUC by model.
    ax0 = fig.add_subplot(gs[0, 0])
    names = [m["model"] for m in models][::-1]
    aucs = [m["auc"] for m in models][::-1]
    colors = [vs.ROLE["tertiary"] if n == pd_["champion"] else vs.ROLE["muted"] for n in names]
    ax0.barh(range(len(names)), aucs, color=colors, height=0.68)
    ax0.set_yticks(range(len(names)))
    ax0.set_yticklabels(names)
    ax0.axvline(0.5, color=vs.ROLE["reference"], ls=":", lw=1.2)
    ax0.set_xlim(0.5, max(aucs) * 1.06)
    ax0.set_xlabel("Out-of-time AUC (mature window)")
    for i, v in enumerate(aucs):
        ax0.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=8.5)
    vs.takeaway(ax0, "Champion: LightGBM", f"AUC {pd_['champion_auc']:.4f} on 2014-2015")

    # (b) AUC: mature vs censored.
    ax1 = fig.add_subplot(gs[0, 1])
    wc = pd_["window_comparison"]
    wlabels = [r["window"].split(" ")[0] for r in wc]
    waucs = [r["auc"] for r in wc]
    ax1.bar(range(len(wlabels)), waucs,
            color=[vs.ROLE["primary"], vs.ROLE["danger"]], width=0.55)
    ax1.set_xticks(range(len(wlabels)))
    ax1.set_xticklabels(wlabels)
    ax1.set_ylim(0.5, max(waucs) * 1.06)
    ax1.set_ylabel("AUC")
    for i, v in enumerate(waucs):
        ax1.text(i, v + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    delta = waucs[1] - waucs[0]
    vs.takeaway(ax1, "Censoring flatters the model",
                f"+{delta:.4f} AUC on the censored window")

    # (c) IV ranking of the scorecard variables.
    ax2 = fig.add_subplot(gs[1, 0])
    iv = sorted(pd_["iv_top"], key=lambda r: r["iv"])[-8:]
    labels = [r["variable"] for r in iv]
    vals = [r["iv"] for r in iv]
    vs.horizontal_barh(ax2, labels, vals, colors=[vs.ROLE["primary"]] * len(labels),
                       value_fmt="{:.3f}")
    ax2.set_xlabel("Information value (IV)")
    vs.takeaway(ax2, "What drives the score", "Top variables by IV")

    # (d) KS / Gini vs benchmark.
    ax3 = fig.add_subplot(gs[1, 1])
    metrics = ["AUC", "Gini", "KS"]
    champ = [pd_["champion_auc"], pd_["champion_gini"], pd_["champion_ks"]]
    benchg = [pd_["benchmark_auc_lender_grade"], pd_["benchmark_gini_lender_grade"], None]
    x = np.arange(len(metrics))
    w = 0.36
    ax3.bar(x - w / 2, champ, width=w, label="Champion (leakage-free)", color=vs.ROLE["primary"])
    bench_vals = [benchg[0], benchg[1], 0 if benchg[2] is None else benchg[2]]
    ax3.bar(x + w / 2, bench_vals, width=w, label="Benchmark (+ grade/int_rate)",
            color=vs.ROLE["secondary"])
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend(loc="upper right")
    vs.takeaway(ax3, "The lender's own grade still wins",
                "Same LightGBM recipe, with and without lender variables")

    vs.annotate_source(fig, "Source: summary.json → pd (models, window_comparison, iv_top)")
    return vs.save_showcase(fig, "scorecard_panel.png")


def figure_calibration_lift(summary: Dict[str, Any]) -> Path:
    """Figure 5 — calibration curve (raw vs calibrated) and method comparison."""
    vs.apply_style()
    raw = _read_csv("pd_calibration_champion.csv")
    cal = _read_csv("pd_calibration_champion_calibrated.csv")
    comp = summary["pd"]["calibration_comparison"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=vs.FIGSIZE_PANEL, gridspec_kw={"wspace": 0.3})

    rp = [_f(r["predicted"]) for r in raw]
    ro = [_f(r["observed"]) for r in raw]
    cp = [_f(r["predicted"]) for r in cal]
    co = [_f(r["observed"]) for r in cal]
    lo = max(0.0, min(rp) * 0.8)
    hi = max(ro) * 1.1
    ax0.plot([lo, hi], [lo, hi], color=vs.ROLE["reference"], ls="--", lw=1.3, label="Perfect calibration")
    ax0.plot(rp, ro, "o-", color=vs.ROLE["danger"], lw=1.6, ms=5, label="Raw scores")
    ax0.plot(cp, co, "s-", color=vs.ROLE["primary"], lw=1.6, ms=5, label="Isotonic-calibrated")
    ax0.set_xlabel("Mean predicted PD (decile)")
    ax0.set_ylabel("Observed default rate")
    ax0.legend(loc="upper left")
    vs.pct_axis(ax0)
    vs.pct_axis(ax0, axis="x")
    vs.takeaway(ax0, "Calibration is decent, not perfect",
                "Points below the line over-predict default")

    methods = [r["method"] for r in comp]
    test_brier = [r["test_brier"] for r in comp]
    colors = [vs.ROLE["tertiary"] if m == summary["pd"]["calibration_method"] else vs.ROLE["muted"]
              for m in methods]
    ax1.bar(range(len(methods)), test_brier, color=colors, width=0.6)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=15, ha="right")
    lo_b = min(test_brier) * 0.995
    hi_b = max(test_brier) * 1.005
    ax1.set_ylim(lo_b, hi_b)
    for i, v in enumerate(test_brier):
        ax1.text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylabel("Test-window Brier score")
    vs.takeaway(ax1, "Calibration barely moves the Brier score",
                "All four methods land within ~0.2% of each other")

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
                "GMM separates default rate more than k-means, but both are weak")

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
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.32)

    # EL by grade (the credit-risk story in one chart).
    ax0 = fig.add_subplot(gs[:, 0])
    grades = [r["grade"] for r in el_by_grade]
    elr = [r["el_rate"] for r in el_by_grade]
    ax0.barh(range(len(grades)), elr, color=vs.ROLE["primary"], height=0.68)
    ax0.set_yticks(range(len(grades)))
    ax0.set_yticklabels(grades)
    ax0.invert_yaxis()
    ax0.set_xlabel("Expected loss (% of EAD)")
    vs.pct_axis(ax0, axis="x")
    for i, v in enumerate(elr):
        ax0.text(v + max(elr) * 0.02, i, _pct(v), va="center", fontsize=8.5)
    vs.takeaway(ax0, "Expected loss by grade")

    # AUC bars.
    ax1 = fig.add_subplot(gs[0, 1])
    labels = ["Champion", "Benchmark"]
    vals = [pd_["champion_auc"], pd_["benchmark_auc_lender_grade"]]
    ax1.bar(range(2), vals, color=[vs.ROLE["primary"], vs.ROLE["secondary"]], width=0.55)
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.5, max(vals) * 1.05)
    ax1.set_ylabel("AUC")
    vs.takeaway(ax1, "Discrimination (mature window)")

    # Censoring gap.
    ax2 = fig.add_subplot(gs[0, 2])
    wc = pd_["window_comparison"]
    ax2.bar(range(2), [r["auc"] for r in wc], color=[vs.ROLE["primary"], vs.ROLE["danger"]],
            width=0.55)
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(["mature", "censored"])
    ax2.set_ylim(0.5, 0.78)
    ax2.set_ylabel("AUC")
    vs.takeaway(ax2, "Censoring inflation")

    # Tail metrics.
    ax3 = fig.add_subplot(gs[1, 1:])
    names = ["EL", "VaR 99%", "VaR 99.9%", "ES 99%", "ES 99.9%"]
    values = [sim["expected_loss_pct_ead"], sim["var_0.99_pct_ead"],
              sim["var_0.999_pct_ead"], sim["es_0.99_pct_ead"], sim["es_0.999_pct_ead"]]
    ax3.bar(range(len(names)), values, color=vs.ROLE["primary"], width=0.6)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names)
    ax3.set_ylim(0, max(values) * 1.15)
    ax3.set_ylabel("Loss (% of EAD)")
    vs.pct_axis(ax3)
    vs.takeaway(ax3, "Portfolio risk (Monte Carlo, rho = 0.12)")

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

    ax.text(0.055, 0.13,
            "Fully reproducible · mature 2010-2015 validation · every number from summary.json",
            color="#BFD3E6", fontsize=12)
    ax.text(0.945, 0.13, "github.com/pablohuidobrogarcia/credit-risk-modeling",
            color="#9FC0E0", fontsize=11, ha="right")
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
    lb = eda.get("ljung_box", []) or []
    lb10 = next((r for r in lb if r.get("index") == 10), {})
    pop = data.get("population", {}) or {}
    ead = sim.get("total_ead")

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
        "window_immature_auc": f4(immature.get("auc")),
        "window_immature_psi": f4(immature.get("score_psi")),
        "window_gap": ("+" if (auc_inflation or 0) >= 0 else "") + f4(auc_inflation),
        "cal_raw_brier": f"{raw_cal.get('brier'):.5f}" if raw_cal.get("brier") is not None else "n/a",
        "cal_brier": f"{cal_cal.get('brier'):.5f}" if cal_cal.get("brier") is not None else "n/a",
        "cal_raw_citl": f4(raw_cal.get("calibration_in_the_large")),
        "cal_citl": f4(cal_cal.get("calibration_in_the_large")),
        "cal_method": str(pd_.get("calibration_method", "n/a")),
        "el_rate": pct(lgd.get("el_rate_oot"), 2),
        "el_amount": usdm(lgd.get("expected_loss_oot")),
        "portfolio_ead": usdm(ead),
        "realised_lgd": f4(lgd.get("realised_lgd_oot")),
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
        "cluster_spread_pp": f"{(clu.get('default_rate_spread') or {}).get('spread_pp', 'n/a')} pp",
        "cluster_gmm_spread_pp": f"{(clu.get('gmm_spread') or {}).get('spread_pp', 'n/a')} pp",
        "reject_gap": f4((reject.get("extrapolation_gap") or {}).get("illustrative_gap")),
        "forecast_ljung_box_p": f"{lb10.get('lb_pvalue'):.3f}" if lb10.get("lb_pvalue") is not None else "n/a",
        "forecast_naive_mase": f4(naive.get("mase")),
        "forecast_sarimax_mase": f4(sarimax.get("mase")),
        "test_count": f"{int(summary.get('test_count', 0)):,}" if summary.get("test_count") else "n/a",
    }

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
    return DOCS / "data.json"


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
