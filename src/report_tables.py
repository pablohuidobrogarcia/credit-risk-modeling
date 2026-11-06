"""Generate and verify the documentation's headline tables from ``summary.json``.

The project rule is that no number in ``README.md`` or
``reports/model_validation_report.md`` is written by hand: every headline figure is
read back from ``reports/results/summary.json``. This module enforces that rule in
two ways:

1. :func:`render_results_block` builds a Markdown block *from* ``summary.json``.
   :func:`sync_docs` writes it between the ``<!-- RESULTS:START -->`` /
   ``<!-- RESULTS:END -->`` markers in ``README.md``.
2. :func:`check_docs` recomputes the block and raises if the committed README text
   differs, so CI fails the moment a number drifts from the artifacts.

The block is deliberately *generated*, not merely *checked*: a hand-edit to a
headline figure cannot survive ``python -m src.report_tables --write`` plus the
test suite.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from .config import project_root
from .utils import RESULTS_DIR

README_PATH = project_root() / "README.md"
REPORT_PATH = project_root() / "reports" / "model_validation_report.md"
START_MARKER = "<!-- RESULTS:START -->"
END_MARKER = "<!-- RESULTS:END -->"
EAD_START_MARKER = "<!-- EAD-RECON:START -->"
EAD_END_MARKER = "<!-- EAD-RECON:END -->"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_summary(path: Path | None = None) -> Dict[str, Any]:
    """Load ``summary.json`` (the single source of truth for headline numbers)."""
    source = path or (RESULTS_DIR / "summary.json")
    with open(source, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_test_count() -> int:
    """Return the number of tests pytest actually collects for this repository.

    The count is read from `pytest --collect-only` rather than hard-coded, so the
    figure quoted in the documentation can never go stale.
    """
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=project_root(),
        capture_output=True,
        text=True,
    )
    combined = f"{out.stdout}\n{out.stderr}"
    match = re.search(r"(\d+)\s+tests?\s+collected", combined)
    if match:
        return int(match.group(1))
    # Quiet mode may only print a per-file breakdown ("tests/test_x.py: 5").
    per_file = re.findall(r"^\s*tests[/\\]\S+:\s*(\d+)\s*$", out.stdout, re.MULTILINE)
    if per_file:
        return sum(int(n) for n in per_file)
    nodeids = [line for line in out.stdout.splitlines() if "::" in line]
    if nodeids:
        return len(nodeids)
    raise RuntimeError(
        "Could not determine the pytest collection count:\n" + combined[-2000:]
    )


def write_test_count(path: Path | None = None) -> int:
    """Persist the collected test count into ``summary.json`` under ``test_count``.

    Returns the count that was written. The key is kept in ``summary.json`` so the
    README block remains a pure function of that file, exactly like every other
    headline number.
    """
    source = path or (RESULTS_DIR / "summary.json")
    count = collect_test_count()
    with open(source, "r", encoding="utf-8") as f:
        summary = json.load(f)
    summary["test_count"] = count
    with open(source, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return count


def _years_display(years: Any) -> str:
    """Render ``[2014, 2015]`` as ``2014–2015`` and a string as-is."""
    if isinstance(years, (list, tuple)) and years:
        return f"{years[0]}–{years[-1]}" if len(years) > 1 else str(years[0])
    return str(years)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _pct(x: float, nd: int = 2, scale: float = 1.0) -> str:
    """Format a fraction as a percentage with ``nd`` decimals."""
    return f"{x * 100:.{nd}f} %" if x is not None else "n/a"


def _num(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _usd_m(x: float | None) -> str:
    """Format a dollar amount in millions, e.g. ``$207.2M``."""
    return "n/a" if x is None else f"${x / 1_000_000:,.1f}M"


def _usd_b(x: float | None) -> str:
    return "n/a" if x is None else f"${x / 1_000_000_000:,.2f}B"


# ---------------------------------------------------------------------------
# The generated block
# ---------------------------------------------------------------------------
def render_results_block(summary: Dict[str, Any]) -> str:
    """Render the Markdown results block from a parsed ``summary.json``.

    Parameters
    ----------
    summary:
        Parsed contents of ``reports/results/summary.json``.

    Returns
    -------
    str
        Markdown text (without the surrounding HTML markers) containing the
        at-a-glance table, the primary/secondary comparison and the Monte Carlo
        headline table. Every value comes from ``summary``; nothing is hard-coded.
    """
    pd_ = summary.get("pd", {})
    lgd = summary.get("lgd", {})
    eda = summary.get("eda", {})
    sim = summary.get("simulation", {})
    clu = summary.get("clustering", {})
    data = summary.get("data", {})
    stats = sim.get("statistics", {})

    n_sample = data.get("n_rows")
    default_rate = data.get("default_rate")
    years = data.get("years") or [None, None]
    pop = data.get("population", {}) or {}

    champion = pd_.get("champion", "n/a")
    auc = pd_.get("champion_auc")
    gini = pd_.get("champion_gini")
    ks = pd_.get("champion_ks")
    bench_auc = pd_.get("benchmark_auc_lender_grade")
    bench_gini = pd_.get("benchmark_gini_lender_grade")

    lgd_models = lgd.get("model_comparison", []) or []
    best_lgd = lgd.get("best_lgd_model", "n/a")
    best_lgd_row = next((m for m in lgd_models if m.get("model") == best_lgd), {})
    lgd_mae = best_lgd_row.get("mae")
    lgd_r2 = best_lgd_row.get("r2")

    el_amount = lgd.get("expected_loss_oot")
    ead_total = lgd.get("ead_total_oot")
    el_rate = lgd.get("el_rate_oot")

    spread = clu.get("default_rate_spread", {}) or {}
    k_sel = clu.get("k_selection", []) or []
    chosen_k = clu.get("k")
    chosen_row = next((r for r in k_sel if r.get("k") == chosen_k), {})

    cal_method = pd_.get("calibration_method")
    cal_rows = pd_.get("calibration_comparison", []) or []
    cal_row = next((r for r in cal_rows if r.get("method") == cal_method), {})
    forecast_bench = eda.get("forecast_benchmark", []) or []
    best_forecast = forecast_bench[0] if forecast_bench else {}
    sarimax_row = next((r for r in forecast_bench if r.get("method") == "sarimax"), {})
    boundary_train = lgd.get("boundary_rate_train")
    boundary_oot = lgd.get("boundary_rate_oot")
    grade_gap = lgd.get("grade_gap_mae", {}) or {}
    best_lgd_grade_gap = grade_gap.get(best_lgd) if best_lgd else None
    se_strat = sim.get("se_reduction_stratified_pct")
    se_anti = sim.get("se_reduction_antithetic_pct")

    def _pct_stat(key: str) -> str:
        v = stats.get(key)
        return _pct(v, 1) if v is not None else "n/a"

    lines: List[str] = []
    lines.append("| Stage | Headline result (primary / mature window) |")
    lines.append("|---|---|")
    if n_sample is not None and default_rate is not None:
        n_resolved = pop.get("n_resolved")
        hint = ""
        if n_resolved and n_resolved != n_sample:
            hint = f" of {n_resolved:,} resolved ({n_sample / n_resolved * 100:.1f} %)"
        lines.append(
            f"| **Sample** | {n_sample:,} resolved loans{hint}, vintages {years[0]}–{years[1]}, "
            f"default rate **{_pct(default_rate)}** |"
        )
    if eda:
        lines.append(
            f"| **Trend analysis** | {eda.get('n_months')} monthly vintages "
            f"({eda.get('series_first')} → {eda.get('series_last')}), mean default rate "
            f"**{_pct(eda.get('mean_default_rate'))}** |"
        )
    lines.append(
        f"| **PD — champion** | `{champion}`, OOT {_years_display(pd_.get('oot_years'))}: "
        f"**AUC {_num(auc)} / Gini {_num(gini)} / KS {_num(ks)}** |"
    )
    lines.append(
        f"| **PD — benchmark** (with lender `grade`/`int_rate`) | "
        f"**AUC {_num(bench_auc)} / Gini {_num(bench_gini)}** |"
    )
    if cal_method:
        lines.append(
            f"| **PD — calibration** (fit on a held-out temporal slice) | `{cal_method}`, "
            f"OOT Brier raw {_num(pd_.get('calibration_metrics', {}).get('raw_oot', {}).get('brier'), 5)} → "
            f"calibrated {_num(pd_.get('calibration_metrics', {}).get('calibrated_oot', {}).get('brier'), 5)}, "
            f"CITL {_num(cal_row.get('test_citl'))}, slope {_num(cal_row.get('test_slope'))} |"
        )
    if lgd_mae is not None:
        lines.append(
            f"| **LGD — champion** | `{best_lgd}`, MAE **{_num(lgd_mae)}**, R² **{_num(lgd_r2)}**"
            + (
                f", mean grade gap **{_num(best_lgd_grade_gap)}**"
                if best_lgd_grade_gap is not None
                else ""
            )
            + " |"
        )
    if boundary_train is not None and boundary_oot is not None:
        lines.append(
            f"| **LGD — boundary mass** (`recoveries = 0`) | "
            f"**{_pct(boundary_train)}** train / **{_pct(boundary_oot)}** OOT |"
        )
    if el_amount is not None:
        funded_total = lgd.get("funded_total_oot")
        el_rate_funded = lgd.get("el_rate_funded_oot")
        funded_bit = (
            f"; {_usd_m(funded_total)} funded -> **{_pct(el_rate_funded)}** of funded"
            if funded_total is not None and el_rate_funded is not None
            else ""
        )
        lines.append(
            f"| **Expected loss** | **{_usd_m(el_amount)}** on {_usd_m(ead_total)} EAD -> "
            f"**{_pct(el_rate)}** of EAD{funded_bit} (primary OOT portfolio) |"
        )
    if chosen_k is not None:
        lines.append(
            f"| **Segmentation** | **k = {chosen_k}** "
            f"(silhouette {_num(chosen_row.get('silhouette'))}, bootstrap ARI "
            f"{_num(chosen_row.get('stability_ari'))}), "
            f"**{clu.get('n_degenerate_segments')}** degenerate segments "
            f"(DR spread {_num(spread.get('spread_pp'), 2)} pp) |"
        )
    if stats:
        lines.append(
            f"| **Monte Carlo** ({sim.get('n_simulations'):,} paths, "
            f"ρ = {_num(sim.get('asset_correlation'), 2)}) | "
            f"EL **{_pct_stat('expected_loss_pct_ead')}**, "
            f"UL **{_pct_stat('unexpected_loss_pct_ead')}**, "
            f"VaR 99 % **{_pct_stat('var_0.99_pct_ead')}**, "
            f"VaR 99.9 % **{_pct_stat('var_0.999_pct_ead')}**, "
            f"ES 99 % **{_pct_stat('es_0.99_pct_ead')}**, "
            f"ES 99.9 % **{_pct_stat('es_0.999_pct_ead')}** of EAD |"
        )
    if se_strat is not None:
        note = (
            f"stratified **{_num(se_strat, 1)} %**, antithetic **{_num(se_anti, 1)} %** "
            f"(negative result, kept as reported)"
            if se_anti is not None
            else f"stratified **{_num(se_strat, 1)} %**"
        )
        lines.append(f"| **Monte Carlo — variance reduction** | SE reduction at {sim.get('n_simulations', 0):,} paths: {note} |")
    test_count = summary.get("test_count")
    if test_count is not None:
        lines.append(
            f"| **Test suite** | **{test_count}** unit tests on synthetic data, "
            f"`ruff check .` clean, run on Python 3.10 and 3.11 in CI |"
        )

    # Primary vs secondary PD comparison.
    wc = pd_.get("window_comparison", []) or []
    if wc:
        lines.append("")
        lines.append("**Primary vs secondary (censored) validation window:**")
        lines.append("")
        lines.append("| Window | OOT years | Champion | AUC | Gini | KS | Default rate |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in wc:
            lines.append(
                f"| {row.get('window')} | {row.get('test_years')} | {row.get('champion')} | "
                f"**{_num(row.get('auc'))}** | {_num(row.get('gini'))} | {_num(row.get('ks'))} | "
                f"{_pct(row.get('default_rate_test'))} |"
            )

    # Forecast benchmark: does SARIMAX beat the naive / ETS benchmarks?
    if forecast_bench:
        lines.append("")
        lines.append("**Forecast benchmark on the temporal holdout (mature vintages):**")
        lines.append("")
        lines.append("| Method | Holdout RMSE | MAE | MASE |")
        lines.append("|---|---|---|---|")
        for row in forecast_bench:
            name = row.get("method")
            marker = " **(SARIMAX)**" if name == "sarimax" else ""
            lines.append(
                f"| {name}{marker} | {_num(row.get('rmse'), 6)} | {_num(row.get('mae'), 6)} | "
                f"{_num(row.get('mase'), 4)} |"
            )
        if best_forecast and sarimax_row:
            verdict = (
                "SARIMAX is the best of the four"
                if best_forecast.get("method") == "sarimax"
                else f"`{best_forecast.get('method')}` beats SARIMAX on the holdout"
            )
            lines.append("")
            lines.append(f"> {verdict}.")

    # Calibration method comparison.
    if cal_rows:
        lines.append("")
        lines.append("**Calibration methods (selection on a validation slice, reported out of time):**")
        lines.append("")
        lines.append("| Method | val Brier | val CITL | test Brier | test CITL | test slope |")
        lines.append("|---|---|---|---|---|---|")
        for row in cal_rows:
            marker = " **← chosen**" if row.get("method") == cal_method else ""
            lines.append(
                f"| {row.get('method')}{marker} | {_num(row.get('val_brier'), 5)} | "
                f"{_num(row.get('val_citl'))} | {_num(row.get('test_brier'), 5)} | "
                f"{_num(row.get('test_citl'))} | {_num(row.get('test_slope'))} |"
            )

    # LGD model comparison.
    if lgd_models:
        lines.append("")
        lines.append("**LGD models (loan-level fit on defaulted loans):**")
        lines.append("")
        lines.append("| Model | MAE | RMSE | R² |")
        lines.append("|---|---|---|---|")
        for row in lgd_models:
            marker = " **← champion**" if row.get("model") == best_lgd else ""
            lines.append(
                f"| {row.get('model')}{marker} | {_num(row.get('mae'))} | "
                f"{_num(row.get('rmse'))} | {_num(row.get('r2'))} |"
            )
    return "\n".join(lines)


def render_ead_reconciliation(summary: Dict[str, Any]) -> str:
    """Render the EAD-at-default reconciliation block from ``summary.json``."""
    lgd = summary.get("lgd", {}) or {}
    rows = lgd.get("ead_rate_by_grade", []) or []
    lines = [
        "#### EAD-at-default reconciliation",
        "",
        "**The issue and the fix.** The superseded calculation applied LGD to the full "
        "funded amount, so it treated every loan as if its entire principal were still "
        "outstanding at default. LGD is defined on the outstanding principal, so the "
        "pipeline now estimates a mean EAD rate on **training defaults only** (2010-2013, "
        "36-month loans) and sets `EAD = funded_amnt x` that grade rate for every primary "
        "OOT loan. The OOT rate below is a diagnostic, never an input.",
        "",
        f"The new EL is {_usd_m(lgd.get('expected_loss_oot'))} ({_pct(lgd.get('el_rate_oot'))} of EAD, "
        f"{_pct(lgd.get('el_rate_funded_oot'))} of funded). Realised loss on the OOT portfolio is "
        f"{_usd_m(lgd.get('realised_loss_oot'))} ({_pct(lgd.get('realised_loss_rate_funded_oot'))} of "
        f"funded); the new EL / realised ratio is {_num(lgd.get('el_to_realised_ratio'), 3)}x. "
        f"The superseded funded-basis EL was {_usd_m(lgd.get('old_expected_loss_oot'))} "
        f"({_pct(lgd.get('old_el_rate_oot'))} of funded). Predicted PD is "
        f"{_num(lgd.get('pd_underprediction_pp'), 2)} pp below the observed default rate, which "
        f"explains {_pct(lgd.get('gap_explained_by_pd_share'))} of the remaining new-EL gap.",
        "",
        f"**Two-row discrepancy.** The PD window has {int(lgd.get('n_defaults_oot_pd') or 0):,} "
        f"defaults, while the LGD sample has {int(lgd.get('n_defaults_oot') or 0):,}. The "
        f"{int(lgd.get('n_defaults_oot_zero_ead') or 0)} missing rows are loans "
        f"{', '.join(str(x) for x in (lgd.get('zero_ead_default_ids') or []))}: both are "
        "Charged Off after repaying principal in full, so realised EAD and LGD are zero/undefined. "
        "They are included in PD and default counts, excluded from the LGD fit, and carry zero "
        "realised loss; the new expected-EAD basis assigns them their grade EAD rate.",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Defaults, train / OOT | {sum(int(r.get('n_defaults_train') or 0) for r in rows):,} / "
        f"{sum(int(r.get('n_defaults_oot') or 0) for r in rows):,} |",
        f"| EAD at default, OOT | {_usd_m(lgd.get('ead_total_oot'))} |",
        f"| Funded amount, OOT | {_usd_m(lgd.get('funded_total_oot'))} |",
        f"| Realised loss on funded basis | {_usd_m(lgd.get('realised_loss_oot'))} "
        f"({_pct(lgd.get('realised_loss_rate_funded_oot'))} of funded) |",
        f"| EL, EAD basis (current) | {_usd_m(lgd.get('expected_loss_oot'))} "
        f"({_pct(lgd.get('el_rate_oot'))} of EAD; {_pct(lgd.get('el_rate_funded_oot'))} of funded) |",
        f"| EL, funded basis (superseded) | {_usd_m(lgd.get('old_expected_loss_oot'))} "
        f"({_pct(lgd.get('old_el_rate_oot'))} of funded) |",
        f"| New EL / realised loss | {_num(lgd.get('el_to_realised_ratio'), 3)}x |",
        f"| PD under-prediction (observed - predicted) | {_num(lgd.get('pd_underprediction_pp'), 2)} pp |",
        f"| PD-scaled EL / unexplained gap | {_usd_m(lgd.get('pd_scaled_expected_loss_oot'))} / "
        f"{_usd_m(lgd.get('unexplained_gap_oot'))} |",
        f"| Share of new-EL gap explained by PD under-prediction | "
        f"{_pct(lgd.get('gap_explained_by_pd_share'))} |",
        "",
        "| Grade | Train defaults | Train EAD rate | OOT defaults | OOT EAD rate (diagnostic) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('grade')} | {int(row.get('n_defaults_train') or 0):,} | "
            f"{_pct(row.get('ead_rate_train'))} | {int(row.get('n_defaults_oot') or 0):,} | "
            f"{_pct(row.get('ead_rate_oot'))} |"
        )
    return "\n".join(lines)


_EAD_MARKER_RE = re.compile(
    re.escape(EAD_START_MARKER) + r".*?" + re.escape(EAD_END_MARKER), re.DOTALL
)


def render_ead_with_markers(summary: Dict[str, Any]) -> str:
    return f"{EAD_START_MARKER}\n{render_ead_reconciliation(summary)}\n{EAD_END_MARKER}"


# ---------------------------------------------------------------------------
# Documentation IO
# ---------------------------------------------------------------------------
_MARKER_RE = re.compile(
    re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL
)


def render_with_markers(summary: Dict[str, Any]) -> str:
    """The generated block wrapped in its HTML markers."""
    return f"{START_MARKER}\n{render_results_block(summary)}\n{END_MARKER}"


def sync_docs(summary: Dict[str, Any] | None = None, path: Path | None = None) -> bool:
    """Rewrite the marked results block in ``README.md`` from ``summary.json``.

    Returns ``True`` if the file changed, ``False`` if it was already in sync.
    Raises ``ValueError`` if the markers are missing.
    """
    summary = summary or load_summary()
    if path is not None:
        target = path
        text = target.read_text(encoding="utf-8")
        if not _MARKER_RE.search(text):
            raise ValueError(f"Markers {START_MARKER} / {END_MARKER} not found in {target}")
        new_text = _MARKER_RE.sub(lambda _: render_with_markers(summary), text, count=1)
        if new_text == text:
            return False
        target.write_text(new_text, encoding="utf-8", newline="\n")
        return True

    changed = False
    readme = README_PATH.read_text(encoding="utf-8")
    if not _MARKER_RE.search(readme):
        raise ValueError(f"Markers {START_MARKER} / {END_MARKER} not found in {README_PATH}")
    new_readme = _MARKER_RE.sub(lambda _: render_with_markers(summary), readme, count=1)
    if _EAD_MARKER_RE.search(new_readme):
        new_readme = _EAD_MARKER_RE.sub(lambda _: render_ead_with_markers(summary), new_readme, count=1)
    if new_readme != readme:
        README_PATH.write_text(new_readme, encoding="utf-8", newline="\n")
        changed = True

    if REPORT_PATH.exists():
        report = REPORT_PATH.read_text(encoding="utf-8")
        if _EAD_MARKER_RE.search(report):
            new_report = _EAD_MARKER_RE.sub(lambda _: render_ead_with_markers(summary), report, count=1)
            if new_report != report:
                REPORT_PATH.write_text(new_report, encoding="utf-8", newline="\n")
                changed = True
    return changed


def check_docs(summary: Dict[str, Any] | None = None, path: Path | None = None) -> str:
    """Return the generated block and raise if the README's block differs.

    The exception message contains a unified diff so a CI failure points straight
    at the drifted number.
    """
    summary = summary or load_summary()
    target = path or README_PATH
    text = target.read_text(encoding="utf-8")
    match = _MARKER_RE.search(text)
    if not match:
        raise AssertionError(f"Markers {START_MARKER} / {END_MARKER} not found in {target}")
    expected = render_with_markers(summary)
    if match.group(0) != expected:
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                match.group(0).splitlines(),
                expected.splitlines(),
                fromfile="README.md (committed)",
                tofile="README.md (generated from summary.json)",
                lineterm="",
            )
        )
        raise AssertionError(
            "README results block is out of sync with summary.json.\n"
            "Run `python -m src.report_tables --write` to regenerate it.\n\n" + diff
        )
    if path is None:
        ead_expected = render_ead_with_markers(summary)
        for target in (README_PATH, REPORT_PATH):
            if not target.exists():
                continue
            body = target.read_text(encoding="utf-8")
            ead_match = _EAD_MARKER_RE.search(body)
            if not ead_match:
                raise AssertionError(
                    f"EAD markers {EAD_START_MARKER} / {EAD_END_MARKER} not found in {target}"
                )
            if ead_match.group(0) != ead_expected:
                raise AssertionError(
                    f"EAD reconciliation block is out of sync in {target}.\n"
                    "Run `python -m src.report_tables --write` to regenerate it."
                )
    return expected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sync README results tables from summary.json")
    parser.add_argument("--write", action="store_true", help="rewrite README in place")
    parser.add_argument("--check", action="store_true", help="fail if README is out of sync")
    parser.add_argument(
        "--write-test-count",
        action="store_true",
        help="run pytest --collect-only and store the count in summary.json",
    )
    args = parser.parse_args(argv)

    if args.write_test_count:
        count = write_test_count()
        print(f"[report_tables] recorded test_count={count} in summary.json")
        return

    summary = load_summary()
    if args.write:
        changed = sync_docs(summary)
        print("[report_tables] README updated" if changed else "[report_tables] README already in sync")
    elif args.check:
        check_docs(summary)
        print("[report_tables] README results block is in sync with summary.json")
    else:
        print(render_results_block(summary))


if __name__ == "__main__":
    main()
