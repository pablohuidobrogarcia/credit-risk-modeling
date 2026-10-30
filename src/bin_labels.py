"""Human-readable labels for the scorecard bins.

The exported points table stores WoE-bin keys such as
``(2.6e+04, 3.1e+04] & (3.1e+04, 3.5e+04]``.  Those keys are exact and must
never change (the simulator and the parity test depend on them), but they are
unreadable in a select.  This module turns each key into a short label while
keeping the original key alongside it.

The formatter is deliberately pure: it takes a variable name and a bin string
and returns text.  The browser has a mirror implementation for fallbacks and
the tests assert that the two agree on the exported cases.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

VARIABLE_LABELS = {
    "annual_inc": "Annual income ($)",
    "dti": "Debt-to-income ratio (%)",
    "inq_last_6mths": "Credit inquiries in the last 6 months",
    "revol_util": "Credit card utilization (%)",
    "installment_to_income": "Monthly payment vs income",
    "home_ownership": "Home ownership",
    "purpose": "Loan purpose",
}

VARIABLE_HELP = {
    "annual_inc": "Gross yearly income reported at application.",
    "dti": "Monthly debt payments divided by monthly income.",
    "inq_last_6mths": "How many lenders checked the applicant's credit recently.",
    "revol_util": "Share of available revolving credit currently used.",
    "installment_to_income": "Monthly loan payment as a share of monthly income.",
    "home_ownership": "Whether the applicant rents, owns, or has a mortgage.",
    "purpose": "The reason the borrower gave for the loan.",
}

CATEGORY_LABELS = {
    "MORTGAGE": "Mortgage",
    "OWN": "Own home",
    "RENT": "Rent",
    "ANY": "Any",
    "NONE": "None",
    "__missing__": "Not reported",
    "__other__": "Other / rare categories",
    "credit_card": "Credit card",
    "debt_consolidation": "Debt consolidation",
    "home_improvement": "Home improvement",
    "major_purchase": "Major purchase",
    "other": "Other",
    "medical": "Medical",
    "car": "Car",
    "small_business": "Small business",
    "house": "House",
    "moving": "Moving",
    "vacation": "Vacation",
    "renewable_energy": "Renewable energy",
    "educational": "Educational",
    "wedding": "Wedding",
}

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_INTERVAL_RE = re.compile(
    r"^\s*[\(\[]\s*(?P<lo>-inf|" + _NUM + r")\s*,\s*(?P<hi>inf|" + _NUM + r")\s*[\)\]]\s*$"
)


def _fmt_number(value: float, decimals: int = 1) -> str:
    if not math.isfinite(value):
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def _fmt_money(value: float) -> str:
    return f"${int(round(value)):,}"


def _parts(bin_key: str) -> List[Any]:
    """Split a bin key into interval tuples or category strings."""
    parts: List[Any] = []
    for raw in str(bin_key).split("&"):
        text = raw.strip()
        match = _INTERVAL_RE.match(text)
        if match:
            lo_raw, hi_raw = match.group("lo"), match.group("hi")
            lo = None if lo_raw == "-inf" else float(lo_raw)
            hi = None if hi_raw == "inf" else float(hi_raw)
            parts.append((lo, hi))
        elif text:
            parts.append(text)
    return parts


def _format_interval(variable: str, lo: Optional[float], hi: Optional[float]) -> str:
    if variable == "inq_last_6mths":
        if lo is None and hi is not None:
            return "0 or 1" if hi <= 1 else f"0 to {int(hi)}"
        if lo is not None and hi is None:
            return f"{int(round(lo)) + 1} or more"
        if lo is not None and hi is not None:
            if hi - lo <= 1:
                return str(int(round(lo)) + 1)
            return f"{int(round(lo)) + 1} to {int(round(hi))}"
    if variable == "annual_inc":
        if lo is None and hi is not None:
            return f"Up to {_fmt_money(hi)}"
        if lo is not None and hi is None:
            return f"More than {_fmt_money(lo)}"
        if lo is not None and hi is not None:
            return f"{_fmt_money(lo)} - {_fmt_money(hi)}"
    if variable in {"dti", "revol_util", "installment_to_income"}:
        scale = 100.0 if variable == "installment_to_income" else 1.0
        unit = "%"
        if lo is None and hi is not None:
            return f"Up to {_fmt_number(hi * scale)}{unit}"
        if lo is not None and hi is None:
            return f"More than {_fmt_number(lo * scale)}{unit}"
        if lo is not None and hi is not None:
            return f"{_fmt_number(lo * scale)}{unit} - {_fmt_number(hi * scale)}{unit}"
    if lo is None and hi is None:
        return "Any value"
    if lo is None:
        return f"Up to {_fmt_number(hi)}"
    if hi is None:
        return f"More than {_fmt_number(lo)}"
    return f"{_fmt_number(lo)} - {_fmt_number(hi)}"


def _adjacent(a: Tuple[Optional[float], Optional[float]], b: Tuple[Optional[float], Optional[float]]) -> bool:
    return a[1] is not None and b[0] is not None and abs(a[1] - b[0]) < 1e-9


def format_bin(variable: str, bin_key: str) -> str:
    """Return a readable label for one exported bin key."""
    parts = _parts(bin_key)
    if not parts:
        return "Any value"
    if all(isinstance(p, str) for p in parts):
        return ", or ".join(CATEGORY_LABELS.get(str(p), str(p).replace("_", " ").strip().capitalize()) for p in parts)
    if any(isinstance(p, str) for p in parts):
        labels = []
        for part in parts:
            if isinstance(part, str):
                labels.append(CATEGORY_LABELS.get(part, part.replace("_", " ").strip().capitalize()))
            else:
                labels.append(_format_interval(variable, part[0], part[1]))
        return ", or ".join(labels)

    intervals = sorted(parts, key=lambda p: (p[0] is not None, p[0] if p[0] is not None else 0))
    merged: List[List[Optional[float]]] = []
    for lo, hi in intervals:
        if merged and _adjacent((merged[-1][0], merged[-1][1]), (lo, hi)):
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    if len(merged) == 1:
        return _format_interval(variable, merged[0][0], merged[0][1])
    return ", or ".join(_format_interval(variable, lo, hi) for lo, hi in merged)


def variable_label(variable: str) -> str:
    return VARIABLE_LABELS.get(variable, variable.replace("_", " ").strip().capitalize())


def variable_help(variable: str) -> str:
    return VARIABLE_HELP.get(variable, "Applicant characteristic used by the scorecard.")


def _sort_key(variable: str, option: Dict[str, Any]) -> Tuple[int, float, int]:
    parts = _parts(option["key"])
    if variable in {"home_ownership", "purpose"} or not parts or isinstance(parts[0], str):
        return (1, 0.0, -int(option.get("n", 0)))
    first = parts[0]
    lo = first[0] if first[0] is not None else -1e18
    return (0, float(lo), -int(option.get("n", 0)))


def build_var_meta(
    points_rows: Sequence[Dict[str, Any]],
    test_counts: Optional[Dict[str, Dict[str, int]]] = None,
    base_points: float = 0.0,
) -> Dict[str, Any]:
    """Build select options, shares, presets and labels for the simulator."""
    test_counts = test_counts or {}
    by_var: Dict[str, List[Dict[str, Any]]] = {}
    for row in points_rows:
        variable = str(row["variable"])
        counts = test_counts.get(variable, {})
        key = str(row["bin"])
        n_test = int(counts.get(key, 0)) if counts else int(row.get("n", 0))
        by_var.setdefault(variable, []).append({
            "key": key,
            "label": format_bin(variable, key),
            "n": n_test,
            "points": round(float(row.get("points", 0.0)), 3),
            "woe": round(float(row.get("woe", 0.0)), 8),
        })

    meta: Dict[str, Any] = {}
    for variable, options in by_var.items():
        options.sort(key=lambda opt: _sort_key(variable, opt))
        if variable in {"home_ownership", "purpose"}:
            options.sort(key=lambda opt: (-int(opt["n"]), str(opt["label"])))
        total = sum(int(opt["n"]) for opt in options)
        for opt in options:
            opt["share"] = round((opt["n"] / total), 6) if total else 0.0
        default = max(options, key=lambda opt: (int(opt["n"]), opt["points"]))
        lower = max(options, key=lambda opt: opt["points"])
        higher = min(options, key=lambda opt: opt["points"])
        meta[variable] = {
            "variable": variable,
            "label": variable_label(variable),
            "help": variable_help(variable),
            "options": options,
            "default": default["key"],
            "lower_risk": lower["key"],
            "higher_risk": higher["key"],
        }

    presets = {
        "typical": {v: m["default"] for v, m in meta.items()},
        "lower": {v: m["lower_risk"] for v, m in meta.items()},
        "higher": {v: m["higher_risk"] for v, m in meta.items()},
    }
    return {"variables": meta, "presets": presets, "base_points": round(base_points, 3)}


def test_counts_from_woe(
    woe_rows: Iterable[Dict[str, Any]],
    points_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """Count test-window rows in each published scorecard bin via exact WoE."""
    lookup: Dict[str, Dict[float, str]] = {}
    for row in points_rows:
        variable = str(row["variable"])
        lookup.setdefault(variable, {})[round(float(row["woe"]), 6)] = str(row["bin"])
    counts: Dict[str, Dict[str, int]] = {}
    for row in woe_rows:
        for variable, woe_to_bin in lookup.items():
            if variable not in row:
                continue
            key = woe_to_bin.get(round(float(row[variable]), 6))
            if key is None:
                continue
            counts.setdefault(variable, {}).setdefault(key, 0)
            counts[variable][key] += 1
    return counts
