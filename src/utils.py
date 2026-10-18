"""Shared helpers: reproducibility, IO, plotting setup."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")  # headless / CI friendly
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config, project_root  # noqa: E402

# Consistent, professional look for every figure in the project.
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.autolayout": False,
    }
)

FIGURES_DIR = project_root() / "reports" / "figures"
RESULTS_DIR = project_root() / "reports" / "results"


def set_seeds(seed: int = 42) -> None:
    """Set every relevant global seed so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Persist a figure under ``reports/figures`` with a stable name."""
    ensure_dirs()
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def save_json(obj: Dict[str, Any], name: str) -> Path:
    """Persist a (small) JSON artifact under ``reports/results``."""
    ensure_dirs()
    path = RESULTS_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """Persist a results table as CSV under ``reports/results``."""
    ensure_dirs()
    path = RESULTS_DIR / name
    df.to_csv(path, index=index)
    return path


def load_json(name: str) -> Dict[str, Any]:
    with open(RESULTS_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)
