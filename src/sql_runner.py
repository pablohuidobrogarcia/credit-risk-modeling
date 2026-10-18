"""Execute the documented SQL suite (HiveQL-style) against DuckDB.

The SQL files in ``sql/`` are written to be portable to Hive/Spark. Here they
are executed by DuckDB over the parquet extract, which is the local stand-in
for a distributed warehouse.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd

from .config import Config, project_root


def _register_views(con: duckdb.DuckDBPyConnection, source: Path) -> None:
    con.execute(f"CREATE OR REPLACE VIEW accepted AS SELECT * FROM '{source.as_posix()}'")


def run_sql_file(sql_path: Path, source: Path | None = None) -> Dict[str, pd.DataFrame]:
    """Run every statement in a ``.sql`` file and return {label: DataFrame}."""
    c = Config.load()
    if source is None:
        source = c.path("modeling_parquet")
    con = duckdb.connect()
    con.execute("SET threads TO 2")
    _register_views(con, source)
    text = sql_path.read_text(encoding="utf-8")
    statements = [s.strip() for s in _split_statements(text) if s.strip()]
    results: Dict[str, pd.DataFrame] = {}
    for i, stmt in enumerate(statements, 1):
        df = con.execute(stmt).df()
        label = _label_from_statement(stmt, i)
        results[label] = df
    con.close()
    return results


def _strip_comments(text: str) -> str:
    """Remove ``--`` line comments so that ``;`` inside comments is ignored."""
    lines = []
    for line in text.splitlines():
        idx = line.find("--")
        lines.append(line[:idx] if idx >= 0 else line)
    return "\n".join(lines)


def _split_statements(text: str):
    return _strip_comments(text).split(";")


def _is_comment_only(statement: str) -> bool:
    lines = [ln for ln in statement.splitlines() if ln.strip() and not ln.strip().startswith("--")]
    return len(lines) == 0


def _label_from_statement(statement: str, index: int) -> str:
    body = statement.strip().lower()
    if body.startswith("select") and "from accepted" in body and "group by grade, issue_year" in body:
        return f"stmt_{index}_grade_by_year"
    if "group by grade" in body:
        return f"stmt_{index}_by_grade"
    if "group by issue_year, issue_ym" in body:
        return f"stmt_{index}_vintage"
    return f"stmt_{index}"


def run_all_sql(source: Path | None = None) -> Dict[str, pd.DataFrame]:
    """Run every SQL file in ``sql/`` and persist small result tables."""
    sql_dir = project_root() / "sql"
    all_results: Dict[str, pd.DataFrame] = {}
    for path in sorted(sql_dir.glob("*.sql")):
        for label, df in run_sql_file(path, source=source).items():
            key = f"{path.stem}__{label}"
            all_results[key] = df
            df.to_csv(project_root() / "reports" / "results" / f"sql_{key}.csv", index=False)
    return all_results
