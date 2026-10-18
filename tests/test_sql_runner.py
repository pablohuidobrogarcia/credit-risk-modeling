"""SQL suite splitting and execution."""
from __future__ import annotations

import pytest

from src.sql_runner import _is_comment_only, _split_statements


def test_split_ignores_semicolons_in_comments():
    text = "SELECT 1; -- not a statement; really\nSELECT 2;\n"
    stmts = [s.strip() for s in _split_statements(text) if s.strip()]
    assert len(stmts) == 2
    assert stmts[0].startswith("SELECT 1")
    assert stmts[1].startswith("SELECT 2")


def test_comment_only_detection():
    assert _is_comment_only("-- just a note\n   \n")
    assert not _is_comment_only("SELECT 1")


def test_run_sql_file_on_synthetic(tmp_path):
    duckdb = pytest.importorskip("duckdb")  # noqa: F841  (skip the test if DuckDB is unavailable)
    import pandas as pd

    from src.sql_runner import run_sql_file

    df = pd.DataFrame({"grade": ["A", "A", "B"], "default_flag": [0, 1, 0], "issue_year": [2015, 2015, 2016]})
    parquet = tmp_path / "sample.parquet"
    df.to_parquet(parquet)

    sql = tmp_path / "q.sql"
    sql.write_text(
        "SELECT grade, COUNT(*) AS n, AVG(default_flag) AS default_rate\n"
        "FROM accepted\n"
        "GROUP BY grade\n"
        "ORDER BY grade;\n",
        encoding="utf-8",
    )
    res = run_sql_file(sql, source=parquet)
    out = list(res.values())[0]
    assert list(out["grade"]) == ["A", "B"]
    assert out.loc[out["grade"] == "A", "n"].iloc[0] == 2
