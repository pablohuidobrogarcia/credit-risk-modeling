# Credit-risk modeling pipeline
# Usage: make <target>   (on Windows use ./run.ps1 or python -m src.pipeline)
PY ?= python

.PHONY: help install data quality sql eda pd lgd clustering simulation all test lint docs docs-check clean

help:
	@echo "Targets:"
	@echo "  make install      install Python dependencies"
	@echo "  make data         convert raw CSVs to parquet + stratified sample"
	@echo "  make all          run the full pipeline end to end"
	@echo "  make test         run the unit-test suite"
	@echo "  make lint         run ruff"
	@echo "  make docs         regenerate the README results tables from summary.json"
	@echo "  make docs-check   fail if the README results tables drifted"
	@echo "  make clean        remove generated artefacts (keeps data/raw)"

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.pipeline data

quality:
	$(PY) -m src.pipeline quality

sql:
	$(PY) -m src.pipeline sql

eda:
	$(PY) -m src.pipeline eda

pd:
	$(PY) -m src.pipeline pd

lgd:
	$(PY) -m src.pipeline lgd

clustering:
	$(PY) -m src.pipeline clustering

simulation:
	$(PY) -m src.pipeline simulation

all:
	$(PY) -m src.pipeline all

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

docs:
	$(PY) -m src.report_tables --write

docs-check:
	$(PY) -m src.report_tables --check

clean:
	$(PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['reports/figures','reports/results','data/processed']]; [p.unlink(missing_ok=True) for p in pathlib.Path('.').glob('reports/results/_*.log')]"
