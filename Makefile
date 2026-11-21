# One-command pipeline. Run `make help` to list targets.
PYTHON ?= $(shell command -v python3.11 2>/dev/null || echo /opt/homebrew/bin/python3.11)
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: help setup data download clean-data eda discount payback-notebook customers kpi-notebook analysis forecast tag export all test lint format

help:
	@echo "setup      create .venv, install deps, install pre-commit hooks"
	@echo "data       download raw data from Kaggle, then clean -> Parquet"
	@echo "download   only download raw data"
	@echo "clean-data only clean raw data -> data/interim/<dev|full>/"
	@echo "eda        run notebooks/01_eda.ipynb end to end (figures -> reports/figures/)"
	@echo "discount   discount payback tables -> reports/discount/ (FDDA_MODE=full for final numbers)"
	@echo "customers  KPIs, RFM, cohorts, win-back playbook -> reports/customers/"
	@echo "analysis   eda + discount + customers + their notebooks"
	@echo "forecast   backtest + 12-week forecast -> reports/forecast/, notebook 04"
	@echo "tag        train/evaluate the product tagger, tag Myntra, notebook 05"
	@echo "export     (Phase 7)"
	@echo "all        everything, in order"
	@echo "test | lint | format"

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,analysis,forecast,tagging]"
	$(VENV)/bin/pre-commit install

data: download clean-data

download:
	$(PY) -m fdda.data.download

clean-data:
	$(PY) -m fdda.data.clean

eda:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

discount:
	$(PY) -m fdda.analysis.discount

payback-notebook:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_discount_payback.ipynb

customers:
	$(PY) -m fdda.analysis.customers

kpi-notebook:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_kpis_and_rfm.ipynb

analysis: eda discount payback-notebook customers kpi-notebook

forecast:
	$(PY) -m fdda.forecast.backtest
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_forecasting.ipynb

tag:
	$(PY) -m fdda.tagging.run all
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/05_product_tagging.ipynb

export:
	@echo "export: not built yet (Phase 7)"; exit 1

all: data analysis forecast tag export

test:
	$(PY) -m pytest

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

format:
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .
