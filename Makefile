# One-command pipeline. Run `make help` to list targets.
PYTHON ?= $(shell command -v python3.11 2>/dev/null || echo /opt/homebrew/bin/python3.11)
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: help setup data download clean-data analysis forecast tag export all test lint format

help:
	@echo "setup      create .venv, install deps, install pre-commit hooks"
	@echo "data       download raw data from Kaggle, then clean -> Parquet"
	@echo "download   only download raw data"
	@echo "clean-data only clean raw data -> data/interim/<dev|full>/"
	@echo "analysis | forecast | tag | export   (later phases)"
	@echo "all        everything, in order"
	@echo "test | lint | format"

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	$(VENV)/bin/pre-commit install

data: download clean-data

download:
	$(PY) -m fdda.data.download

clean-data:
	$(PY) -m fdda.data.clean

analysis:
	@echo "analysis: not built yet (Phases 3-4)"; exit 1

forecast:
	@echo "forecast: not built yet (Phase 5)"; exit 1

tag:
	@echo "tag: not built yet (Phase 6)"; exit 1

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
