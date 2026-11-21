"""Shared pytest fixtures: copy the synthetic CSVs into a temp raw/ layout and clean them."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from fdda.data.clean import clean_hm, clean_myntra

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))

from make_fixtures import EXPECTED  # noqa: E402

MYNTRA_COLUMNS = {
    "id": "id",
    "name": "name",
    "price": "price",
    "mrp": "mrp",
    "rating": "rating",
    "rating_total": "ratingTotal",
    "seller": "seller",
}


@pytest.fixture
def expected() -> dict:
    """Exact counts of the planted edge cases."""
    return EXPECTED


@pytest.fixture
def hm_raw(tmp_path: Path) -> Path:
    """Temp copy of the H&M fixture files."""
    dest = tmp_path / "raw" / "hm"
    shutil.copytree(FIXTURES / "hm", dest)
    return dest


@pytest.fixture
def myntra_raw(tmp_path: Path) -> Path:
    """Temp copy of the Myntra fixture file."""
    dest = tmp_path / "raw" / "myntra"
    shutil.copytree(FIXTURES / "myntra", dest)
    return dest


def run_hm(raw: Path, out: Path, *, dev_mode: bool = False, seed: int = 42, fraction: float = 0.1):
    """Clean the H&M fixture with test defaults."""
    return clean_hm(
        raw,
        out,
        dev_mode=dev_mode,
        sample_fraction=fraction,
        seed=seed,
        price_rescale_factor=1000,
        price_outlier_multiple=5,
    )


def run_myntra(raw: Path, out: Path, *, dev_mode: bool = False):
    """Clean the Myntra fixture with test defaults."""
    return clean_myntra(
        raw,
        out,
        dev_mode=dev_mode,
        sample_fraction=0.5,
        seed=42,
        max_discount=0.9,
        columns=MYNTRA_COLUMNS,
    )


@pytest.fixture
def hm_full(hm_raw: Path, tmp_path: Path) -> tuple[dict, Path]:
    """H&M fixture cleaned in full mode: (summary, output dir)."""
    out = tmp_path / "interim" / "full" / "hm"
    return run_hm(hm_raw, out), out


@pytest.fixture
def myntra_full(myntra_raw: Path, tmp_path: Path) -> tuple[dict, Path]:
    """Myntra fixture cleaned in full mode: (summary, output dir)."""
    out = tmp_path / "interim" / "full" / "myntra"
    return run_myntra(myntra_raw, out), out
