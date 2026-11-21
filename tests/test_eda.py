"""Tests for fdda.analysis.eda and fdda.features.pricing.band_case_sql on fixtures."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from fdda.analysis import eda
from fdda.config import load_settings
from fdda.features.pricing import band_case_sql

EDGES = [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.90]
LABELS = ["0-5%", "5-15%", "15-25%", "25-35%", "35-45%", "45-55%", "55%+"]


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


@pytest.fixture
def interim(hm_full, myntra_full) -> Path:
    """Interim dir holding both cleaned fixtures (hm/ and myntra/)."""
    _, hm_out = hm_full
    return hm_out.parent


def test_band_case_sql_boundaries(con):
    case = band_case_sql("d", EDGES, LABELS)
    got = [
        con.execute(f"SELECT {case} FROM (SELECT {v}::DOUBLE AS d)").fetchone()[0]
        for v in [0, 0.049, 0.05, 0.149, 0.15, 0.549, 0.55, 0.9]
    ]
    assert got == ["0-5%", "0-5%", "5-15%", "5-15%", "15-25%", "45-55%", "55%+", "55%+"]


def test_band_case_sql_validates_inputs():
    with pytest.raises(ValueError):
        band_case_sql("d", EDGES, LABELS[:-1])
    with pytest.raises(ValueError):
        band_case_sql("d", [0.5, 0.1, 0.9], ["a", "b"])


def test_cleaning_steps_table_ends_at_final_rows(hm_full):
    summary, _ = hm_full
    table = eda.cleaning_steps_table(summary)
    assert table["rows_left"].iloc[0] == summary["raw_rows"]
    assert table["rows_left"].iloc[-1] == summary["final_rows"]
    assert table["rows_removed"].sum() == summary["raw_rows"] - summary["final_rows"]


def test_data_quality_has_row_per_column(con, interim):
    tx = eda.hm_transactions(interim)
    dq = eda.data_quality(con, tx)
    n_cols = len(con.execute(f"SELECT * FROM {tx} LIMIT 0").df().columns)
    assert len(dq) == n_cols
    assert (dq["null_pct"] == 0).all()  # cleaned output has no nulls


def test_hm_category_mix_shares_and_other(con, interim, expected):
    mix = eda.hm_category_mix(con, eda.hm_transactions(interim), top_n=3)
    assert len(mix) == 4 and mix["category"].iloc[-1] == "Other"
    assert mix["units"].sum() == expected["hm_final_rows"]
    assert mix["unit_share"].sum() == pytest.approx(1)
    assert mix["revenue_share"].sum() == pytest.approx(1)


def test_hm_category_mix_rejects_bad_column(con, interim):
    with pytest.raises(ValueError):
        eda.hm_category_mix(con, eda.hm_transactions(interim), level="x; DROP")


def test_hm_weekly_sales_sum_to_total(con, interim, expected):
    weekly = eda.hm_weekly_sales(con, eda.hm_transactions(interim))
    assert weekly["units"].sum() == expected["hm_final_rows"]
    assert weekly["week"].is_monotonic_increasing


def test_hm_discount_band_mix_covers_all_units(con, interim, expected):
    mix = eda.hm_discount_band_mix(con, eda.hm_transactions(interim), EDGES, LABELS, 0.9)
    assert list(mix["band"]) == LABELS
    assert mix["units"].sum() == expected["hm_final_rows"]
    assert mix["units_share"].sum() == pytest.approx(1)
    # Fixture markdowns are 0/10/20/30/50%, so full price must be the biggest band.
    assert mix.loc[mix["band"] == "0-5%", "units_share"].item() == mix["units_share"].max()


def test_myntra_category_rules_from_config(con):
    rules = load_settings()["eda"]["myntra_category_rules"]
    case = eda.myntra_category_sql("n", rules)
    cases = {
        "Men Solid Oversized Cotton T-shirt": "Tops & T-shirts",
        "Men Slim Fit Casual Shirt": "Shirts",
        "Women Floral Kurta with Palazzos": "Ethnic wear",
        "Gold-Plated Jhumka Earrings": "Jewellery & watches",
        "Set of 2 Cotton Pillow Covers": "Home",
        "Unisex Running Shoes": "Footwear",
        "Mystery Item": "Other",
    }
    for name, want in cases.items():
        got = con.execute(f"SELECT {case} FROM (SELECT ? AS n)", [name]).fetchone()[0]
        assert got == want, name


def test_myntra_mixes(con, interim, expected):
    rules = load_settings()["eda"]["myntra_category_rules"]
    products = eda.myntra_products(interim)
    cat = eda.myntra_category_mix(con, products, rules)
    assert cat["listings"].sum() == expected["my_final_rows"]
    assert cat["listing_share"].sum() == pytest.approx(1)
    bands = eda.myntra_discount_band_mix(con, products, EDGES, LABELS)
    assert list(bands["band"]) == LABELS
    assert bands["listings"].sum() == expected["my_final_rows"]


def test_samples_are_repeatable(con, interim):
    tx = eda.hm_transactions(interim)
    a = eda.hm_price_sample(con, tx, 50, 42, ["Garment Upper body"])
    b = eda.hm_price_sample(con, tx, 50, 42, ["Garment Upper body"])
    assert a.equals(b) and len(a) == 50
    assert set(a["category"]) <= {"Garment Upper body", "Other"}


def test_save_table_writes_csv(tmp_path, interim, con):
    mix = eda.hm_channel_mix(con, eda.hm_transactions(interim))
    path = eda.save_table(mix, "channel", tmp_path)
    assert path.exists() and path.read_text().startswith("sales_channel_id")
