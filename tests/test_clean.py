"""Tests for the cleaning rules in fdda.data.clean, run on synthetic fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from conftest import run_hm, run_myntra
from fdda.data.clean import sample_predicate, write_summary


def _tx(out: Path) -> pd.DataFrame:
    """Read cleaned transactions (small fixture, so pandas is fine here)."""
    glob = str(out / "transactions" / "*" / "*.parquet")
    return duckdb.sql(f"SELECT * FROM read_parquet('{glob}', hive_partitioning = true)").df()


# ---------------------------------------------------------------- H&M


def test_hm_drop_counts_match_planted_edge_cases(hm_full, expected):
    summary, _ = hm_full
    assert summary["raw_rows"] == expected["hm_raw_rows"]
    assert summary["dropped_missing_or_unparseable"] == expected["hm_dropped_missing"]
    assert summary["dropped_nonpositive_price"] == expected["hm_dropped_price"]
    assert summary["dropped_unknown_article"] == expected["hm_dropped_article"]
    assert summary["dropped_dev_sample"] == 0
    assert summary["final_rows"] == expected["hm_final_rows"]


def test_hm_output_has_no_invalid_rows(hm_full):
    _, out = hm_full
    tx = _tx(out)
    assert (tx["price"] > 0).all()
    assert tx[["t_dat", "customer_id", "article_id"]].notna().all().all()
    assert "0999999001" not in set(tx["article_id"])


def test_hm_duplicates_are_kept_as_units(hm_full, expected):
    summary, out = hm_full
    tx = _tx(out)
    key = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    assert len(tx) - len(tx.drop_duplicates(key)) >= expected["hm_duplicates"]
    assert summary["duplicate_rows_kept_as_units"] >= expected["hm_duplicates"]


def test_hm_article_ids_are_padded_to_10_digits(hm_full):
    _, out = hm_full
    tx = _tx(out)
    assert tx["article_id"].str.len().eq(10).all()
    assert tx["product_code"].str.len().eq(7).all()


def test_hm_category_hierarchy_joined(hm_full):
    _, out = hm_full
    tx = _tx(out)
    for column in ["product_group_name", "product_type_name", "section_name", "garment_group_name"]:
        assert tx[column].notna().all(), column


def test_hm_partitioned_by_month(hm_full):
    summary, out = hm_full
    parts = sorted(p.name for p in (out / "transactions").iterdir())
    assert parts == [f"year_month={m}" for m in summary["rows_by_month"]]
    assert parts[0] == "year_month=2018-09" and parts[-1] == "year_month=2019-03"
    assert sum(summary["rows_by_month"].values()) == summary["final_rows"]


def test_hm_date_range_logged(hm_full):
    summary, _ = hm_full
    assert summary["date_min"] >= "2018-09-20"
    assert summary["date_max"] <= "2019-03-31"


def test_hm_indexed_price_uses_rescale_factor(hm_full):
    _, out = hm_full
    tx = _tx(out)
    assert (tx["indexed_price"] - tx["price"] * 1000).abs().max() < 1e-9


def test_hm_price_outliers_flagged_not_dropped(hm_full, expected):
    summary, out = hm_full
    assert summary["price_outlier_rows_flagged"] == expected["hm_outliers"]
    assert _tx(out)["is_price_outlier"].sum() == expected["hm_outliers"]


def test_hm_articles_and_customers_written(hm_full):
    _, out = hm_full
    articles = pd.read_parquet(out / "articles.parquet")
    customers = pd.read_parquet(out / "customers.parquet")
    assert articles["article_id"].is_unique and len(articles) == 40
    assert set(customers["customer_id"]) == set(_tx(out)["customer_id"])


def test_hm_dev_sample_is_customer_level_and_deterministic(hm_raw, hm_full, tmp_path):
    _, full_out = hm_full
    full = _tx(full_out)
    dev1, dev2 = tmp_path / "dev1", tmp_path / "dev2"
    s1 = run_hm(hm_raw, dev1, dev_mode=True, fraction=0.1)
    run_hm(hm_raw, dev2, dev_mode=True, fraction=0.1)
    a, b = _tx(dev1), _tx(dev2)

    assert set(a["customer_id"]) == set(b["customer_id"])  # deterministic
    n_all = full["customer_id"].nunique()
    assert 0 < a["customer_id"].nunique() < 0.3 * n_all  # roughly 10%
    # Every sampled customer keeps all of their rows.
    full_counts = full.groupby("customer_id").size()
    dev_counts = a.groupby("customer_id").size()
    assert (full_counts.loc[dev_counts.index] == dev_counts).all()
    assert s1["final_rows"] + s1["dropped_dev_sample"] == len(full)


def test_hm_dev_sample_changes_with_seed(hm_raw, tmp_path):
    run_hm(hm_raw, tmp_path / "s1", dev_mode=True, seed=1, fraction=0.3)
    run_hm(hm_raw, tmp_path / "s2", dev_mode=True, seed=2, fraction=0.3)
    c1 = set(_tx(tmp_path / "s1")["customer_id"])
    c2 = set(_tx(tmp_path / "s2")["customer_id"])
    assert c1 != c2


def test_hm_missing_raw_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="make download"):
        run_hm(tmp_path / "nope", tmp_path / "out")


def test_sample_predicate_rejects_bad_fraction():
    with pytest.raises(ValueError):
        sample_predicate("x", 0, 42)
    with pytest.raises(ValueError):
        sample_predicate("x", 1.5, 42)


# ---------------------------------------------------------------- Myntra


def test_myntra_drop_counts_match_planted_edge_cases(myntra_full, expected):
    summary, _ = myntra_full
    assert summary["raw_rows"] == expected["my_raw_rows"]
    assert summary["dropped_missing_id_or_name"] == expected["my_dropped_missing"]
    assert summary["dropped_duplicate_id"] == expected["my_dropped_duplicate"]
    assert summary["dropped_invalid_price"] == expected["my_dropped_price"]
    assert summary["dropped_invalid_mrp"] == expected["my_dropped_mrp"]
    assert summary["final_rows"] == expected["my_final_rows"]


def test_myntra_discount_computed_and_clipped(myntra_full, expected):
    summary, out = myntra_full
    df = pd.read_parquet(out / "products.parquet")
    assert df["discount_pct"].between(0, 0.9).all()
    assert summary["clipped_discount_rows"] == expected["my_clipped"]
    unclipped = df[df["discount_pct"] < 0.9]
    recomputed = 1 - unclipped["price"] / unclipped["mrp"]
    assert (recomputed - unclipped["discount_pct"]).abs().max() < 1e-9
    assert (df["mrp"] >= df["price"]).all()


def test_myntra_formatted_prices_parsed(myntra_full):
    _, out = myntra_full
    df = pd.read_parquet(out / "products.parquet")
    parsed = df[(df["price"] == 1299) & (df["mrp"] == 2599)]
    assert len(parsed) >= 3


def test_myntra_duplicate_keeps_first_occurrence(myntra_full):
    _, out = myntra_full
    df = pd.read_parquet(out / "products.parquet")
    assert df["id"].is_unique
    assert not (df["price"] == 1).any()  # the later duplicates had price "1"


def test_myntra_dev_sample_is_subset(myntra_raw, myntra_full, tmp_path):
    _, full_out = myntra_full
    summary = run_myntra(myntra_raw, tmp_path / "dev", dev_mode=True)
    full_ids = set(pd.read_parquet(full_out / "products.parquet")["id"])
    dev_ids = set(pd.read_parquet(tmp_path / "dev" / "products.parquet")["id"])
    assert dev_ids < full_ids and len(dev_ids) == summary["final_rows"]


def test_write_summary_roundtrip(hm_full, tmp_path):
    summary, _ = hm_full
    path = tmp_path / "reports" / "summary.json"
    write_summary(summary, path)
    assert json.loads(path.read_text())["final_rows"] == summary["final_rows"]
