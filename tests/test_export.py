"""Tests for the Power BI star-schema export (fdda.export.powerbi)."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from fdda.analysis import customers as cu
from fdda.analysis import discount as dc
from fdda.config import load_settings
from fdda.export import powerbi as pbi


def test_dim_date_daily_with_week_and_month_keys():
    d = pbi.build_dim_date("2020-01-06", "2020-02-09", "2020-01-08", "2020-02-02")
    assert len(d) == 35 and d["date_key"].is_unique
    assert d.loc[d.date == "2020-01-08", "week_start"].item() == pd.Timestamp("2020-01-06")
    assert d.loc[d.date == "2020-02-05", "month_start"].item() == pd.Timestamp("2020-02-01")
    assert d["in_sales_data"].sum() == 26
    assert d.loc[d.date == "2020-02-03", "is_forecast_period"].item()
    assert d["date_key"].iloc[0] == 20200106


def test_dim_discount_band_flags_ceiling():
    b = pbi.build_dim_discount_band([0, 0.05, 0.15, 0.9], ["0-5%", "5-15%", "15%+"], "5-15%")
    assert b["is_ceiling_band"].tolist() == [False, True, False]
    assert b["is_full_price"].tolist() == [True, False, False]
    assert b["band_order"].tolist() == [1, 2, 3]


@pytest.fixture
def exported(hm_full, myntra_full, tmp_path):
    settings = copy.deepcopy(load_settings())
    settings["_root"] = tmp_path
    settings["paths"].update({"interim": "interim", "processed": "processed", "reports": "reports"})
    settings["duckdb"] = {}
    settings["discount_analysis"].update(
        {"min_articles_per_category": 5, "min_articles_per_band": 1}
    )
    dc.run(settings, "full")
    cu_info = cu.run(settings, "full")
    info = pbi.export(settings, "full")
    out = tmp_path / "processed" / "powerbi"
    tables = {p.stem: pd.read_parquet(p) for p in out.glob("*.parquet")}
    return info, tables, cu_info, tmp_path


def test_export_writes_star_schema(exported):
    info, tables, cu_info, root = exported
    expected = {
        "dim_date",
        "dim_article",
        "dim_product_group",
        "dim_channel",
        "dim_discount_band",
        "dim_customer_segment",
        "fact_sales",
        "fact_kpi_monthly",
        "fact_customer",
        "fact_cohort_retention",
        "fact_discount_bands",
        "fact_discount_thresholds",
    }
    assert expected <= set(tables)
    assert "fact_forecast" not in tables  # no forecast outputs in this fixture run -> skipped
    assert (root / "dashboard" / "data_dictionary.md").exists()
    assert info["tables"]["fact_sales"]["rows"] == len(tables["fact_sales"])


def test_fact_sales_totals_match_transactions(exported):
    _, t, cu_info, _ = exported
    fs = t["fact_sales"]
    assert fs["units"].sum() == cu_info["transactions"]
    assert (fs["full_price_value"] >= fs["indexed_revenue"] - 1e-9).all()
    assert (fs["discounted_units"] <= fs["units"]).all()
    assert not fs.duplicated(["date_key", "article_id", "channel_id", "discount_band"]).any()


def test_every_fact_key_exists_in_its_dimension(exported):
    _, t, _, _ = exported
    fs = t["fact_sales"]
    assert set(fs["date_key"]) <= set(t["dim_date"]["date_key"])
    assert set(fs["article_id"]) <= set(t["dim_article"]["article_id"])
    assert set(fs["channel_id"]) <= set(t["dim_channel"]["channel_id"])
    assert set(fs["discount_band"]) <= set(t["dim_discount_band"]["discount_band"])
    assert set(t["fact_kpi_monthly"]["date_key"]) <= set(t["dim_date"]["date_key"])
    assert set(t["fact_customer"]["segment"]) <= set(t["dim_customer_segment"]["segment"])
    groups = set(t["dim_product_group"]["product_group_name"])
    assert set(t["dim_article"]["product_group_name"].dropna()) <= groups
    for name in (
        "dim_date",
        "dim_article",
        "dim_channel",
        "dim_discount_band",
        "dim_customer_segment",
        "dim_product_group",
    ):
        key = t[name].columns[0] if name != "dim_date" else "date_key"
        assert t[name][key].is_unique, name


def test_fact_customer_has_surrogate_key_and_no_raw_ids(exported):
    _, t, cu_info, _ = exported
    fc = t["fact_customer"]
    assert len(fc) == cu_info["customers"] and fc["customer_key"].is_unique
    assert "customer_id" not in fc.columns
    assert t["dim_discount_band"]["is_ceiling_band"].sum() == 1
