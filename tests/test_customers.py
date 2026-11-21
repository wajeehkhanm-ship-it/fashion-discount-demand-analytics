"""Tests for RFM (fdda.features.rfm), KPIs, cohorts and the win-back playbook."""

from __future__ import annotations

import copy
from datetime import date, timedelta

import duckdb
import pandas as pd
import pytest

from fdda.analysis import cohorts, kpis
from fdda.analysis import customers as cu
from fdda.analysis import discount as dc
from fdda.config import load_settings
from fdda.features.rfm import build_customer_rfm, check_rule_coverage, score_sql

RULES = load_settings()["customers"]["rfm_segments"]


def _tx(rows: list[tuple]) -> pd.DataFrame:
    """rows: (customer, date, channel, indexed_price, discount)."""
    return pd.DataFrame(
        rows, columns=["customer_id", "t_dat", "sales_channel_id", "indexed_price", "discount"]
    )


def test_score_sql_ties_share_a_score():
    con = duckdb.connect()
    con.register("v", pd.DataFrame({"x": [1, 1, 1, 1, 2, 3, 4, 5, 6, 7]}))
    got = con.execute(f"SELECT x, {score_sql('x')} AS s FROM v ORDER BY x").df()
    assert got.loc[got.x == 1, "s"].nunique() == 1 and got.loc[got.x == 1, "s"].iloc[0] == 1
    assert got["s"].between(1, 5).all() and got["s"].max() == 5
    low = con.execute(
        f"SELECT x, {score_sql('x', higher_is_better=False)} AS s FROM v ORDER BY x"
    ).df()
    assert low.loc[low.x == 7, "s"].item() == 1  # largest value is worst when lower is better


def test_config_rules_cover_every_cell():
    assert check_rule_coverage(RULES) == []
    assert (1, 1) in check_rule_coverage([{"segment": "x", "r": [2, 5], "f": [1, 5]}])


def test_build_customer_rfm_values(tmp_path):
    d0 = date(2020, 1, 1)
    rows = [
        ("a", d0, 1, 10.0, 0.0),
        ("a", d0, 2, 5.0, 0.0),  # same day, 2 channels
        ("a", d0 + timedelta(days=100), 2, 20.0, 0.0),
        ("b", d0 + timedelta(days=10), 1, 7.0, 0.0),
        ("c", d0 + timedelta(days=200), 2, 3.0, 0.0),  # last day in data
    ]
    con = duckdb.connect()
    con.register("tx", _tx(rows))
    info = build_customer_rfm(con, "tx", tmp_path / "rfm.parquet", RULES, [90, 120, 180])
    assert info["snapshot_date"] == str(d0 + timedelta(days=201))
    df = pd.read_parquet(tmp_path / "rfm.parquet").set_index("customer_id")
    assert df.loc["a", "frequency"] == 2 and df.loc["a", "orders"] == 3
    assert df.loc["a", "monetary"] == pytest.approx(35)
    assert df.loc["a", "recency_days"] == 101 and df.loc["c", "recency_days"] == 1
    assert df.loc["a", "churned_90d"] and not df.loc["a", "churned_120d"]
    assert df.loc["b", "churned_180d"]
    # a's year before its last purchase includes all of its spend
    assert df.loc["a", "revenue_last_year"] == pytest.approx(35)
    assert (df["segment"] != "Unassigned").all()


def test_monthly_kpis_orders_aov_and_partial_months():
    rows = [
        ("a", date(2020, 1, 15), 1, 10.0, 0.0),
        ("a", date(2020, 1, 15), 1, 10.0, 0.3),
        ("a", date(2020, 1, 15), 2, 10.0, 0.0),
        ("b", date(2020, 1, 31), 1, 30.0, 0.0),
        ("b", date(2020, 2, 1), 1, 40.0, 0.5),
        ("c", date(2020, 2, 29), 2, 20.0, 0.0),
    ]
    con = duckdb.connect()
    con.register("tx", _tx(rows))
    m = kpis.monthly_kpis(con, "tx", 0.05).set_index("month")
    jan, feb = m.loc["2020-01"], m.loc["2020-02"]
    assert jan["orders"] == 3 and jan["units"] == 4 and jan["customers"] == 2
    assert jan["aov"] == pytest.approx(60 / 3)
    assert jan["discounted_revenue_share"] == pytest.approx(10 / 60)
    assert jan["is_partial_month"] and not feb["is_partial_month"]
    assert feb["revenue_mom"] == pytest.approx(60 / 60 - 1)
    tot = kpis.kpi_totals(con, "tx", 0.05)
    assert tot["orders"] == 5 and tot["customers"] == 3


def test_retention_matrix_and_curve():
    rows = [
        ("a", date(2020, 1, 5), 1, 1.0, 0.0),
        ("a", date(2020, 2, 5), 1, 1.0, 0.0),
        ("b", date(2020, 1, 9), 1, 1.0, 0.0),
        ("b", date(2020, 3, 9), 1, 1.0, 0.0),
        ("c", date(2020, 2, 2), 1, 1.0, 0.0),
        ("c", date(2020, 3, 2), 1, 1.0, 0.0),
    ]
    con = duckdb.connect()
    con.register("tx", _tx(rows))
    long = cohorts.retention_matrix(con, "tx")
    wide = cohorts.retention_wide(long)
    assert wide.loc["2020-01", 0] == 1 and wide.loc["2020-01", 1] == 0.5
    assert wide.loc["2020-01", 2] == 0.5 and wide.loc["2020-02", 1] == 1
    curve = cohorts.average_retention_curve(long).set_index("months_since")
    # k=1: both cohorts observable -> (1 + 1) / 3; k=2: only Jan cohort -> 1 / 2
    assert curve.loc[1, "retention"] == pytest.approx(2 / 3)
    assert curve.loc[2, "retention"] == pytest.approx(1 / 2)


def test_round_down_and_offer_cap():
    assert cu.round_down_to(0.199) == pytest.approx(0.15)
    assert cu.round_down_to(0.20) == pytest.approx(0.20)
    labels = ["0-5%", "5-15%", "15-25%"]
    summary = pd.DataFrame(
        {
            "segment": ["At Risk"],
            "customers": [10],
            "customer_share": [0.1],
            "churned_share": [1.0],
            "median_recency_days": [300],
            "revenue_at_stake": [100.0],
            "revenue_at_stake_per_customer": [10.0],
            "channel2_share": [0.7],
        }
    )
    prefs = pd.DataFrame(
        {"segment": ["At Risk"], "category": ["Tops"], "unit_share": [0.5], "rank": [1]}
    )
    profile = pd.DataFrame(
        {
            "segment": ["At Risk"] * 3,
            "band": labels,
            "lift_vs_all": [0.8, 1.5, 1.1],
            "discounted_unit_share": [0.4] * 3,
            "median_discount_when_discounted": [0.33] * 3,
        }
    )
    cfg = [{"segment": "At Risk", "action": "a", "channel": "c", "timing": "t", "kpi": "k"}]
    pb = cu.build_playbook(summary, prefs, profile, cfg, {"ceiling_discount": 0.2}, labels).iloc[0]
    assert pb["offer_discount"] == pytest.approx(0.20) and pb["offer_capped"]
    assert pb["responds_to_band"] == "5-15%"
    assert pb["favourite_categories"] == "Tops (50%)"


def test_run_end_to_end_on_fixture(hm_full, myntra_full, tmp_path):
    settings = copy.deepcopy(load_settings())
    settings["_root"] = tmp_path
    settings["paths"].update({"interim": "interim", "processed": "processed", "reports": "reports"})
    settings["duckdb"] = {}
    settings["discount_analysis"].update(
        {"min_articles_per_category": 5, "min_articles_per_band": 1}
    )
    with pytest.raises(FileNotFoundError, match="make discount"):
        cu.run(settings, "full")
    dc.run(settings, "full")
    info = cu.run(settings, "full")
    out = tmp_path / "reports" / "customers"
    assert (tmp_path / "reports" / "winback_playbook.md").exists()
    assert info["unassigned"] == 0
    seg = pd.read_csv(out / "segment_summary_full.csv")
    assert seg["customers"].sum() == info["customers"]
    assert seg["customer_share"].sum() == pytest.approx(1)
    m = pd.read_csv(out / "monthly_kpis_full.csv")
    assert m["units"].sum() == info["transactions"]
    pb = pd.read_csv(out / "winback_playbook_full.csv")
    assert (pb["offer_discount"] <= info["offer_cap"] + 1e-9).all()
