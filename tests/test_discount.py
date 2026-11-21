"""Tests for reference prices (fdda.features.pricing) and payback logic (fdda.analysis.discount)."""

from __future__ import annotations

import copy
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest

from fdda.analysis import discount as dc
from fdda.config import load_settings
from fdda.features.pricing import build_article_weeks

EDGES = [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.90]
LABELS = ["0-5%", "5-15%", "15-25%", "25-35%", "35-45%", "45-55%", "55%+"]
MONDAY = date(2019, 1, 7)


def _tx_rows(article: str, week: int, price: float, n: int, outlier: bool = False) -> list[dict]:
    day = MONDAY + timedelta(weeks=week, days=2)
    return [
        {
            "article_id": article,
            "product_code": article[:7],
            "product_group_name": "Garment Upper body",
            "product_type_name": "T-shirt",
            "garment_group_name": "Jersey Basic",
            "section_name": "Womens",
            "t_dat": day,
            "price": price,
            "indexed_price": price * 1000,
            "is_price_outlier": outlier,
        }
        for _ in range(n)
    ]


@pytest.fixture
def article_weeks(tmp_path) -> pd.DataFrame:
    """Article x week table built from a hand-made transaction set."""
    rows = []
    # A: 4 full-price weeks (5 sales each at 1.0), then 2 weeks at 0.7 (30% off).
    for w in range(4):
        rows += _tx_rows("A000000001", w, 1.0, 5)
    for w in (4, 5):
        rows += _tx_rows("A000000001", w, 0.7, 8)
    # B: one sale per week -> too few for the rolling window, lifetime max is used.
    for w, p in [(0, 2.0), (1, 2.0), (2, 1.5)]:
        rows += _tx_rows("B000000001", w, p, 1)
    # C: a flagged outlier must not set the reference price.
    rows += _tx_rows("C000000001", 0, 1.0, 6) + _tx_rows("C000000001", 0, 9.0, 1, outlier=True)
    con = duckdb.connect()
    con.register("tx_df", pd.DataFrame(rows))
    out = tmp_path / "aw.parquet"
    summary = build_article_weeks(
        con,
        "tx_df",
        out,
        quantile=0.95,
        window_weeks=8,
        min_obs=5,
        max_discount=0.9,
        price_rescale_factor=1000,
        edges=EDGES,
        labels=LABELS,
    )
    assert summary["article_weeks"] == 6 + 3 + 1
    assert summary["units_in_article_weeks"] == len(rows)
    return pd.read_parquet(out)


def test_reference_price_and_discount_band(article_weeks):
    a = article_weeks[article_weeks.article_id == "A000000001"].sort_values("week_idx")
    assert a["indexed_ref_price"].round(6).tolist() == [1000.0] * 6
    assert a["median_discount"].round(6).tolist() == [0, 0, 0, 0, 0.3, 0.3]
    assert a["band"].tolist() == ["0-5%"] * 4 + ["25-35%"] * 2
    assert not a["used_fallback"].any()


def test_lifetime_fallback_when_window_is_thin(article_weeks):
    b = article_weeks[article_weeks.article_id == "B000000001"].sort_values("week_idx")
    assert b["used_fallback"].all()
    assert b["indexed_ref_price"].round(6).tolist() == [2000.0] * 3
    assert b["median_discount"].round(6).tolist() == [0, 0, 0.25]


def test_outliers_do_not_set_reference(article_weeks):
    c = article_weeks[article_weeks.article_id == "C000000001"]
    assert c["indexed_ref_price"].round(6).item() == 1000.0
    assert c["units"].item() == 7


def test_add_margins_formula():
    aw = pd.DataFrame({"indexed_revenue": [70.0], "indexed_ref_price": [100.0], "units": [1]})
    out = dc.add_margins(aw, [0.5, 0.6])
    # cost = (1 - m) * 100; margin = 70 - cost
    assert out["margin_50"].item() == pytest.approx(20)
    assert out["margin_60"].item() == pytest.approx(30)


def _toy_aw() -> pd.DataFrame:
    """One article: full-price weeks 0-3 sell 10, discounted weeks 4-5 (20% off) sell 15."""
    rows = []
    for w in range(6):
        disc = 0.2 if w >= 4 else 0.0
        units = 15 if w >= 4 else 10
        rows.append(
            {
                "article_id": "X",
                "week_idx": w,
                "product_group_name": "G",
                "units": units,
                "indexed_revenue": units * 100 * (1 - disc),
                "indexed_ref_price": 100.0,
                "median_discount": disc,
                "mean_discount": disc,
                "band": "15-25%" if disc else "0-5%",
            }
        )
    return dc.add_margins(pd.DataFrame(rows), [0.6])


def test_attach_baseline_local_vs_lifetime():
    aw = _toy_aw()
    life = dc.attach_baseline(aw, "0-5%", [0.6], None)
    assert (life["base_units"] == 10).all() and (life["n_base_weeks"] == 4).all()
    local = dc.attach_baseline(aw, "0-5%", [0.6], 1).set_index("week_idx")
    # Week 0's local baseline is week 1 only (itself excluded); week 5 has no full-price
    # neighbour within 1 week, so it is dropped.
    assert local.loc[0, "n_base_weeks"] == 1
    assert 5 not in local.index
    assert local.loc[4, "n_base_weeks"] == 1  # only week 3


def test_band_economics_indexes():
    aw = _toy_aw()
    econ = dc.band_economics(
        dc.attach_baseline(aw, "0-5%", [0.6], None), LABELS, "product_group_name", [0.6], 1, 1
    )
    row = econ[(econ.category == dc.ALL) & (econ.band == "15-25%")].iloc[0]
    assert row["units_index"] == pytest.approx(1.5)
    assert row["revenue_index"] == pytest.approx(1.5 * 0.8)
    # margin per unit: full price 60, at 20% off 80 - 40 = 40 -> 15*40 / (10*60) = 1.0
    assert row["margin_index_60"] == pytest.approx(1.0)
    assert row["band_discount"] == pytest.approx(0.2)


def _econ(values, discounts):
    return pd.DataFrame(
        {
            "category": pd.Categorical([dc.ALL] * len(values)),
            "band": LABELS[: len(values)],
            "band_discount": discounts,
            "margin_index_60": values,
        }
    )


def test_payback_threshold_finds_turn():
    th = dc.payback_threshold(
        _econ([1, 1.1, 1.15, 1.05, 0.9], [0, 0.1, 0.2, 0.3, 0.4]), "margin_index_60"
    )
    row = th.iloc[0]
    assert row["ceiling_band"] == "15-25%" and row["ceiling_discount"] == pytest.approx(0.2)
    assert row["first_negative_band"] == "25-35%" and row["turns_negative"]
    assert row["peak_band"] == "15-25%"


def test_payback_threshold_immediate_and_never():
    first = dc.payback_threshold(_econ([1, 0.8, 0.6], [0, 0.1, 0.2]), "margin_index_60").iloc[0]
    assert first["ceiling_band"] == "0-5%"
    never = dc.payback_threshold(_econ([1, 1.2, 1.3], [0, 0.1, 0.2]), "margin_index_60").iloc[0]
    assert never["ceiling_band"] == "15-25%" and not never["turns_negative"]


def test_fe_elasticity_recovers_known_beta():
    rng = np.random.default_rng(0)
    n_art, n_wk, beta = 300, 30, -2.0
    art = np.repeat(np.arange(n_art), n_wk)
    wk = np.tile(np.arange(n_wk), n_art)
    disc = rng.choice([0, 0, 0.1, 0.2, 0.3, 0.5], size=art.size)
    log_units = (
        rng.normal(2, 1, n_art)[art]
        + rng.normal(0, 0.5, n_wk)[wk]
        + beta * np.log1p(-disc)
        + rng.normal(0, 0.2, art.size)
    )
    df = pd.DataFrame(
        {"article_id": art, "week_idx": wk, "units": np.exp(log_units), "mean_discount": disc}
    )
    res = dc.fe_elasticity(df)
    assert res["beta"] == pytest.approx(beta, abs=0.05)
    assert res["ci_low"] < beta < res["ci_high"]
    assert res["n_articles"] == n_art and res["n_weeks"] == n_wk


def test_implied_optimal_discount():
    assert dc.implied_optimal_discount(-2.0, 0.6) == pytest.approx(0.2)
    assert dc.implied_optimal_discount(-1.0, 0.6) == 0.0
    assert dc.implied_optimal_discount(float("nan"), 0.6) == 0.0


def test_add_age_bucket():
    aw = pd.DataFrame({"article_id": ["a"] * 4, "week_idx": [10, 13, 18, 70]})
    out = dc.add_age_bucket(aw, [0, 4, 8, 13, 26, 52])
    assert out["age_bucket"].tolist() == [0, 0, 2, 5]


def test_run_end_to_end_on_fixture(hm_full, myntra_full, tmp_path):
    settings = copy.deepcopy(load_settings())
    settings["_root"] = tmp_path
    settings["paths"].update({"interim": "interim", "processed": "processed", "reports": "reports"})
    settings["duckdb"] = {}
    settings["discount_analysis"].update(
        {"min_articles_per_category": 5, "min_articles_per_band": 1}
    )
    summary = dc.run(settings, "full")
    out = tmp_path / "reports" / "discount"
    for name in ["band_economics", "thresholds", "elasticity", "myntra_crosscheck"]:
        assert (out / f"{name}_full.csv").exists()
    assert (out / "discount_summary_full.json").exists()
    assert summary["vs_full_price"]["overall"]["ceiling_band"] in LABELS
    th = pd.read_csv(out / "thresholds_full.csv")
    assert set(th["reference"]) == {"full price", "from 5-15%"}
    econ = pd.read_csv(out / "band_economics_full.csv")
    assert set(econ["baseline"]) == {"local", "lifetime"}
    life_fp = econ[
        (econ.baseline == "lifetime") & (econ.category == dc.ALL) & (econ.band == "0-5%")
    ]
    assert life_fp["units_index"].item() == pytest.approx(1.0)
    my = pd.read_csv(out / "myntra_crosscheck_full.csv")
    base = my[(my.category == dc.ALL) & (my.band == "0-5%")]
    assert base["mean_ratings_index"].item() == pytest.approx(1.0)


def test_payback_threshold_among_discounts():
    # Below full price everywhere, but 15-25% beats 5-15% before falling again.
    econ = _econ([1, 0.74, 0.80, 0.51, 0.45], [0, 0.1, 0.2, 0.3, 0.4])
    vs_full = dc.payback_threshold(econ, "margin_index_60").iloc[0]
    assert vs_full["ceiling_band"] == "0-5%" and vs_full["reference"] == "full price"
    among = dc.payback_threshold(econ, "margin_index_60", start_band="5-15%").iloc[0]
    assert among["ceiling_band"] == "15-25%" and among["reference"] == "from 5-15%"
    assert among["ceiling_index"] == pytest.approx(0.80)
    assert among["first_negative_band"] == "25-35%"
