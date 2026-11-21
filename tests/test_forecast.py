"""Tests for the forecasting pieces (fdda.forecast)."""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest

from fdda.forecast import backtest as bt
from fdda.forecast.series import weekly_category_series

BUCKETS = [[1, 4], [5, 8], [9, 12]]


def _series(n_weeks: int = 104, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2018-09-24", periods=n_weeks, freq="W-MON")
    y = 1000 + 300 * np.sin(np.arange(n_weeks) * 2 * np.pi / 52) + rng.normal(0, 20, n_weeks)
    return pd.DataFrame({"ds": ds, "y": y, "promo_share": rng.uniform(0.2, 0.4, n_weeks)})


def test_rolling_origins_cover_last_weeks():
    assert bt.rolling_origins(104, 12, 4, 53) == [56, 68, 80, 92]
    with pytest.raises(ValueError, match="training weeks"):
        bt.rolling_origins(60, 12, 4, 53)


def test_folds_are_non_overlapping_and_end_at_last_week():
    s = _series()
    out = list(bt.folds(s, 12, 4, 53))
    assert [k for k, _, _ in out] == [1, 2, 3, 4]
    tests = pd.concat([te for _, _, te in out])
    assert tests["ds"].is_unique and len(tests) == 48
    assert tests["ds"].max() == s["ds"].max()
    for _, tr, te in out:
        assert tr["ds"].max() < te["ds"].min()  # no leakage


def test_seasonal_naive_uses_same_week_last_year():
    s = _series()
    target = s["ds"].iloc[60:63]
    got = bt.seasonal_naive(s.iloc[:60], target, 52)
    assert np.allclose(got, s["y"].iloc[8:11].to_numpy())
    with pytest.raises(ValueError):
        bt.seasonal_naive(s.iloc[:5], s["ds"].iloc[10:12], 52)


def test_mape_and_wape():
    y, yhat = np.array([100.0, 200.0, 0.0]), np.array([110.0, 150.0, 5.0])
    assert bt.mape(y, yhat) == pytest.approx((0.1 + 0.25) / 2)  # zero actual skipped
    assert bt.wape(y, yhat) == pytest.approx((10 + 50 + 5) / 300)


def test_bucket_label():
    assert [bt.bucket_label(h, BUCKETS) for h in (1, 4, 5, 12)] == ["1-4", "1-4", "5-8", "9-12"]
    with pytest.raises(ValueError):
        bt.bucket_label(13, BUCKETS)


def test_score_adds_overall_bucket_and_all_categories():
    rows = []
    for cat in ["A", "B"]:
        for h in range(1, 13):
            rows.append({"category": cat, "model": "m", "h": h, "y": 100.0, "yhat": 90.0})
    m = bt.score(pd.DataFrame(rows), BUCKETS)
    assert set(m["bucket"].astype(str)) == {"1-4", "5-8", "9-12", "1-12"}
    assert set(m["category"]) == {"A", "B", bt.ALL}
    allrow = m[(m.category == bt.ALL) & (m.bucket == "1-12")].iloc[0]
    assert allrow["wape"] == pytest.approx(0.1) and allrow["n"] == 24


def test_interval_coverage():
    p = pd.DataFrame(
        {
            "model": ["p"] * 4,
            "y": [1, 2, 3, 4],
            "yhat_lower": [0, 0, 5, 0],
            "yhat_upper": [2, 3, 6, 3],
        }
    )
    assert bt.interval_coverage(p).set_index("model").loc["p", "coverage"] == pytest.approx(0.5)


def test_weekly_series_drops_partial_weeks_and_fills_zero():
    start = date(2019, 1, 3)  # Thursday -> its week is partial
    rows = []
    for d in range(0, 40):
        day = start + timedelta(days=d)
        if day >= date(2019, 1, 21) and day < date(2019, 1, 28):
            rows.append(("B", day, 0.0))  # category A has no sales this week
            continue
        rows += [("A", day, 0.0), ("A", day, 0.3), ("B", day, 0.0)]
    df = pd.DataFrame(rows, columns=["product_group_name", "t_dat", "discount"])
    con = duckdb.connect()
    con.register("tx", df)
    s = weekly_category_series(con, "tx", "product_group_name", 2, 0.05)
    weeks = sorted(s["ds"].unique())
    assert pd.Timestamp(weeks[0]) == pd.Timestamp("2019-01-07")  # partial first week dropped
    last_day = start + timedelta(days=39)
    assert pd.Timestamp(weeks[-1]) + pd.Timedelta(days=6) <= pd.Timestamp(last_day)
    a = s[s.category == "A"].set_index("ds")
    assert a.loc["2019-01-21", "y"] == 0  # filled
    assert a.loc["2019-01-07", "promo_share"] == pytest.approx(0.5)


def test_prophet_smoke():
    pytest.importorskip("prophet")
    from fdda.forecast.prophet_model import fit_predict

    s = _series()
    pred = fit_predict(
        s.iloc[:92],
        s.iloc[92:],
        {"yearly_seasonality": 3, "uncertainty_samples": 50},
        use_promo=True,
    )
    assert len(pred) == 12 and (pred["yhat"] >= 0).all()
    assert (pred["yhat_lower"] <= pred["yhat"]).all() and (pred["yhat"] <= pred["yhat_upper"]).all()
    assert bt.wape(s["y"].iloc[92:].to_numpy(), pred["yhat"].to_numpy()) < 0.2
