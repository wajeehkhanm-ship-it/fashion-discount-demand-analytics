"""Rolling-origin backtest and forward forecast (PROJECT_PLAN 4.6).

Models:
* seasonal_naive: the same week last year (y at t - 52 weeks).
* prophet: Prophet with yearly seasonality.
* prophet_promo: Prophet + promo-intensity regressor. In the backtest it is given the
  realised promo share for the test weeks, i.e. it assumes the promo calendar is planned in
  advance. The forward forecast uses the same week last year's promo share as the plan.

Folds: `n_folds` non-overlapping origins covering the last n_folds x horizon weeks. Metrics:
MAPE (mean of |error| / actual) and WAPE (sum |error| / sum actual) by horizon bucket.

Usage: `python -m fdda.forecast.backtest [--dev | --full]`
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fdda.config import data_mode, load_settings, resolve_path
from fdda.data.clean import connect
from fdda.forecast.prophet_model import fit_predict
from fdda.forecast.series import weekly_category_series
from fdda.log import get_logger

log = get_logger(__name__)
ALL = "All categories"
MODELS = ["seasonal_naive", "prophet", "prophet_promo"]


def _q(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


# ------------------------------------------------------------------ pieces


def rolling_origins(n_weeks: int, horizon: int, n_folds: int, min_train: int) -> list[int]:
    """Origin indexes (first test week) for non-overlapping folds ending at the last week."""
    origins = [n_weeks - horizon * (n_folds - k) for k in range(n_folds)]
    if origins[0] < min_train:
        raise ValueError(f"first fold has {origins[0]} training weeks; need at least {min_train}")
    return origins


def folds(series: pd.DataFrame, horizon: int, n_folds: int, min_train: int) -> Iterator[tuple]:
    """Yield (fold, train, test) for one category's series sorted by ds."""
    s = series.sort_values("ds").reset_index(drop=True)
    for k, origin in enumerate(rolling_origins(len(s), horizon, n_folds, min_train), 1):
        yield k, s.iloc[:origin], s.iloc[origin : origin + horizon]


def seasonal_naive(history: pd.DataFrame, target_ds: pd.Series, season: int) -> np.ndarray:
    """Forecast each target week with the value `season` weeks earlier from `history`."""
    lookup = history.set_index("ds")["y"]
    lagged = pd.to_datetime(target_ds) - pd.Timedelta(weeks=season)
    values = lookup.reindex(lagged).to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("history does not reach back one season for every target week")
    return values


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute percentage error over points with y > 0 (as a fraction)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    mask = y > 0
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / y[mask])) if mask.any() else np.nan


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted absolute percentage error: sum |error| / sum actual (as a fraction)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / y.sum()) if y.sum() > 0 else np.nan


def bucket_label(h: int, buckets: list[list[int]]) -> str:
    """Horizon bucket name for step h, e.g. 6 -> '5-8'."""
    for lo, hi in buckets:
        if lo <= h <= hi:
            return f"{lo}-{hi}"
    raise ValueError(f"horizon {h} not in any bucket")


def score(preds: pd.DataFrame, buckets: list[list[int]]) -> pd.DataFrame:
    """MAPE/WAPE per category, model and horizon bucket (+ '1-H' overall, + all categories)."""
    p = preds.copy()
    p["bucket"] = [bucket_label(h, buckets) for h in p["h"]]
    whole = f"{buckets[0][0]}-{buckets[-1][1]}"
    p = pd.concat([p, p.assign(bucket=whole)], ignore_index=True)
    p = pd.concat([p, p.assign(category=ALL)], ignore_index=True)
    rows = []
    for (cat, model, bucket), g in p.groupby(["category", "model", "bucket"], observed=True):
        rows.append(
            {
                "category": cat,
                "model": model,
                "bucket": bucket,
                "mape": mape(g["y"], g["yhat"]),
                "wape": wape(g["y"], g["yhat"]),
                "n": len(g),
            }
        )
    out = pd.DataFrame(rows)
    out["bucket"] = pd.Categorical(
        out["bucket"], categories=[f"{lo}-{hi}" for lo, hi in buckets] + [whole], ordered=True
    )
    return out.sort_values(["category", "model", "bucket"]).reset_index(drop=True)


def interval_coverage(preds: pd.DataFrame) -> pd.DataFrame:
    """Share of actuals inside the prediction interval, per Prophet model."""
    p = preds.dropna(subset=["yhat_lower"])
    p = p.assign(inside=(p["y"] >= p["yhat_lower"]) & (p["y"] <= p["yhat_upper"]))
    return p.groupby("model")["inside"].mean().rename("coverage").reset_index()


# ------------------------------------------------------------------ driver


def run_backtest(series: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Predictions for every category, fold, model and test week."""
    h, k, season = cfg["horizon_weeks"], cfg["n_folds"], cfg["season_length_weeks"]
    rows = []
    for cat, s in series.groupby("category", observed=True):
        for fold, train, test in folds(s, h, k, min_train=season + 1):
            origin = test["ds"].iloc[0]
            base = test[["ds", "y"]].assign(
                category=cat, fold=fold, origin=origin, h=np.arange(1, len(test) + 1)
            )
            rows.append(
                base.assign(model="seasonal_naive", yhat=seasonal_naive(train, test["ds"], season))
            )
            for model, promo in [("prophet", False), ("prophet_promo", True)]:
                pred = fit_predict(train, test, cfg["prophet"], use_promo=promo)
                rows.append(
                    base.assign(
                        model=model,
                        **{c: pred[c].to_numpy() for c in ["yhat", "yhat_lower", "yhat_upper"]},
                    )
                )
        log.info("backtest done: %s", cat)
    return pd.concat(rows, ignore_index=True)


def forward_forecast(series: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """12-week-ahead forecasts from all data, for both Prophet specs and the naive baseline.

    Future promo share = the same week last year (a planned-calendar proxy).
    """
    h, season = cfg["horizon_weeks"], cfg["season_length_weeks"]
    rows = []
    for cat, s in series.groupby("category", observed=True):
        s = s.sort_values("ds")
        last = s["ds"].max()
        fut = pd.DataFrame(
            {"ds": pd.date_range(last + pd.Timedelta(weeks=1), periods=h, freq="W-MON")}
        )
        lag = s.set_index("ds")["promo_share"]
        fut["promo_share"] = lag.reindex(fut["ds"] - pd.Timedelta(weeks=season)).to_numpy()
        fut["promo_share"] = fut["promo_share"].fillna(s["promo_share"].tail(season).mean())
        base = fut.assign(category=cat, h=np.arange(1, h + 1))
        rows.append(base.assign(model="seasonal_naive", yhat=seasonal_naive(s, fut["ds"], season)))
        for model, promo in [("prophet", False), ("prophet_promo", True)]:
            pred = fit_predict(s, fut, cfg["prophet"], use_promo=promo)
            rows.append(
                base.assign(
                    model=model,
                    **{c: pred[c].to_numpy() for c in ["yhat", "yhat_lower", "yhat_upper"]},
                )
            )
    return pd.concat(rows, ignore_index=True)


def build_summary(metrics: pd.DataFrame, coverage: pd.DataFrame, cfg: dict, info: dict) -> dict:
    """Headline numbers: MAPE/WAPE per model, best Prophet spec, categories beating naive."""
    whole = f"{cfg['horizon_buckets'][0][0]}-{cfg['horizon_buckets'][-1][1]}"
    m = metrics[metrics["bucket"].astype(str) == whole]
    overall = m[m["category"] == ALL].set_index("model")
    best = min(["prophet", "prophet_promo"], key=lambda x: overall.loc[x, "wape"])
    per_cat = m[m["category"] != ALL].pivot(index="category", columns="model", values="wape")
    beats = per_cat.index[per_cat[best] < per_cat["seasonal_naive"]].tolist()
    return {
        **info,
        "horizon_bucket_overall": whole,
        "overall": overall[["mape", "wape"]].to_dict(orient="index"),
        "best_prophet_model": best,
        "best_beats_naive_overall_wape": bool(
            overall.loc[best, "wape"] < overall.loc["seasonal_naive", "wape"]
        ),
        "best_beats_naive_overall_mape": bool(
            overall.loc[best, "mape"] < overall.loc["seasonal_naive", "mape"]
        ),
        "categories_where_best_beats_naive_wape": beats,
        "n_categories": int(per_cat.shape[0]),
        "interval_coverage": coverage.set_index("model")["coverage"].to_dict(),
    }


def run(settings: dict[str, Any], mode: str) -> dict[str, Any]:
    """Build series, backtest, score, forecast forward, and write reports/forecast/."""
    cfg = settings["forecast"]
    t0 = time.perf_counter()
    processed = resolve_path(settings, "processed") / mode
    txd_path = processed / "transactions_discount.parquet"
    if not txd_path.exists():
        raise FileNotFoundError(f"{txd_path} missing; run `make customers` (Phase 4) first.")
    out = resolve_path(settings, "reports") / "forecast"
    out.mkdir(parents=True, exist_ok=True)
    con = connect(settings)
    series = weekly_category_series(
        con,
        f"read_parquet({_q(txd_path)})",
        cfg["category_level"],
        cfg["top_n_categories"],
        cfg["promo_threshold"],
    )
    log.info(
        "series: %d categories x %d weeks", series["category"].nunique(), series["ds"].nunique()
    )
    preds = run_backtest(series, cfg)
    metrics = score(preds, cfg["horizon_buckets"])
    coverage = interval_coverage(preds)
    forecast = forward_forecast(series, cfg)
    info = {
        "mode": mode,
        "categories": [str(c) for c in series["category"].cat.categories],
        "weeks": int(series["ds"].nunique()),
        "first_week": str(series["ds"].min().date()),
        "last_week": str(series["ds"].max().date()),
        "test_weeks": f"{preds['ds'].min().date()} to {preds['ds'].max().date()}",
        "horizon_weeks": cfg["horizon_weeks"],
        "n_folds": cfg["n_folds"],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    summary = build_summary(metrics, coverage, cfg, info)
    for name, df in {
        "weekly_series": series,
        "backtest_predictions": preds,
        "backtest_metrics": metrics,
        "interval_coverage": coverage,
        "forward_forecast": forecast,
    }.items():
        df.to_csv(out / f"{name}_{mode}.csv", index=False)
    (out / f"forecast_summary_{mode}.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    log.info("Wrote forecast outputs to %s (%.0fs)", out, info["seconds"])
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Backtest and forecast weekly category demand")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dev", dest="mode", action="store_const", const="dev")
    mode.add_argument("--full", dest="mode", action="store_const", const="full")
    args = parser.parse_args(argv)
    settings = load_settings()
    run(settings, args.mode or data_mode(settings))


if __name__ == "__main__":
    main()
