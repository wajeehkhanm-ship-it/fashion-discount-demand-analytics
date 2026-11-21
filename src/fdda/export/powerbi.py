"""Power BI star schema export (PROJECT_PLAN 4.8).

Writes model-ready tables to `data/processed/powerbi/` (Parquet by default, CSV with --csv)
plus `dashboard/data_dictionary.md` describing every table and column.

Dimensions
* dim_date            one row per day (weekly facts use the week's Monday, monthly facts the 1st)
* dim_article         H&M article attributes
* dim_product_group   product groups (links articles and forecasts)
* dim_channel         sales channels
* dim_discount_band   discount bands, with the Phase 3 markdown-ceiling band flagged
* dim_customer_segment RFM segments with rules and the win-back playbook

Facts
* fact_sales              week x article x channel x discount band (units, revenue,
                          full-price value)
* fact_kpi_monthly        monthly orders, customers, AOV (distinct counts are not additive)
* fact_customer           one row per customer (RFM scores, segment, churn flags), surrogate key
* fact_cohort_retention   acquisition cohort x months since first purchase
* fact_forecast           weekly actuals, backtest forecasts and the 12-week forward forecast
* fact_forecast_metrics   backtest MAPE / WAPE by category, model and horizon bucket
* fact_discount_bands     Phase 3 band indexes (units, revenue, margin at 50/60/70%)
* fact_discount_thresholds Phase 3 payback ceilings

Usage: `python -m fdda.export.powerbi [--dev | --full] [--csv]`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from fdda.analysis import eda
from fdda.config import data_mode, load_settings, resolve_path
from fdda.data.clean import connect
from fdda.features.pricing import band_case_sql
from fdda.log import get_logger

log = get_logger(__name__)

CHANNELS = {1: "Channel 1", 2: "Channel 2"}
DESCRIPTIONS = {
    "dim_date": (
        "Calendar, one row per day. Weekly facts use the week's Monday; monthly facts the 1st."
    ),
    "dim_article": "H&M article attributes (one row per article_id).",
    "dim_product_group": "Product groups; links dim_article and fact_forecast.",
    "dim_channel": "Sales channels. H&M does not name them; channel 2 is commonly read as online.",
    "dim_discount_band": "Discount bands; is_ceiling_band marks the Phase 3 markdown ceiling.",
    "dim_customer_segment": "RFM segments with their R/F rules, size and win-back playbook.",
    "fact_sales": "Units and indexed revenue by week x article x channel x discount band.",
    "fact_kpi_monthly": "Monthly orders, active customers and AOV (not additive across months).",
    "fact_customer": "One row per customer: RFM, segment, churn flags (customer_key is surrogate).",
    "fact_cohort_retention": "Share of each acquisition cohort buying k months later.",
    "fact_forecast": "Weekly units: actuals, backtest forecasts and the 12-week forward forecast.",
    "fact_forecast_metrics": "Backtest MAPE and WAPE by category, model and horizon bucket.",
    "fact_discount_bands": "Phase 3 within-article indexes by category and discount band.",
    "fact_discount_thresholds": "Phase 3 payback ceilings by category, measure and reference.",
}


def _q(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def date_key(series: pd.Series) -> pd.Series:
    """yyyymmdd integer key from dates."""
    return pd.to_datetime(series).dt.strftime("%Y%m%d").astype("int64")


# ------------------------------------------------------------------ dimensions


def build_dim_date(start: str, end: str, data_first: str, data_last: str) -> pd.DataFrame:
    """Daily calendar from `start` to `end` with week, month and data-coverage flags."""
    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    d["date_key"] = date_key(d["date"])
    d["year"] = d["date"].dt.year
    d["quarter"] = "Q" + d["date"].dt.quarter.astype(str)
    d["month_num"] = d["date"].dt.month
    d["month_name"] = d["date"].dt.strftime("%b")
    d["year_month"] = d["date"].dt.strftime("%Y-%m")
    d["month_start"] = d["date"].dt.to_period("M").dt.start_time
    d["week_start"] = d["date"] - pd.to_timedelta(d["date"].dt.day_of_week, unit="D")
    d["iso_week"] = d["date"].dt.isocalendar().week.astype(int)
    d["is_week_start"] = d["date"].dt.day_of_week == 0
    d["in_sales_data"] = (d["date"] >= data_first) & (d["date"] <= data_last)
    d["is_forecast_period"] = d["date"] > pd.Timestamp(data_last)
    return d


def build_dim_discount_band(
    edges: list[float], labels: list[str], ceiling: str | None
) -> pd.DataFrame:
    """Discount bands with sort order, edges and the markdown-ceiling flag."""
    return pd.DataFrame(
        {
            "discount_band": labels,
            "band_order": range(1, len(labels) + 1),
            "lower_edge": edges[:-1],
            "upper_edge": edges[1:],
            "is_full_price": [i == 0 for i in range(len(labels))],
            "is_ceiling_band": [b == ceiling for b in labels],
        }
    )


def build_dim_segment(
    rules: list[dict[str, Any]], summary: pd.DataFrame, playbook: pd.DataFrame
) -> pd.DataFrame:
    """Segment dimension: rule text, size metrics and playbook actions."""
    seg = pd.DataFrame(
        {
            "segment": [r["segment"] for r in rules],
            "segment_order": range(1, len(rules) + 1),
            "r_score_range": [f"{r['r'][0]}-{r['r'][1]}" for r in rules],
            "f_score_range": [f"{r['f'][0]}-{r['f'][1]}" for r in rules],
        }
    )
    keep = [
        "segment",
        "customers",
        "customer_share",
        "median_recency_days",
        "avg_frequency",
        "avg_monetary",
        "monetary_share",
        "revenue_at_stake",
        "churned_share",
    ]
    seg = seg.merge(summary[keep], on="segment", how="left")
    pb_cols = [
        "segment",
        "offer_discount",
        "responds_to_band",
        "favourite_categories",
        "action",
        "channel",
        "timing",
        "kpi",
    ]
    pb = (
        playbook[[c for c in pb_cols if c in playbook]]
        if len(playbook)
        else pd.DataFrame(columns=pb_cols)
    )
    seg = seg.merge(pb, on="segment", how="left")
    seg["is_winback_segment"] = seg["segment"].isin(pb["segment"])
    return seg


# ------------------------------------------------------------------ export


def export(settings: dict[str, Any], mode: str, fmt: str = "parquet") -> dict[str, Any]:
    """Build and write every table; returns row counts and file sizes."""
    t0 = time.perf_counter()
    p = settings["pricing"]
    edges, labels = p["discount_bands"]["edges"], p["discount_bands"]["labels"]
    thr = settings["customers"]["discounted_threshold"]
    root = Path(settings["_root"])
    interim = resolve_path(settings, "interim") / mode
    processed = resolve_path(settings, "processed") / mode
    reports = resolve_path(settings, "reports")
    out = resolve_path(settings, "processed") / "powerbi"
    out.mkdir(parents=True, exist_ok=True)
    needed = {
        "transactions_discount": processed / "transactions_discount.parquet",
        "article_weeks": processed / "article_weeks.parquet",
        "customer_rfm": processed / "customer_rfm.parquet",
    }
    missing = [k for k, v in needed.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} for '{mode}'; run `make analysis` first.")
    con = connect(settings)
    tables: dict[str, pd.DataFrame] = {}

    # fact_sales: per-unit discount band; full-price value = the article-week reference price,
    # or the price paid when that is higher (consistent with discounts floored at 0%).
    band = band_case_sql("t.discount", edges, labels)
    fact_sales = con.execute(f"""
        SELECT CAST(strftime(date_trunc('week', t.t_dat), '%Y%m%d') AS BIGINT) AS date_key,
               t.article_id,
               t.sales_channel_id AS channel_id,
               {band} AS discount_band,
               count(*) AS units,
               sum(t.indexed_price) AS indexed_revenue,
               sum(greatest(a.indexed_ref_price, t.indexed_price)) AS full_price_value,
               count(*) FILTER (WHERE t.discount >= {float(thr)}) AS discounted_units,
               sum(t.indexed_price) FILTER (WHERE t.discount >= {float(thr)}) AS discounted_revenue
        FROM read_parquet({_q(needed["transactions_discount"])}) t
        JOIN read_parquet({_q(needed["article_weeks"])}) a
          ON t.article_id = a.article_id AND date_trunc('week', t.t_dat)::DATE = a.week
        GROUP BY 1, 2, 3, 4
    """).df()
    fact_sales["discounted_revenue"] = fact_sales["discounted_revenue"].fillna(0.0)
    tables["fact_sales"] = fact_sales

    first, last = con.execute(
        f"SELECT min(t_dat), max(t_dat) FROM read_parquet({_q(needed['transactions_discount'])})"
    ).fetchone()
    articles = con.execute(f"""
        SELECT article_id, product_code, prod_name, product_type_name, product_group_name,
               garment_group_name, section_name, colour_group_name, graphical_appearance_name
        FROM {eda.hm_articles(interim)}
    """).df()
    tables["dim_article"] = articles
    group_units = fact_sales.merge(articles[["article_id", "product_group_name"]], on="article_id")
    groups = group_units.groupby("product_group_name")["units"].sum().sort_values(ascending=False)
    fc_dir = reports / "forecast"
    fc_series = fc_dir / f"weekly_series_{mode}.csv"
    forecast_cats = (
        set(pd.read_csv(fc_series)["category"].unique()) if fc_series.exists() else set()
    )
    all_groups = sorted(set(articles["product_group_name"].dropna()))
    tables["dim_product_group"] = pd.DataFrame(
        {
            "product_group_name": all_groups,
            "units_rank": [
                int(groups.index.get_loc(g)) + 1 if g in groups.index else None for g in all_groups
            ],
            "is_forecast_category": [g in forecast_cats for g in all_groups],
        }
    )
    tables["dim_channel"] = pd.DataFrame(
        {"channel_id": list(CHANNELS), "channel_name": list(CHANNELS.values())}
    )
    disc_summary = reports / "discount" / f"discount_summary_{mode}.json"
    ceiling = (
        json.loads(disc_summary.read_text())["among_discounts"]["overall"]["ceiling_band"]
        if disc_summary.exists()
        else None
    )
    tables["dim_discount_band"] = build_dim_discount_band(edges, labels, ceiling)

    cust_dir = reports / "customers"
    seg_summary = pd.read_csv(cust_dir / f"segment_summary_{mode}.csv")
    pb_path = cust_dir / f"winback_playbook_{mode}.csv"
    playbook = pd.read_csv(pb_path) if pb_path.exists() else pd.DataFrame()
    tables["dim_customer_segment"] = build_dim_segment(
        settings["customers"]["rfm_segments"], seg_summary, playbook
    )

    kpi = pd.read_csv(cust_dir / f"monthly_kpis_{mode}.csv")
    kpi.insert(0, "date_key", date_key(kpi["month"] + "-01"))
    tables["fact_kpi_monthly"] = kpi.rename(columns={"revenue": "indexed_revenue"})

    churn_cols = ", ".join(
        f"churned_{int(w)}d"
        for w in sorted(
            set(settings["customers"]["churn_sensitivity_days"])
            | {settings["customers"]["churn_window_days"]}
        )
    )
    tables["fact_customer"] = con.execute(f"""
        SELECT row_number() OVER (ORDER BY customer_id) AS customer_key,
               segment, r_score, f_score, m_score, recency_days, frequency, orders, units,
               monetary AS indexed_revenue, revenue_last_year AS revenue_at_stake,
               channel2_share, strftime(first_date, '%Y-%m') AS first_purchase_month,
               CAST(strftime(last_date, '%Y%m%d') AS BIGINT) AS last_purchase_date_key,
               {churn_cols}
        FROM read_parquet({_q(needed["customer_rfm"])})
    """).df()
    tables["fact_cohort_retention"] = pd.read_csv(cust_dir / f"cohort_retention_{mode}.csv")

    if fc_series.exists():
        series = pd.read_csv(fc_series, parse_dates=["ds"])
        preds = pd.read_csv(fc_dir / f"backtest_predictions_{mode}.csv", parse_dates=["ds"])
        fwd = pd.read_csv(fc_dir / f"forward_forecast_{mode}.csv", parse_dates=["ds"])
        actual = series.assign(kind="actual", model="actual", units=series["y"])
        back = preds.assign(kind="backtest", units=preds["y"])
        ahead = fwd.assign(kind="forward", units=pd.NA)
        cols = [
            "ds",
            "category",
            "kind",
            "model",
            "units",
            "yhat",
            "yhat_lower",
            "yhat_upper",
            "fold",
            "h",
            "promo_share",
        ]
        fc = pd.concat([d.reindex(columns=cols) for d in (actual, back, ahead)], ignore_index=True)
        fc.insert(0, "date_key", date_key(fc["ds"]))
        tables["fact_forecast"] = fc.drop(columns="ds")
        tables["fact_forecast_metrics"] = pd.read_csv(fc_dir / f"backtest_metrics_{mode}.csv")
        last_day = max(pd.Timestamp(last), fwd["ds"].max() + pd.Timedelta(days=6))
    else:
        log.warning("No forecast outputs for '%s'; skipping forecast tables.", mode)
        last_day = pd.Timestamp(last)

    disc_dir = reports / "discount"
    if (disc_dir / f"band_economics_{mode}.csv").exists():
        tables["fact_discount_bands"] = pd.read_csv(disc_dir / f"band_economics_{mode}.csv")
        tables["fact_discount_thresholds"] = pd.read_csv(disc_dir / f"thresholds_{mode}.csv")

    week0 = pd.Timestamp(first) - pd.Timedelta(days=pd.Timestamp(first).weekday())
    start = min(week0, pd.Timestamp(first).replace(day=1))  # monthly facts key on the 1st
    tables["dim_date"] = build_dim_date(
        str(start.date()),
        str(pd.Timestamp(last_day).date()),
        str(pd.Timestamp(first).date()),
        str(pd.Timestamp(last).date()),
    )

    info: dict[str, Any] = {"mode": mode, "format": fmt, "tables": {}}
    for name, df in tables.items():
        path = out / f"{name}.{'parquet' if fmt == 'parquet' else 'csv'}"
        if fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        info["tables"][name] = {"rows": int(len(df)), "mb": round(path.stat().st_size / 1e6, 2)}
        log.info("%-26s %9d rows %8.2f MB", name, len(df), path.stat().st_size / 1e6)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    (out / "_export_info.json").write_text(json.dumps(info, indent=2) + "\n")
    write_data_dictionary(tables, info, root / "dashboard" / "data_dictionary.md")
    return info


def write_data_dictionary(
    tables: dict[str, pd.DataFrame], info: dict[str, Any], path: Path
) -> None:
    """Markdown list of tables, row counts and column types (generated, not hand-written)."""
    lines = [
        "# Power BI data dictionary",
        "",
        f"*Generated by `python -m fdda.export.powerbi` ({info['mode']} data, {info['format']}). "
        "Files are written to `data/processed/powerbi/` (git-ignored).*",
        "",
        "| Table | Rows | Size (MB) | Description |",
        "|---|---|---|---|",
    ]
    for name in tables:
        t = info["tables"][name]
        lines.append(f"| `{name}` | {t['rows']:,} | {t['mb']} | {DESCRIPTIONS.get(name, '')} |")
    fs = tables["fact_sales"]
    rev, full = fs["indexed_revenue"].sum(), fs["full_price_value"].sum()
    checks = {
        "Units (sum of fact_sales[units])": f"{int(fs['units'].sum()):,}",
        "Revenue (indexed)": f"{rev:,.0f}",
        "Full-price value": f"{full:,.0f}",
        "Avg Discount % (1 - revenue / full-price value)": f"{1 - rev / full:.1%}",
        "Discounted Revenue Share": f"{fs['discounted_revenue'].sum() / rev:.1%}",
        "Customers (rows of fact_customer)": f"{len(tables['fact_customer']):,}",
    }
    lines += [
        "",
        "## Check totals",
        "",
        "With no filters applied, the dashboard's cards should show these values.",
        "",
        "| Measure | Expected |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in checks.items()],
    ]
    for name, df in tables.items():
        lines += ["", f"## {name}", "", "| Column | Type |", "|---|---|"]
        lines += [f"| `{c}` | {df[c].dtype} |" for c in df.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Export the Power BI star schema")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dev", dest="mode", action="store_const", const="dev")
    mode.add_argument("--full", dest="mode", action="store_const", const="full")
    parser.add_argument("--csv", action="store_true", help="write CSV instead of Parquet")
    args = parser.parse_args(argv)
    settings = load_settings()
    info = export(settings, args.mode or data_mode(settings), "csv" if args.csv else "parquet")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
