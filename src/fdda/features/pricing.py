"""Pricing features: reference price, discount %, discount bands and the article x week table.

Method (PROJECT_PLAN 4.2):
* Reference (full) price per article-week = the 95th percentile of the article's observed
  price over a trailing 8-week window (current week included). Flagged price outliers are
  ignored. If the window has fewer than `min_obs` usable sales, the article's lifetime max
  price is used instead.
* discount = 1 - price / reference price, clipped to [0, max_discount].
* Rows are aggregated to article x week: units, indexed revenue, indexed reference price,
  median and mean discount, and the discount band of the week's median discount.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import duckdb


def _q(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def band_case_sql(column: str, edges: list[float], labels: list[str]) -> str:
    """SQL CASE expression mapping a discount fraction in `column` to its band label.

    `edges` are ascending band boundaries (e.g. [0, 0.05, ..., 0.9]); band i covers
    [edges[i], edges[i+1]). Values at or above the last inner edge fall in the last band,
    and values below the first edge fall in the first band.
    """
    if len(labels) != len(edges) - 1:
        raise ValueError("need exactly one label per band (len(edges) - 1)")
    if list(edges) != sorted(edges):
        raise ValueError("band edges must be ascending")
    parts = [
        f"WHEN {column} < {float(upper)} THEN '{label}'"
        for upper, label in zip(edges[1:-1], labels[:-1], strict=True)
    ]
    return f"CASE {' '.join(parts)} ELSE '{labels[-1]}' END"


def week_index_sql(date_col: str) -> str:
    """SQL for an integer week index (Monday-start weeks), so windows can use RANGE."""
    return f"(epoch(date_trunc('week', {date_col})) // 604800)::INTEGER"


def build_transaction_discounts(
    con: duckdb.DuckDBPyConnection,
    tx: str,
    article_weeks_path: Path,
    out_path: Path,
    max_discount: float,
) -> int:
    """Write transactions with a per-unit `discount` vs their article-week reference price.

    Keeps only the columns later phases need (KPIs, RFM, Power BI export). Returns the row
    count. Requires the article x week table from `build_article_weeks`.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT t.t_dat, t.customer_id, t.article_id, t.product_code, t.product_group_name,
                   t.product_type_name, t.sales_channel_id, t.indexed_price,
                   least(greatest(1 - t.indexed_price / a.indexed_ref_price, 0),
                         {float(max_discount)}) AS discount
            FROM {tx} t
            JOIN read_parquet({_q(article_weeks_path)}) a
              ON t.article_id = a.article_id
             AND date_trunc('week', t.t_dat)::DATE = a.week
        ) TO {_q(out_path)} (FORMAT parquet)
    """)
    return con.execute(f"SELECT count(*) FROM read_parquet({_q(out_path)})").fetchone()[0]


def build_article_weeks(
    con: duckdb.DuckDBPyConnection,
    tx: str,
    out_path: Path,
    *,
    quantile: float,
    window_weeks: int,
    min_obs: int,
    max_discount: float,
    price_rescale_factor: float,
    edges: list[float],
    labels: list[str],
) -> dict[str, Any]:
    """Compute reference prices and write the article x week table to `out_path` (Parquet).

    `tx` is a relation over cleaned transactions (see `fdda.analysis.eda.hm_transactions`).
    Returns a summary dict (row counts, share of article-weeks using the lifetime fallback).
    """
    if not 0 < quantile <= 1 or window_weeks < 1 or min_obs < 1:
        raise ValueError("invalid reference-price parameters")
    t0 = time.perf_counter()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    usable = "CASE WHEN NOT is_price_outlier THEN price END"
    band = band_case_sql("median_discount", edges, labels)
    con.execute(f"""
        COPY (
            WITH base AS (
                SELECT article_id, product_code, product_group_name, product_type_name,
                       garment_group_name, section_name, price, indexed_price,
                       is_price_outlier,
                       date_trunc('week', t_dat)::DATE AS week,
                       {week_index_sql("t_dat")} AS week_idx
                FROM {tx}
            ),
            ref AS (
                SELECT *,
                       quantile_cont({usable}, {float(quantile)}) OVER w AS ref_window,
                       count({usable}) OVER w AS n_window,
                       max({usable}) OVER (PARTITION BY article_id) AS ref_lifetime
                FROM base
                WINDOW w AS (PARTITION BY article_id ORDER BY week_idx
                             RANGE BETWEEN {int(window_weeks) - 1} PRECEDING AND CURRENT ROW)
            ),
            rows AS (
                SELECT *,
                       n_window < {int(min_obs)} OR ref_window IS NULL AS used_fallback,
                       CASE WHEN n_window >= {int(min_obs)} AND ref_window IS NOT NULL
                            THEN ref_window ELSE ref_lifetime END AS ref_price
                FROM ref
            ),
            disc AS (
                SELECT *, least(greatest(1 - price / ref_price, 0), {float(max_discount)})
                          AS discount
                FROM rows WHERE ref_price IS NOT NULL
            ),
            aw AS (
                SELECT article_id, week, week_idx,
                       any_value(product_code) AS product_code,
                       any_value(product_group_name) AS product_group_name,
                       any_value(product_type_name) AS product_type_name,
                       any_value(garment_group_name) AS garment_group_name,
                       any_value(section_name) AS section_name,
                       count(*) AS units,
                       sum(indexed_price) AS indexed_revenue,
                       any_value(ref_price) * {float(price_rescale_factor)} AS indexed_ref_price,
                       median(discount) AS median_discount,
                       avg(discount) AS mean_discount,
                       bool_or(used_fallback) AS used_fallback
                FROM disc GROUP BY article_id, week, week_idx
            )
            SELECT *, {band} AS band FROM aw ORDER BY article_id, week_idx
        ) TO {_q(out_path)} (FORMAT parquet)
    """)
    rel = f"read_parquet({_q(out_path)})"
    n_aw, n_art, units, fallback = con.execute(f"""
        SELECT count(*), count(DISTINCT article_id), sum(units), avg(used_fallback::INT)
        FROM {rel}
    """).fetchone()
    n_rows = con.execute(f"SELECT count(*) FROM {tx}").fetchone()[0]
    return {
        "transactions": n_rows,
        "units_in_article_weeks": int(units),
        "dropped_no_reference_price": n_rows - int(units),
        "article_weeks": n_aw,
        "articles": n_art,
        "share_article_weeks_lifetime_fallback": fallback,
        "seconds": round(time.perf_counter() - t0, 1),
    }
