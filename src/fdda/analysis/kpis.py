"""Monthly KPIs, category revenue share and top articles/styles (PROJECT_PLAN 4.4).

Input relation: transactions with a per-unit `discount` (see
`fdda.features.pricing.build_transaction_discounts`). An "order" is one customer's purchases
on one day in one channel, because H&M has no order id. Revenue is indexed revenue.
"""

from __future__ import annotations

import re

import duckdb
import pandas as pd

MOM_COLS = ["revenue", "orders", "units", "aov", "customers"]


def _check_col(name: str) -> str:
    if not re.fullmatch(r"[a-z_]+", name):
        raise ValueError(f"invalid column name: {name}")
    return name


def monthly_kpis(
    con: duckdb.DuckDBPyConnection, txd: str, discounted_threshold: float
) -> pd.DataFrame:
    """Revenue, orders, units, AOV, customers, MoM % changes and discounted revenue share.

    `is_partial_month` marks months the data only partly covers (first and last month),
    whose MoM changes are not comparable.
    """
    thr = float(discounted_threshold)
    df = con.execute(f"""
        SELECT strftime(t_dat, '%Y-%m') AS month,
               sum(indexed_price) AS revenue,
               count(DISTINCT (customer_id, t_dat, sales_channel_id)) AS orders,
               count(*) AS units,
               count(DISTINCT customer_id) AS customers,
               sum(indexed_price) FILTER (WHERE discount >= {thr}) / sum(indexed_price)
                   AS discounted_revenue_share,
               count(*) FILTER (WHERE discount >= {thr}) / count(*) AS discounted_unit_share,
               min(t_dat) AS first_day, max(t_dat) AS last_day
        FROM {txd} GROUP BY 1 ORDER BY 1
    """).df()
    df["aov"] = df["revenue"] / df["orders"]
    df["units_per_order"] = df["units"] / df["orders"]
    first = pd.to_datetime(df["first_day"])
    last = pd.to_datetime(df["last_day"])
    df["is_partial_month"] = (first.dt.day > 1) | (last.dt.day < last.dt.days_in_month)
    for col in MOM_COLS:
        df[f"{col}_mom"] = df[col].pct_change()
    return df.drop(columns=["first_day", "last_day"])


def kpi_totals(con: duckdb.DuckDBPyConnection, txd: str, discounted_threshold: float) -> dict:
    """Whole-period totals: revenue, orders, units, customers, AOV, discounted share."""
    thr = float(discounted_threshold)
    rev, orders, units, cust, disc = con.execute(f"""
        SELECT sum(indexed_price), count(DISTINCT (customer_id, t_dat, sales_channel_id)),
               count(*), count(DISTINCT customer_id),
               sum(indexed_price) FILTER (WHERE discount >= {thr}) / sum(indexed_price)
        FROM {txd}
    """).fetchone()
    return {
        "indexed_revenue": rev,
        "orders": orders,
        "units": units,
        "customers": cust,
        "aov": rev / orders,
        "units_per_order": units / orders,
        "discounted_revenue_share": disc,
    }


def category_share_monthly(
    con: duckdb.DuckDBPyConnection, txd: str, level: str = "product_group_name"
) -> pd.DataFrame:
    """Revenue and units per month and category, with each category's share of the month."""
    lvl = _check_col(level)
    df = con.execute(f"""
        SELECT strftime(t_dat, '%Y-%m') AS month, {lvl} AS category,
               sum(indexed_price) AS revenue, count(*) AS units
        FROM {txd} GROUP BY 1, 2 ORDER BY 1, 3 DESC
    """).df()
    df["revenue_share"] = df["revenue"] / df.groupby("month")["revenue"].transform("sum")
    return df


def top_items(
    con: duckdb.DuckDBPyConnection,
    txd: str,
    articles: str,
    by: str,
    n: int,
    discounted_threshold: float,
) -> pd.DataFrame:
    """Top `n` articles (`by='article_id'`) or styles (`by='product_code'`) by revenue."""
    key = _check_col(by)
    thr = float(discounted_threshold)
    extra = (
        "any_value(a.colour_group_name) AS colour, "
        if key == "article_id"
        else "count(DISTINCT t.article_id) AS colour_variants, "
    )
    df = con.execute(f"""
        WITH tot AS (SELECT sum(indexed_price) AS total FROM {txd})
        SELECT t.{key}, any_value(a.prod_name) AS prod_name,
               any_value(t.product_type_name) AS product_type, {extra}
               count(*) AS units, sum(t.indexed_price) AS revenue,
               sum(t.indexed_price) / any_value(tot.total) AS revenue_share,
               avg((t.discount >= {thr})::INT) AS discounted_unit_share
        FROM {txd} t
        LEFT JOIN {articles} a ON t.article_id = a.article_id
        CROSS JOIN tot
        GROUP BY t.{key} ORDER BY revenue DESC LIMIT {int(n)}
    """).df()
    df.insert(0, "rank", range(1, len(df) + 1))
    return df
