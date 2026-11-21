"""Exploratory data analysis queries (Phase 2). All heavy lifting happens in DuckDB.

Every function takes an open DuckDB connection and a *relation* string (SQL that can
follow FROM, e.g. from `hm_transactions`) and returns a small pandas DataFrame.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from fdda.features.pricing import band_case_sql

OTHER = "Other"


def _q(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


# ------------------------------------------------------------------ relations


def hm_transactions(interim_dir: Path) -> str:
    """Relation for the cleaned, month-partitioned H&M transactions under `interim_dir`."""
    glob = Path(interim_dir) / "hm" / "transactions" / "*" / "*.parquet"
    return f"read_parquet({_q(glob)}, hive_partitioning = true)"


def hm_articles(interim_dir: Path) -> str:
    """Relation for the cleaned H&M article catalog."""
    return f"read_parquet({_q(Path(interim_dir) / 'hm' / 'articles.parquet')})"


def myntra_products(interim_dir: Path) -> str:
    """Relation for the cleaned Myntra listings."""
    return f"read_parquet({_q(Path(interim_dir) / 'myntra' / 'products.parquet')})"


# --------------------------------------------------------------- data quality


def load_cleaning_summary(reports_dir: Path, dataset: str, mode: str) -> dict[str, Any]:
    """Read the JSON summary that `fdda.data.clean` wrote for a dataset/mode."""
    path = Path(reports_dir) / f"data_summary_{dataset}_{mode}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def cleaning_steps_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Turn a cleaning summary into a step-by-step table: rule, rows removed, rows left."""
    rows, left = [("raw file", 0, summary["raw_rows"])], summary["raw_rows"]
    for key, value in summary.items():
        if key.startswith("dropped_"):
            left -= value
            rows.append((key.removeprefix("dropped_").replace("_", " "), value, left))
    table = pd.DataFrame(rows, columns=["step", "rows_removed", "rows_left"])
    table["pct_of_raw_left"] = table["rows_left"] / summary["raw_rows"]
    return table


def data_quality(con: duckdb.DuckDBPyConnection, relation: str) -> pd.DataFrame:
    """Per-column quality table: type, non-null count, % null, approx distinct, min, max."""
    df = con.execute(f"SUMMARIZE SELECT * FROM {relation}").df()
    df = df.rename(columns={"column_name": "column", "column_type": "type"})
    df["null_pct"] = pd.to_numeric(df["null_percentage"], errors="coerce") / 100
    df["non_null"] = (df["count"] * (1 - df["null_pct"])).round().astype("int64")
    return df[["column", "type", "non_null", "null_pct", "approx_unique", "min", "max"]]


# ---------------------------------------------------------------- category mix


def _fold_other(df: pd.DataFrame, label_col: str, top_n: int, sum_cols: list[str]) -> pd.DataFrame:
    """Keep the `top_n` rows (df must be sorted descending) and sum the rest into 'Other'."""
    if len(df) <= top_n:
        return df.reset_index(drop=True)
    head, tail = df.iloc[:top_n], df.iloc[top_n:]
    other = {label_col: OTHER, **{c: tail[c].sum() for c in sum_cols}}
    return pd.concat([head, pd.DataFrame([other])], ignore_index=True)


def hm_category_mix(
    con: duckdb.DuckDBPyConnection, tx: str, level: str = "product_group_name", top_n: int = 8
) -> pd.DataFrame:
    """Units and indexed revenue by an article-hierarchy level, with shares; tail -> 'Other'."""
    if not re.fullmatch(r"[a-z_]+", level):
        raise ValueError(f"invalid column name: {level}")
    df = con.execute(f"""
        SELECT {level} AS category, count(*) AS units, sum(indexed_price) AS indexed_revenue
        FROM {tx} GROUP BY 1 ORDER BY units DESC
    """).df()
    df = _fold_other(df, "category", top_n, ["units", "indexed_revenue"])
    df["unit_share"] = df["units"] / df["units"].sum()
    df["revenue_share"] = df["indexed_revenue"] / df["indexed_revenue"].sum()
    return df


def hm_weekly_sales(con: duckdb.DuckDBPyConnection, tx: str) -> pd.DataFrame:
    """Weekly units, indexed revenue and unique customers (weeks start Monday)."""
    return con.execute(f"""
        SELECT date_trunc('week', t_dat)::DATE AS week,
               count(*) AS units,
               sum(indexed_price) AS indexed_revenue,
               count(DISTINCT customer_id) AS customers
        FROM {tx} GROUP BY 1 ORDER BY 1
    """).df()


def hm_channel_mix(con: duckdb.DuckDBPyConnection, tx: str) -> pd.DataFrame:
    """Units and indexed revenue by sales_channel_id, with shares."""
    df = con.execute(f"""
        SELECT sales_channel_id, count(*) AS units, sum(indexed_price) AS indexed_revenue
        FROM {tx} GROUP BY 1 ORDER BY 1
    """).df()
    df["unit_share"] = df["units"] / df["units"].sum()
    df["revenue_share"] = df["indexed_revenue"] / df["indexed_revenue"].sum()
    return df


# --------------------------------------------------------------------- prices


def hm_price_sample(
    con: duckdb.DuckDBPyConnection, tx: str, n: int, seed: int, top_groups: list[str]
) -> pd.DataFrame:
    """Reservoir sample of indexed prices with product group (tail groups -> 'Other')."""
    groups = ", ".join(_q(g) for g in top_groups) or "NULL"
    return con.execute(f"""
        SELECT indexed_price,
               CASE WHEN product_group_name IN ({groups}) THEN product_group_name
                    ELSE '{OTHER}' END AS category
        FROM {tx} USING SAMPLE reservoir({int(n)} ROWS) REPEATABLE ({int(seed)})
    """).df()


def price_quantiles(
    con: duckdb.DuckDBPyConnection, relation: str, column: str, by: str | None = None
) -> pd.DataFrame:
    """P10/P25/median/P75/P90 of `column`, overall or per `by` group."""
    for name in filter(None, [column, by]):
        if not re.fullmatch(r"[a-z_]+", name):
            raise ValueError(f"invalid column name: {name}")
    group_sel = f"{by} AS grp," if by else "'all' AS grp,"
    group_by = "GROUP BY 1" if by else ""
    return con.execute(f"""
        SELECT {group_sel} count(*) AS n,
               quantile_cont({column}, 0.10) AS p10, quantile_cont({column}, 0.25) AS p25,
               quantile_cont({column}, 0.50) AS median, quantile_cont({column}, 0.75) AS p75,
               quantile_cont({column}, 0.90) AS p90
        FROM {relation} {group_by} ORDER BY n DESC
    """).df()


# ------------------------------------------------------------------ discounts


def hm_discount_vs_max(max_discount: float) -> str:
    """SQL for a *preliminary* H&M discount: 1 - price / the article's lifetime max price.

    Flagged price outliers are excluded from the max. Phase 3 replaces this with a
    rolling 8-week P95 reference price (PROJECT_PLAN 4.2).
    """
    ref = "max(CASE WHEN NOT is_price_outlier THEN price END) OVER (PARTITION BY article_id)"
    return f"coalesce(least(greatest(1 - price / {ref}, 0), {float(max_discount)}), 0)"


def hm_discount_band_mix(
    con: duckdb.DuckDBPyConnection,
    tx: str,
    edges: list[float],
    labels: list[str],
    max_discount: float,
) -> pd.DataFrame:
    """Units and indexed revenue by preliminary discount band (vs article lifetime max)."""
    band = band_case_sql("discount", edges, labels)
    df = con.execute(f"""
        WITH d AS (SELECT indexed_price, {hm_discount_vs_max(max_discount)} AS discount FROM {tx})
        SELECT {band} AS band, count(*) AS units, sum(indexed_price) AS indexed_revenue,
               median(discount) AS median_discount
        FROM d GROUP BY 1
    """).df()
    return _order_bands(df, labels, ["units", "indexed_revenue"])


def _order_bands(df: pd.DataFrame, labels: list[str], share_cols: list[str]) -> pd.DataFrame:
    """Sort rows by band order, include empty bands as zero, and add share columns."""
    df = df.set_index("band").reindex(labels).reset_index()
    for col in share_cols:
        df[col] = df[col].fillna(0)
        df[f"{col}_share"] = df[col] / df[col].sum()
    return df


# --------------------------------------------------------------------- Myntra


def myntra_category_sql(column: str, rules: list[dict[str, Any]]) -> str:
    """SQL CASE assigning a rough category from whole-word keywords; first match wins."""
    parts = []
    for rule in rules:
        words = "|".join(re.escape(k.lower()) for k in rule["keywords"])
        pattern = f"(^|[^a-z0-9-])({words})($|[^a-z0-9-])".replace("'", "''")
        label = rule["category"].replace("'", "''")
        parts.append(f"WHEN regexp_matches(lower({column}), '{pattern}') THEN '{label}'")
    return f"CASE {' '.join(parts)} ELSE '{OTHER}' END"


def myntra_with_category(products: str, rules: list[dict[str, Any]]) -> str:
    """Relation of Myntra listings with an added keyword-derived `category` column."""
    return f"(SELECT *, {myntra_category_sql('name', rules)} AS category FROM {products})"


def myntra_category_mix(
    con: duckdb.DuckDBPyConnection, products: str, rules: list[dict[str, Any]]
) -> pd.DataFrame:
    """Listings per rough category with share, median price/discount and rated share."""
    rel = myntra_with_category(products, rules)
    df = con.execute(f"""
        SELECT category, count(*) AS listings,
               median(price) AS median_price, median(discount_pct) AS median_discount,
               avg((rating_total > 0)::INT) AS share_with_ratings
        FROM {rel} GROUP BY 1 ORDER BY listings DESC
    """).df()
    df["listing_share"] = df["listings"] / df["listings"].sum()
    # Keep 'Other' last so the chart reads top-down by size.
    return pd.concat([df[df.category != OTHER], df[df.category == OTHER]], ignore_index=True)


def myntra_discount_band_mix(
    con: duckdb.DuckDBPyConnection, products: str, edges: list[float], labels: list[str]
) -> pd.DataFrame:
    """Listings by discount band (vs MRP), with the share of listings that have any rating."""
    band = band_case_sql("discount_pct", edges, labels)
    df = con.execute(f"""
        SELECT {band} AS band, count(*) AS listings,
               avg((rating_total > 0)::INT) AS share_with_ratings
        FROM {products} GROUP BY 1
    """).df()
    return _order_bands(df, labels, ["listings"])


def myntra_sample(con: duckdb.DuckDBPyConnection, products: str, n: int, seed: int) -> pd.DataFrame:
    """Reservoir sample of Myntra price, MRP and discount for distribution plots."""
    return con.execute(f"""
        SELECT price, mrp, discount_pct, rating_total
        FROM {products} USING SAMPLE reservoir({int(n)} ROWS) REPEATABLE ({int(seed)})
    """).df()


# --------------------------------------------------------------------- output


def save_table(df: pd.DataFrame, name: str, out_dir: Path) -> Path:
    """Write a result table as CSV (the notebook's numbers come from these files)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path
