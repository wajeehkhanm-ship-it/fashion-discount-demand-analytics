"""Weekly demand series per category (PROJECT_PLAN 4.6).

Weeks start on Monday. Partial weeks at the start and end of the data are dropped so every
point is a full week. `promo_share` = share of the category's units sold at least
`promo_threshold` below the article-week reference price (Phase 3), a promo-intensity signal.
"""

from __future__ import annotations

import re

import duckdb
import pandas as pd


def weekly_category_series(
    con: duckdb.DuckDBPyConnection,
    txd: str,
    level: str,
    top_n: int,
    promo_threshold: float,
) -> pd.DataFrame:
    """Long table: category, ds (week start), y (units), promo_share; top `top_n` categories."""
    if not re.fullmatch(r"[a-z_]+", level):
        raise ValueError(f"invalid column name: {level}")
    thr = float(promo_threshold)
    df = con.execute(f"""
        WITH bounds AS (SELECT min(t_dat) AS lo, max(t_dat) AS hi FROM {txd}),
        top AS (
            SELECT {level} AS category FROM {txd}
            GROUP BY 1 ORDER BY count(*) DESC LIMIT {int(top_n)}
        )
        SELECT t.{level} AS category,
               date_trunc('week', t.t_dat)::DATE AS ds,
               count(*) AS y,
               avg((t.discount >= {thr})::INT) AS promo_share
        FROM {txd} t, bounds b
        WHERE t.{level} IN (SELECT category FROM top)
          AND date_trunc('week', t.t_dat) >= b.lo
          AND date_trunc('week', t.t_dat) + INTERVAL 6 DAY <= b.hi
        GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    df["ds"] = pd.to_datetime(df["ds"])
    # Make every category cover every week (a missing week means zero units).
    weeks = pd.date_range(df["ds"].min(), df["ds"].max(), freq="W-MON")
    full = pd.MultiIndex.from_product([df["category"].unique(), weeks], names=["category", "ds"])
    df = df.set_index(["category", "ds"]).reindex(full).reset_index()
    df["y"] = df["y"].fillna(0).astype(float)
    df["promo_share"] = df["promo_share"].fillna(0.0)
    order = df.groupby("category")["y"].sum().sort_values(ascending=False).index
    df["category"] = pd.Categorical(df["category"], categories=list(order), ordered=True)
    return df.sort_values(["category", "ds"]).reset_index(drop=True)
