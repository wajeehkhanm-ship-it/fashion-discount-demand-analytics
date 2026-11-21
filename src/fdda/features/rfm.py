"""Customer RFM features, segments and churn flags.

* Snapshot date = last transaction date + 1 day (unless set in config).
* Recency = days from the last purchase to the snapshot; frequency = distinct purchase days;
  monetary = indexed revenue.
* Scores 1-5 from percentile ranks, so tied customers share a score (e.g. every
  one-purchase customer gets the same F score) instead of being split arbitrarily.
* Segments come from an R x F rule table in config; first match wins.
* Churned (per window w) = no purchase in the last w days, i.e. recency > w.
* revenue_last_year = indexed revenue in the 365 days up to the customer's last purchase:
  the yearly spend that is lapsing when a customer churns ("revenue at stake").
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import duckdb

UNASSIGNED = "Unassigned"


def _q(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def score_sql(column: str, higher_is_better: bool = True) -> str:
    """SQL scoring `column` 1-5 by percentile rank (ties share a score; 5 = best)."""
    order = "ASC" if higher_is_better else "DESC"
    return f"least(1 + floor(5 * percent_rank() OVER (ORDER BY {column} {order})), 5)::INTEGER"


def segment_case_sql(rules: list[dict[str, Any]], r: str = "r_score", f: str = "f_score") -> str:
    """SQL CASE mapping R and F scores to a segment name using the config rule table."""
    parts = []
    for rule in rules:
        (r_lo, r_hi), (f_lo, f_hi) = rule["r"], rule["f"]
        name = rule["segment"].replace("'", "''")
        parts.append(
            f"WHEN {r} BETWEEN {int(r_lo)} AND {int(r_hi)} "
            f"AND {f} BETWEEN {int(f_lo)} AND {int(f_hi)} THEN '{name}'"
        )
    return f"CASE {' '.join(parts)} ELSE '{UNASSIGNED}' END"


def check_rule_coverage(rules: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return the (R, F) cells that no rule covers (should be empty)."""
    missing = []
    for r in range(1, 6):
        for f in range(1, 6):
            if not any(
                rule["r"][0] <= r <= rule["r"][1] and rule["f"][0] <= f <= rule["f"][1]
                for rule in rules
            ):
                missing.append((r, f))
    return missing


def build_customer_rfm(
    con: duckdb.DuckDBPyConnection,
    tx: str,
    out_path: Path,
    rules: list[dict[str, Any]],
    churn_windows: list[int],
    snapshot: date | str | None = None,
) -> dict[str, Any]:
    """Compute one row per customer (RFM, scores, segment, churn flags) and write Parquet.

    `tx` needs customer_id, t_dat, sales_channel_id and indexed_price. Returns the snapshot
    date and counts.
    """
    if missing := check_rule_coverage(rules):
        raise ValueError(f"RFM rules leave cells uncovered: {missing}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot is None:
        snapshot = con.execute(f"SELECT max(t_dat) + INTERVAL 1 DAY FROM {tx}").fetchone()[0]
    snap = str(snapshot)[:10]
    churn_cols = ", ".join(f"recency_days > {int(w)} AS churned_{int(w)}d" for w in churn_windows)
    con.execute(f"""
        COPY (
            WITH c AS (
                SELECT customer_id,
                       min(t_dat) AS first_date,
                       max(t_dat) AS last_date,
                       count(DISTINCT t_dat) AS frequency,
                       count(DISTINCT (t_dat, sales_channel_id)) AS orders,
                       count(*) AS units,
                       sum(indexed_price) AS monetary,
                       avg((sales_channel_id = 2)::INT) AS channel2_share
                FROM {tx} GROUP BY customer_id
            ),
            ly AS (
                SELECT t.customer_id, sum(t.indexed_price) AS revenue_last_year
                FROM {tx} t JOIN c ON t.customer_id = c.customer_id
                WHERE t.t_dat > c.last_date - INTERVAL 365 DAY
                GROUP BY t.customer_id
            ),
            base AS (
                SELECT c.*, coalesce(ly.revenue_last_year, 0) AS revenue_last_year,
                       date_diff('day', c.last_date, DATE '{snap}') AS recency_days
                FROM c LEFT JOIN ly ON c.customer_id = ly.customer_id
            ),
            scored AS (
                SELECT *,
                       {score_sql("recency_days", higher_is_better=False)} AS r_score,
                       {score_sql("frequency")} AS f_score,
                       {score_sql("monetary")} AS m_score
                FROM base
            )
            SELECT *, {segment_case_sql(rules)} AS segment, {churn_cols} FROM scored
        ) TO {_q(out_path)} (FORMAT parquet)
    """)
    n, unassigned = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE segment = '{UNASSIGNED}') "
        f"FROM read_parquet({_q(out_path)})"
    ).fetchone()
    return {"snapshot_date": snap, "customers": n, "unassigned": unassigned}
