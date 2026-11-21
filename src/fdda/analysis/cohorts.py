"""Monthly acquisition-cohort retention (PROJECT_PLAN 4.5).

Cohort = the month of a customer's first purchase in the data. Retention in month k = share
of the cohort that bought anything k months after the cohort month (k = 0 is 100%).
Caveat: the data starts in Sep 2018, so earlier customers look "new" in the first months
(left censoring); the first cohorts are inflated.
"""

from __future__ import annotations

import duckdb
import pandas as pd


def retention_matrix(con: duckdb.DuckDBPyConnection, tx: str) -> pd.DataFrame:
    """Long table: cohort, months_since, active customers, cohort_size, retention."""
    df = con.execute(f"""
        WITH firsts AS (
            SELECT customer_id, date_trunc('month', min(t_dat))::DATE AS cohort
            FROM {tx} GROUP BY customer_id
        ),
        active AS (
            SELECT DISTINCT customer_id, date_trunc('month', t_dat)::DATE AS month FROM {tx}
        )
        SELECT f.cohort, date_diff('month', f.cohort, a.month) AS months_since,
               count(*) AS customers
        FROM active a JOIN firsts f ON a.customer_id = f.customer_id
        GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    size = df[df["months_since"] == 0].set_index("cohort")["customers"]
    df["cohort_size"] = df["cohort"].map(size)
    df["retention"] = df["customers"] / df["cohort_size"]
    df["cohort"] = pd.to_datetime(df["cohort"]).dt.strftime("%Y-%m")
    return df


def retention_wide(long: pd.DataFrame, max_months: int | None = None) -> pd.DataFrame:
    """Pivot to cohort x months_since (retention), optionally truncated."""
    wide = long.pivot(index="cohort", columns="months_since", values="retention").sort_index()
    if max_months is not None:
        wide = wide.loc[:, [c for c in wide.columns if c <= max_months]]
    return wide


def average_retention_curve(long: pd.DataFrame) -> pd.DataFrame:
    """Customer-weighted average retention by months_since.

    For each k, only cohorts observed for at least k months count in the denominator, so
    young cohorts do not drag the later months down.
    """
    period = pd.PeriodIndex(long["cohort"], freq="M")
    last = (period + long["months_since"].to_numpy()).max()
    sizes = long[long["months_since"] == 0].set_index("cohort")["cohort_size"]
    observed = pd.Series(
        [(last - pd.Period(c, freq="M")).n for c in sizes.index], index=sizes.index
    )
    rows = []
    for k in sorted(long["months_since"].unique()):
        cohorts = observed[observed >= k].index
        active = long[(long["months_since"] == k) & long["cohort"].isin(cohorts)]["customers"].sum()
        base = sizes.loc[cohorts].sum()
        rows.append({"months_since": k, "active": active, "base": base, "retention": active / base})
    return pd.DataFrame(rows)
