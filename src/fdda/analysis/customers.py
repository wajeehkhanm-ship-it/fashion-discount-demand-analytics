"""Customer analysis: KPIs, RFM segments, churn sensitivity, cohorts and win-back playbook.

Usage: `python -m fdda.analysis.customers [--dev | --full]`
Requires the article x week table from the discount analysis (`make discount`) for
per-unit discounts and the discount ceiling that caps win-back offers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from fdda.analysis import cohorts, eda, kpis
from fdda.config import data_mode, load_settings, resolve_path
from fdda.data.clean import connect
from fdda.features.pricing import band_case_sql, build_transaction_discounts
from fdda.features.rfm import build_customer_rfm
from fdda.log import get_logger

log = get_logger(__name__)
ALL = "All customers"


def _q(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


# ------------------------------------------------------------------ segments


def segment_summary(
    con: duckdb.DuckDBPyConnection, rfm: str, churn_window: int, order: list[str]
) -> pd.DataFrame:
    """Per segment: size, recency/frequency/monetary, revenue at stake, churn share."""
    df = con.execute(f"""
        SELECT segment, count(*) AS customers,
               median(recency_days) AS median_recency_days,
               avg(frequency) AS avg_frequency, avg(orders) AS avg_orders,
               avg(monetary) AS avg_monetary, sum(monetary) AS total_monetary,
               sum(revenue_last_year) AS revenue_at_stake,
               avg(revenue_last_year) AS revenue_at_stake_per_customer,
               avg(churned_{int(churn_window)}d::INT) AS churned_share,
               avg(channel2_share) AS channel2_share,
               avg(r_score) AS avg_r, avg(f_score) AS avg_f, avg(m_score) AS avg_m
        FROM {rfm} GROUP BY 1
    """).df()
    df["customer_share"] = df["customers"] / df["customers"].sum()
    df["monetary_share"] = df["total_monetary"] / df["total_monetary"].sum()
    df["segment"] = pd.Categorical(df["segment"], categories=order, ordered=True)
    return df.sort_values("segment").reset_index(drop=True)


def churn_sensitivity(
    con: duckdb.DuckDBPyConnection, rfm: str, windows: list[int], order: list[str]
) -> pd.DataFrame:
    """Churned share and count per segment (and overall) for each churn window."""
    cols = ", ".join(
        f"avg(churned_{int(w)}d::INT) AS churned_share_{int(w)}d, "
        f"sum(churned_{int(w)}d::INT) AS churned_{int(w)}d"
        for w in windows
    )
    seg = con.execute(f"SELECT segment, count(*) AS customers, {cols} FROM {rfm} GROUP BY 1").df()
    tot = con.execute(f"SELECT '{ALL}' AS segment, count(*) AS customers, {cols} FROM {rfm}").df()
    seg["segment"] = pd.Categorical(seg["segment"], categories=order, ordered=True)
    seg = seg.sort_values("segment")
    seg["segment"] = seg["segment"].astype(str)
    return pd.concat([seg, tot], ignore_index=True)


def segment_category_preferences(
    con: duckdb.DuckDBPyConnection, txd: str, rfm: str, level: str, top_k: int = 3
) -> pd.DataFrame:
    """Top `top_k` categories by units per segment, with share of the segment's units."""
    df = con.execute(f"""
        WITH u AS (
            SELECT r.segment, t.{level} AS category, count(*) AS units
            FROM {txd} t JOIN {rfm} r ON t.customer_id = r.customer_id
            GROUP BY 1, 2
        )
        SELECT segment, category, units,
               units / sum(units) OVER (PARTITION BY segment) AS unit_share,
               row_number() OVER (PARTITION BY segment ORDER BY units DESC) AS rank
        FROM u QUALIFY rank <= {int(top_k)} ORDER BY segment, rank
    """).df()
    return df


def segment_discount_profile(
    con: duckdb.DuckDBPyConnection,
    txd: str,
    rfm: str,
    edges: list[float],
    labels: list[str],
    discounted_threshold: float,
) -> pd.DataFrame:
    """Units by discount band per segment, share within segment, and lift vs all customers.

    Lift > 1 means the segment buys a larger share of its units in that band than the
    customer base does: the band the segment "responds to".
    """
    band = band_case_sql("t.discount", edges, labels)
    thr = float(discounted_threshold)
    df = con.execute(f"""
        SELECT r.segment, {band} AS band, count(*) AS units,
               median(t.discount) AS median_discount
        FROM {txd} t JOIN {rfm} r ON t.customer_id = r.customer_id
        GROUP BY 1, 2
    """).df()
    df["unit_share"] = df["units"] / df.groupby("segment")["units"].transform("sum")
    overall = df.groupby("band")["units"].sum()
    overall_share = overall / overall.sum()
    df["lift_vs_all"] = df["unit_share"] / df["band"].map(overall_share)
    disc = con.execute(f"""
        SELECT r.segment,
               avg((t.discount >= {thr})::INT) AS discounted_unit_share,
               median(t.discount) FILTER (WHERE t.discount >= {thr})
                   AS median_discount_when_discounted
        FROM {txd} t JOIN {rfm} r ON t.customer_id = r.customer_id GROUP BY 1
    """).df()
    df["band"] = pd.Categorical(df["band"], categories=labels, ordered=True)
    return df.sort_values(["segment", "band"]).merge(disc, on="segment", how="left")


# ------------------------------------------------------------------ playbook


def round_down_to(x: float, step: float = 0.05) -> float:
    """Round a fraction down to a multiple of `step` (offers are quoted in 5% steps)."""
    return math.floor(x / step + 1e-9) * step


def build_playbook(
    summary: pd.DataFrame,
    prefs: pd.DataFrame,
    profile: pd.DataFrame,
    cfg: list[dict[str, Any]],
    ceiling: dict[str, Any],
    labels: list[str],
) -> pd.DataFrame:
    """One row per win-back segment, combining computed facts with configured actions.

    Offer depth = the segment's median discount when it buys on discount, rounded down to
    5%, and capped at the markdown-ceiling band's median discount.
    """
    cap = round_down_to(ceiling["ceiling_discount"])
    rows = []
    for item in cfg:
        seg = item["segment"]
        s = summary[summary["segment"].astype(str) == seg]
        if s.empty:
            continue
        s = s.iloc[0]
        p = prefs[prefs["segment"] == seg].sort_values("rank")
        prof = profile[profile["segment"] == seg]
        discounted = prof[prof["band"].astype(str) != labels[0]]
        responds = (
            str(discounted.loc[discounted["lift_vs_all"].idxmax(), "band"])
            if len(discounted)
            else None
        )
        typical = float(prof["median_discount_when_discounted"].iloc[0]) if len(prof) else 0.0
        offer = min(round_down_to(typical), cap)
        rows.append(
            {
                "segment": seg,
                "customers": int(s["customers"]),
                "customer_share": s["customer_share"],
                "churned_share": s["churned_share"],
                "median_recency_days": s["median_recency_days"],
                "revenue_at_stake": s["revenue_at_stake"],
                "revenue_at_stake_per_customer": s["revenue_at_stake_per_customer"],
                "favourite_categories": ", ".join(
                    f"{c} ({u:.0%})" for c, u in zip(p["category"], p["unit_share"], strict=True)
                ),
                "discounted_unit_share": float(prof["discounted_unit_share"].iloc[0])
                if len(prof)
                else None,
                "typical_discount_when_discounted": typical,
                "responds_to_band": responds,
                "offer_discount": offer,
                "offer_capped": offer < round_down_to(typical),
                "offer_cap": cap,
                "channel2_share": s["channel2_share"],
                "action": item["action"],
                "channel": item["channel"],
                "timing": item["timing"],
                "kpi": item["kpi"],
            }
        )
    return pd.DataFrame(rows)


def render_playbook_md(
    playbook: pd.DataFrame,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    info: dict[str, Any],
) -> str:
    """Markdown win-back playbook; every number comes from the computed tables."""
    w = info["churn_window"]
    ceiling = info["ceiling"]
    total_stake = playbook["revenue_at_stake"].sum()
    lines = [
        "# Win-back playbook",
        "",
        f"*Produced by `python -m fdda.analysis.customers` ({info['mode']} data, snapshot "
        f"{info['snapshot_date']}). Numbers come from `reports/customers/*_{info['mode']}.csv`.*",
        "",
        "## Summary",
        "",
        f"{info['customers']:,} customers were scored on recency, frequency and monetary value. "
        f"The {len(playbook)} segments below are the churned or at-risk ones: "
        f"{int(playbook['customers'].sum()):,} customers "
        f"({playbook['customer_share'].sum():.0%} of the base) with "
        f"{total_stake:,.0f} indexed revenue at stake (their spend in the year before their last "
        f"purchase). Offers are capped at the markdown ceiling: "
        f"**{ceiling['ceiling_band']} (max {info['offer_cap']:.0%} off)**, beyond which deeper "
        "discounts cut margin.",
        "",
        "## Playbook",
        "",
        "| Segment | Customers | Churned (" + f"{w}d) | Revenue at stake | Per customer | "
        "Favourite categories | Buys on discount | Responds to | Offer | Action | Channel | "
        "Timing | Success KPI |",
        "|" + "---|" * 13,
    ]
    for _, r in playbook.iterrows():
        offer = f"{r['offer_discount']:.0%}" + (" (capped)" if r["offer_capped"] else "")
        lines.append(
            f"| **{r['segment']}** | {r['customers']:,} ({r['customer_share']:.0%}) | "
            f"{r['churned_share']:.0%} | {r['revenue_at_stake']:,.0f} | "
            f"{r['revenue_at_stake_per_customer']:,.0f} | {r['favourite_categories']} | "
            f"{r['discounted_unit_share']:.0%} of units | {r['responds_to_band']} | {offer} | "
            f"{r['action']} | {r['channel']} | {r['timing']} | {r['kpi']} |"
        )
    lines += [
        "",
        "* **Revenue at stake** = indexed revenue in the 365 days up to each customer's last "
        "purchase: the yearly spend that lapses if they do not come back.",
        "* **Responds to** = the discount band where the segment buys a larger share of its units "
        "than the customer base does.",
        "* **Offer** = the segment's median discount when it buys on discount, rounded down to "
        "5%, capped at the markdown ceiling. Run every offer against a holdout group so the "
        "KPI measures incremental, not total, reactivation.",
        "",
        "## All segments",
        "",
        "| Segment | Customers | Share | Median recency (days) | Avg purchase days | "
        "Avg indexed revenue | Share of revenue | Churned |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['segment']} | {int(r['customers']):,} | {r['customer_share']:.1%} | "
            f"{r['median_recency_days']:.0f} | {r['avg_frequency']:.1f} | "
            f"{r['avg_monetary']:,.0f} | {r['monetary_share']:.1%} | {r['churned_share']:.0%} |"
        )
    wins = [c for c in sensitivity.columns if c.startswith("churned_share_")]
    lines += [
        "",
        "## Churn-window sensitivity",
        "",
        "Share of customers counted as churned (no purchase in the last N days) under each window.",
        "",
        "| Segment | " + " | ".join(c.removeprefix("churned_share_") for c in wins) + " |",
        "|---|" + "---|" * len(wins),
    ]
    for _, r in sensitivity.iterrows():
        name = f"**{r['segment']}**" if r["segment"] == ALL else r["segment"]
        lines.append(f"| {name} | " + " | ".join(f"{r[c]:.0%}" for c in wins) + " |")
    lines += [
        "",
        "## Caveats",
        "",
        "* Scores are relative (percentile ranks within this customer base).",
        "* The data ends on the snapshot date, so a long gap may be seasonal rather than "
        "permanent churn; the sensitivity table shows how much the window matters.",
        "* Discount behaviour is observational: a segment that bought on discount may have "
        "bought anyway. The holdout design in the KPI column is what proves incrementality.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ driver


def run(settings: dict[str, Any], mode: str) -> dict[str, Any]:
    """Compute every customer-analysis output for `mode`."""
    cs, p = settings["customers"], settings["pricing"]
    edges, labels = p["discount_bands"]["edges"], p["discount_bands"]["labels"]
    thr = cs["discounted_threshold"]
    level = settings["discount_analysis"]["category_level"]
    interim = resolve_path(settings, "interim") / mode
    processed = resolve_path(settings, "processed") / mode
    reports = resolve_path(settings, "reports")
    out = reports / "customers"
    out.mkdir(parents=True, exist_ok=True)
    aw_path = processed / "article_weeks.parquet"
    disc_summary = reports / "discount" / f"discount_summary_{mode}.json"
    if not aw_path.exists() or not disc_summary.exists():
        raise FileNotFoundError(
            f"Discount-analysis outputs for '{mode}' missing; run `make discount` first."
        )
    con = connect(settings)

    log.info("Per-unit discounts ...")
    txd_path = processed / "transactions_discount.parquet"
    n_tx = build_transaction_discounts(
        con, eda.hm_transactions(interim), aw_path, txd_path, settings["cleaning"]["max_discount"]
    )
    txd = f"read_parquet({_q(txd_path)})"
    articles = eda.hm_articles(interim)

    log.info("KPIs ...")
    monthly = kpis.monthly_kpis(con, txd, thr)
    totals = kpis.kpi_totals(con, txd, thr)
    cat_share = kpis.category_share_monthly(con, txd, level)
    top_articles = kpis.top_items(con, txd, articles, "article_id", cs["top_n"], thr)
    top_styles = kpis.top_items(con, txd, articles, "product_code", cs["top_n"], thr)

    log.info("RFM ...")
    rfm_path = processed / "customer_rfm.parquet"
    windows = sorted(set(cs["churn_sensitivity_days"]) | {cs["churn_window_days"]})
    rfm_info = build_customer_rfm(
        con, txd, rfm_path, cs["rfm_segments"], windows, cs["snapshot_date"]
    )
    rfm = f"read_parquet({_q(rfm_path)})"
    order = [r["segment"] for r in cs["rfm_segments"]]
    summary = segment_summary(con, rfm, cs["churn_window_days"], order)
    sensitivity = churn_sensitivity(con, rfm, cs["churn_sensitivity_days"], order)
    prefs = segment_category_preferences(con, txd, rfm, level)
    profile = segment_discount_profile(con, txd, rfm, edges, labels, thr)

    log.info("Cohorts ...")
    retention = cohorts.retention_matrix(con, txd)
    curve = cohorts.average_retention_curve(retention)

    ceiling = json.loads(disc_summary.read_text())["among_discounts"]["overall"]
    playbook = build_playbook(summary, prefs, profile, cs["winback"], ceiling, labels)

    tables = {
        "monthly_kpis": monthly,
        "category_share_monthly": cat_share,
        "top_articles": top_articles,
        "top_styles": top_styles,
        "segment_summary": summary,
        "churn_sensitivity": sensitivity,
        "segment_categories": prefs,
        "segment_discount_profile": profile,
        "cohort_retention": retention,
        "retention_curve": curve,
        "winback_playbook": playbook,
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}_{mode}.csv", index=False)

    info = {
        "mode": mode,
        "transactions": n_tx,
        **rfm_info,
        "churn_window": cs["churn_window_days"],
        "ceiling": ceiling,
        "offer_cap": round_down_to(ceiling["ceiling_discount"]),
        "totals": totals,
    }
    (out / f"customers_summary_{mode}.json").write_text(
        json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8"
    )
    md = render_playbook_md(playbook, summary, sensitivity, info)
    md_path = (
        reports / "winback_playbook.md" if mode == "full" else out / f"winback_playbook_{mode}.md"
    )
    md_path.write_text(md, encoding="utf-8")
    log.info("Wrote %d tables to %s and %s", len(tables), out, md_path.name)
    return info


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="KPIs, RFM, cohorts and win-back playbook")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dev", dest="mode", action="store_const", const="dev")
    mode.add_argument("--full", dest="mode", action="store_const", const="full")
    args = parser.parse_args(argv)
    settings = load_settings()
    run(settings, args.mode or data_mode(settings))


if __name__ == "__main__":
    main()
