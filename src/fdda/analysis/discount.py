"""Discount payback analysis.

Inputs: the article x week table from `fdda.features.pricing.build_article_weeks`.

1. Band economics: for each article, mean units / indexed revenue / margin per week in each
   discount band divided by the same at full price (band 0); medians across articles, by
   category. Margins use assumed full-price gross margins (no cost data exists):
   unit cost = (1 - m) x reference price.
2. Elasticity: log(units) on log(1 - discount) with article and week fixed effects, per
   category, standard errors clustered by article.
3. Payback threshold: the first band where the marginal index change per extra discount
   point turns negative; the "ceiling" is the band before it.
4. Myntra cross-check: rating count (demand proxy) and price x rating count (revenue proxy)
   by discount band and keyword category. Directional only.

These are associations corrected for article and week effects, not causal estimates:
markdowns are not random (slow sellers get discounted).

Usage: `python -m fdda.analysis.discount [--dev | --full]`
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from fdda.analysis import eda
from fdda.config import data_mode, load_settings, resolve_path
from fdda.data.clean import connect
from fdda.features.pricing import band_case_sql, build_article_weeks
from fdda.log import get_logger

log = get_logger(__name__)
ALL = "All"


def margin_col(m: float) -> str:
    """Column name for a margin scenario, e.g. 0.6 -> 'margin_60'."""
    return f"margin_{round(m * 100)}"


def add_margins(aw: pd.DataFrame, margins: list[float]) -> pd.DataFrame:
    """Add per-article-week margin columns: revenue - (1 - m) x reference price x units."""
    aw = aw.copy()
    for m in margins:
        aw[margin_col(m)] = aw["indexed_revenue"] - (1 - m) * aw["indexed_ref_price"] * aw["units"]
    return aw


# --------------------------------------------------------------- band economics


def attach_baseline(
    aw: pd.DataFrame, full_price_band: str, margins: list[float], window: int | None
) -> pd.DataFrame:
    """Add the full-price baseline each article-week is compared against.

    Adds base_units, base_indexed_revenue, base_margin_<m> (mean per full-price week) and
    n_base_weeks. With `window=None` the baseline is all of the article's full-price weeks
    (lifetime). With an int, it is the article's full-price weeks within +/- `window` weeks,
    excluding the week itself (local; controls for lifecycle decay). Article-weeks with no
    baseline weeks are dropped.
    """
    mcols = [margin_col(m) for m in margins]
    value_cols = ["units", "indexed_revenue", *mcols]
    fp = aw[aw["band"] == full_price_band]
    if window is None:
        base = fp.groupby("article_id")[value_cols].mean().add_prefix("base_")
        base["n_base_weeks"] = fp.groupby("article_id").size()
        return aw.join(base, on="article_id", how="inner")
    con = duckdb.connect()
    con.register("aw", aw[["article_id", "week_idx"]])
    con.register("fp", fp[["article_id", "week_idx", *value_cols]])
    avgs = ", ".join(f"avg(fp.{c}) AS base_{c}" for c in value_cols)
    base = con.execute(f"""
        SELECT a.article_id, a.week_idx, {avgs}, count(*) AS n_base_weeks
        FROM aw a JOIN fp
          ON a.article_id = fp.article_id
         AND fp.week_idx BETWEEN a.week_idx - {int(window)} AND a.week_idx + {int(window)}
         AND fp.week_idx <> a.week_idx
        GROUP BY 1, 2
    """).df()
    con.close()
    return aw.merge(base, on=["article_id", "week_idx"], how="inner")


def band_economics(
    aw_base: pd.DataFrame,
    labels: list[str],
    level: str,
    margins: list[float],
    min_articles_per_category: int,
    min_articles_per_band: int,
) -> pd.DataFrame:
    """Within-article band indexes vs the full-price baseline, medians across articles.

    `aw_base` comes from `attach_baseline`. For each article and band the index is
    mean(value in the band's weeks) / mean(baseline for those weeks); the table reports
    the median across articles, overall ('All') and per category. Output columns:
    category, band, n_articles, band_discount, units_index (+ p25/p75), revenue_index,
    margin_index_<m>.
    """
    mcols = [margin_col(m) for m in margins]
    value_cols = ["units", "indexed_revenue", *mcols]
    sums = value_cols + [f"base_{c}" for c in value_cols]
    per = (
        aw_base.groupby(["article_id", "band"], observed=True)
        .agg(
            category=(level, "first"),
            disc=("median_discount", "median"),
            **{c: (c, "sum") for c in sums},
        )
        .reset_index()
    )
    per = per[(per[[f"base_{c}" for c in mcols]] > 0).all(axis=1)]  # positive baseline margin
    per["units_index"] = per["units"] / per["base_units"]
    per["revenue_index"] = per["indexed_revenue"] / per["base_indexed_revenue"]
    margin_idx = [c.replace("margin_", "margin_index_") for c in mcols]
    for c, ic in zip(mcols, margin_idx, strict=True):
        per[ic] = per[c] / per[f"base_{c}"]

    big = per.groupby("category")["article_id"].nunique()
    big = big[big >= min_articles_per_category].index
    stacked = pd.concat(
        [per.assign(category=ALL), per[per["category"].isin(big)]], ignore_index=True
    )
    idx_cols = ["units_index", "revenue_index", *margin_idx]
    grouped = stacked.groupby(["category", "band"], observed=True)
    out = grouped.agg(
        n_articles=("article_id", "nunique"),
        band_discount=("disc", "median"),
        **{c: (c, "median") for c in idx_cols},
    )
    out["units_index_p25"] = grouped["units_index"].quantile(0.25)
    out["units_index_p75"] = grouped["units_index"].quantile(0.75)
    out = out.reset_index()
    small = out["n_articles"] < min_articles_per_band
    out.loc[small, [*idx_cols, "units_index_p25", "units_index_p75"]] = np.nan
    out["band"] = pd.Categorical(out["band"], categories=labels, ordered=True)
    order = [ALL, *sorted(c for c in out["category"].unique() if c != ALL)]
    out["category"] = pd.Categorical(out["category"], categories=order, ordered=True)
    return out.sort_values(["category", "band"]).reset_index(drop=True)


def payback_threshold(
    econ: pd.DataFrame, value_col: str, start_band: str | None = None
) -> pd.DataFrame:
    """Per category: the ceiling band, i.e. the last band before the marginal index change
    per extra discount point turns negative.

    With `start_band=None` the walk starts at full price (the standard question: does a
    discount beat full price?). With `start_band` set (e.g. the shallowest discount band)
    the walk starts there instead, answering: once an article is marked down, beyond which
    depth do deeper markdowns stop paying back? `econ` must be sorted by band within
    category; bands with a missing index are skipped. If the marginal never turns negative,
    the deepest observed band is the ceiling and `turns_negative` is False.
    """
    rows = []
    for cat, g in econ.groupby("category", observed=True, sort=False):
        g = g.dropna(subset=[value_col, "band_discount"])
        if start_band is not None:
            order = list(g["band"].cat.categories) if hasattr(g["band"], "cat") else None
            labels = order or list(dict.fromkeys(g["band"].astype(str)))
            if start_band not in labels:
                continue
            keep = labels[labels.index(start_band) :]
            g = g[g["band"].astype(str).isin(keep)]
        if len(g) < 2:
            continue
        d = g["band_discount"].to_numpy()
        v = g[value_col].to_numpy()
        bands = g["band"].astype(str).to_list()
        marginal = np.diff(v) / (100 * np.diff(d))  # index change per discount point
        neg = np.flatnonzero(marginal < 0)
        k = int(neg[0]) if len(neg) else len(v) - 1
        rows.append(
            {
                "category": cat,
                "measure": value_col,
                "reference": "full price" if start_band is None else f"from {start_band}",
                "ceiling_band": bands[k],
                "ceiling_discount": float(d[k]),
                "ceiling_index": float(v[k]),
                "turns_negative": bool(len(neg)),
                "first_negative_band": bands[k + 1] if len(neg) else None,
                "marginal_at_turn": float(marginal[k]) if len(neg) else None,
                "peak_band": bands[int(np.argmax(v))],
                "peak_index": float(np.max(v)),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ elasticity


def _drop_singletons(df: pd.DataFrame, fe: list[str]) -> pd.DataFrame:
    """Repeatedly drop rows whose fixed-effect group has a single observation."""
    while True:
        keep = np.ones(len(df), dtype=bool)
        for col in fe:
            keep &= df.groupby(col)[col].transform("size").to_numpy() > 1
        if keep.all():
            return df
        df = df[keep]


def _demean(
    v: np.ndarray, codes: list[np.ndarray], tol: float, max_iter: int
) -> tuple[np.ndarray, int]:
    """Alternating projections: subtract group means for each FE until they vanish."""
    v = v.astype(float).copy()
    counts = [np.bincount(c) for c in codes]
    for it in range(1, max_iter + 1):
        delta = 0.0
        for c, n in zip(codes, counts, strict=True):
            means = np.bincount(c, weights=v, minlength=len(n)) / np.maximum(n, 1)
            v -= means[c]
            delta = max(delta, float(np.abs(means).max()))
        if delta < tol:
            return v, it
    return v, max_iter


def fe_elasticity(
    df: pd.DataFrame,
    tol: float = 1e-8,
    max_iter: int = 500,
    extra_fe: list[str] | None = None,
) -> dict[str, float | int]:
    """Fixed-effects estimate of d log(units) / d log(price ratio).

    Uses x = log(1 - mean_discount) and y = log(units) per article-week, with article and
    week fixed effects plus any `extra_fe` columns (e.g. an article-age bucket); standard
    errors clustered by article.
    """
    fe = ["article_id", "week_idx", *(extra_fe or [])]
    d = df[[*fe, "units", "mean_discount"]].dropna()
    d = d[d["units"] > 0]
    d = _drop_singletons(d, fe)
    if d.empty or d["mean_discount"].nunique() < 2:
        return {"beta": math.nan, "se": math.nan, "n_obs": len(d)}
    codes = [pd.factorize(d[c])[0] for c in fe]
    art, wk = codes[0], codes[1]
    y = np.log(d["units"].to_numpy(dtype=float))
    x = np.log1p(-d["mean_discount"].to_numpy(dtype=float))
    xt, it_x = _demean(x, codes, tol, max_iter)
    yt, it_y = _demean(y, codes, tol, max_iter)
    sxx = float(xt @ xt)
    if sxx <= 0:
        return {"beta": math.nan, "se": math.nan, "n_obs": len(d)}
    beta = float(xt @ yt) / sxx
    resid = yt - beta * xt
    n, g, n_weeks = len(d), art.max() + 1, wk.max() + 1
    score = np.bincount(art, weights=xt * resid)
    correction = (g / (g - 1)) * ((n - 1) / max(n - 1 - n_weeks, 1))
    se = math.sqrt(correction * float(score @ score)) / sxx
    return {
        "beta": beta,
        "se": se,
        "ci_low": beta - 1.96 * se,
        "ci_high": beta + 1.96 * se,
        "n_obs": n,
        "n_articles": int(g),
        "n_weeks": int(n_weeks),
        "share_obs_discounted": float((d["mean_discount"] >= 0.05).mean()),
        "iterations": max(it_x, it_y),
    }


def implied_optimal_discount(beta: float, margin: float) -> float:
    """Margin-maximising discount under constant elasticity `beta` and full-price margin m.

    Margin ~ (m - d) * (1 - d)^beta is maximised at d* = (1 + beta*m) / (1 + beta) when
    beta < -1/m; otherwise any discount lowers margin and d* = 0.
    """
    if not math.isfinite(beta) or beta >= -1 / margin:
        return 0.0
    return (1 + beta * margin) / (1 + beta)


def add_age_bucket(aw: pd.DataFrame, edges: list[int]) -> pd.DataFrame:
    """Add `age_bucket`: index of weeks-since-first-sale within ascending `edges`."""
    age = aw["week_idx"] - aw.groupby("article_id")["week_idx"].transform("min")
    return aw.assign(age_bucket=np.searchsorted(np.asarray(edges), age, side="right") - 1)


def elasticity_by_category(
    aw: pd.DataFrame,
    level: str,
    min_articles: int,
    margins: list[float],
    tol: float,
    max_iter: int,
    age_edges: list[int] | None = None,
) -> pd.DataFrame:
    """Elasticity for 'All' and each category with >= `min_articles` articles.

    With `age_edges`, an article-age bucket is added as a third fixed effect.
    """
    extra = None
    if age_edges:
        aw, extra = add_age_bucket(aw, age_edges), ["age_bucket"]
    sizes = aw.groupby(level)["article_id"].nunique()
    cats = [ALL, *sorted(sizes[sizes >= min_articles].index)]
    rows = []
    for cat in cats:
        sub = aw if cat == ALL else aw[aw[level] == cat]
        res = {"category": cat, **fe_elasticity(sub, tol, max_iter, extra)}
        for m in margins:
            res[f"implied_optimal_discount_{round(m * 100)}"] = implied_optimal_discount(
                res["beta"], m
            )
        rows.append(res)
        log.info("elasticity %-22s beta=%.3f se=%.3f", cat, res["beta"], res.get("se", math.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- Myntra check


def myntra_crosscheck(
    con: duckdb.DuckDBPyConnection,
    products: str,
    rules: list[dict[str, Any]],
    edges: list[float],
    labels: list[str],
    min_listings: int,
) -> pd.DataFrame:
    """Demand / revenue proxies by discount band, overall and by keyword category.

    rating_total (verified ratings) proxies cumulative demand; price x rating_total proxies
    revenue. Indexes are relative to the 0-5% band of the same category.
    """
    rel = eda.myntra_with_category(products, rules)
    band = band_case_sql("discount_pct", edges, labels)
    query = f"""
        SELECT {{grp}} AS category, {band} AS band, count(*) AS listings,
               median(discount_pct) AS band_discount,
               avg((rating_total > 0)::INT) AS share_rated,
               avg(rating_total) AS mean_ratings,
               avg(price * rating_total) AS mean_revenue_proxy
        FROM {rel} GROUP BY 1, 2
    """
    df = pd.concat(
        [
            con.execute(query.format(grp=f"'{ALL}'")).df(),
            con.execute(query.format(grp="category")).df(),
        ],
        ignore_index=True,
    )
    df = df[df["category"] != eda.OTHER]
    df.loc[df["listings"] < min_listings, ["mean_ratings", "mean_revenue_proxy", "share_rated"]] = (
        np.nan
    )
    df["band"] = pd.Categorical(df["band"], categories=labels, ordered=True)
    base = df[df["band"] == labels[0]].set_index("category")
    for col in ["mean_ratings", "mean_revenue_proxy"]:
        df[f"{col}_index"] = df[col] / df["category"].map(base[col])
    order = [ALL, *sorted(c for c in df["category"].unique() if c != ALL)]
    df["category"] = pd.Categorical(df["category"], categories=order, ordered=True)
    return df.sort_values(["category", "band"]).reset_index(drop=True)


# --------------------------------------------------------------------- driver


def _row(df: pd.DataFrame, **match: Any) -> dict[str, Any]:
    sel = df
    for k, v in match.items():
        sel = sel[sel[k].astype(str) == str(v)]
    return sel.iloc[0].to_dict() if len(sel) else {}


def build_summary(
    feat: dict[str, Any],
    econ: pd.DataFrame,
    thresholds: pd.DataFrame,
    elast: pd.DataFrame,
    myntra: pd.DataFrame,
    margins: list[float],
    central: float,
    labels: list[str],
    headline: str,
    mode: str,
) -> dict[str, Any]:
    """Headline numbers for the memo and README (all taken from the computed tables).

    Uses the headline baseline for band results and the age-controlled elasticity.
    """
    cm = f"margin_index_{round(central * 100)}"
    ec = econ[econ["baseline"] == headline]
    base = thresholds[thresholds["baseline"] == headline]
    conditional = f"from {labels[1]}"

    def ceilings(ref: str) -> dict[str, Any]:
        th = base[base["reference"] == ref]
        by_m = {
            m: _row(th, category=ALL, measure=f"margin_index_{round(m * 100)}") for m in margins
        }
        vals = [r["ceiling_discount"] for r in by_m.values() if r]
        per_cat = th[(th["measure"] == cm) & (th["category"].astype(str) != ALL)]
        return {
            "overall": by_m[central],
            "range": {"min": min(vals), "max": max(vals)} if vals else {},
            "by_margin": {str(m): r for m, r in by_m.items()},
            "revenue": _row(th, category=ALL, measure="revenue_index"),
            "by_category": per_cat.to_dict(orient="records"),
        }

    other = thresholds[
        (thresholds["baseline"] != headline)
        & (thresholds["measure"] == cm)
        & (thresholds["reference"] == conditional)
    ]
    my_all = myntra[myntra["category"].astype(str) == ALL]
    return {
        "mode": mode,
        "headline_baseline": headline,
        "article_weeks": feat,
        "central_margin": central,
        "vs_full_price": ceilings("full price"),
        "among_discounts": ceilings(conditional),
        "sensitivity_other_baseline": _row(other, category=ALL),
        "overall_band_economics": ec[ec["category"].astype(str) == ALL].to_dict(orient="records"),
        "elasticity_all": _row(elast, category=ALL, spec="article+week+age FE"),
        "elasticity_all_no_age": _row(elast, category=ALL, spec="article+week FE"),
        "myntra_peak_revenue_band": (
            str(my_all.loc[my_all["mean_revenue_proxy"].idxmax(), "band"]) if len(my_all) else None
        ),
        "myntra_all": my_all.to_dict(orient="records"),
        "bands": labels,
    }


def run(settings: dict[str, Any], mode: str) -> dict[str, Any]:
    """Compute every discount-analysis output for `mode` and write CSV/JSON to reports/discount/."""
    p, da = settings["pricing"], settings["discount_analysis"]
    edges, labels = p["discount_bands"]["edges"], p["discount_bands"]["labels"]
    margins, central = p["margin_scenarios"], p["central_margin"]
    level = da["category_level"]
    interim = resolve_path(settings, "interim") / mode
    out_dir = resolve_path(settings, "reports") / "discount"
    out_dir.mkdir(parents=True, exist_ok=True)
    aw_path = resolve_path(settings, "processed") / mode / "article_weeks.parquet"
    con = connect(settings)

    log.info("Building article x week table (%s) ...", mode)
    feat = build_article_weeks(
        con,
        eda.hm_transactions(interim),
        aw_path,
        quantile=p["reference_price_quantile"],
        window_weeks=p["reference_window_weeks"],
        min_obs=p["reference_min_obs"],
        max_discount=settings["cleaning"]["max_discount"],
        price_rescale_factor=settings["cleaning"]["price_rescale_factor"],
        edges=edges,
        labels=labels,
    )
    log.info("article-weeks: %s", feat)
    aw = add_margins(pd.read_parquet(aw_path), margins)

    measures = [
        "units_index",
        "revenue_index",
        *[f"margin_index_{round(m * 100)}" for m in margins],
    ]
    econs, ths = [], []
    for name, window in [("local", da["baseline_window_weeks"]), ("lifetime", None)]:
        log.info("Band economics, %s baseline ...", name)
        e = band_economics(
            attach_baseline(aw, labels[0], margins, window),
            labels,
            level,
            margins,
            da["min_articles_per_category"],
            da["min_articles_per_band"],
        )
        econs.append(e.assign(baseline=name))
        t = pd.concat(
            [payback_threshold(e, m, ref) for m in measures for ref in (None, labels[1])],
            ignore_index=True,
        )
        ths.append(t.assign(baseline=name))
    econ, thresholds = pd.concat(econs, ignore_index=True), pd.concat(ths, ignore_index=True)

    elasts = []
    for spec, age in [("article+week+age FE", da["age_bucket_edges"]), ("article+week FE", None)]:
        log.info("Elasticity, %s ...", spec)
        el = elasticity_by_category(
            aw,
            level,
            da["min_articles_per_category"],
            margins,
            da["fe_tolerance"],
            da["fe_max_iter"],
            age,
        )
        elasts.append(el.assign(spec=spec))
    elast = pd.concat(elasts, ignore_index=True)

    myntra = myntra_crosscheck(
        con,
        eda.myntra_products(interim),
        settings["eda"]["myntra_category_rules"],
        edges,
        labels,
        da["min_articles_per_band"],
    )

    tables = {
        "band_economics": econ,
        "thresholds": thresholds,
        "elasticity": elast,
        "myntra_crosscheck": myntra,
    }
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}_{mode}.csv", index=False)
    summary = build_summary(
        feat,
        econ,
        thresholds,
        elast,
        myntra,
        margins,
        central,
        labels,
        da["headline_baseline"],
        mode,
    )
    path = out_dir / f"discount_summary_{mode}.json"
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    log.info("Wrote %s and %d tables to %s", path.name, len(tables), out_dir)
    write_memo(settings, mode)
    return summary


def write_memo(settings: dict[str, Any], mode: str) -> Path:
    """Render the pricing memo from the saved discount-analysis outputs.

    Full data -> reports/pricing_recommendation.md; dev -> reports/discount/..._dev.md.
    """
    from fdda.analysis.memo import render_pricing_memo

    reports = resolve_path(settings, "reports")
    d = reports / "discount"
    summary = json.loads((d / f"discount_summary_{mode}.json").read_text(encoding="utf-8"))
    tables = {
        n: pd.read_csv(d / f"{n}_{mode}.csv")
        for n in ["band_economics", "thresholds", "elasticity", "myntra_crosscheck"]
    }
    text = render_pricing_memo(
        summary,
        tables["band_economics"],
        tables["thresholds"],
        tables["elasticity"],
        tables["myntra_crosscheck"],
        settings,
    )
    out = (
        reports / "pricing_recommendation.md"
        if mode == "full"
        else d / f"pricing_recommendation_{mode}.md"
    )
    out.write_text(text, encoding="utf-8")
    log.info("Wrote %s", out)
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Discount payback analysis")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dev", dest="mode", action="store_const", const="dev")
    mode.add_argument("--full", dest="mode", action="store_const", const="full")
    args = parser.parse_args(argv)
    settings = load_settings()
    run(settings, args.mode or data_mode(settings))


if __name__ == "__main__":
    main()
