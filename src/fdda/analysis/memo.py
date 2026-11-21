"""Render the pricing & promotion recommendation memo from the discount-analysis outputs.

Every number in the memo is read from the tables/summary written by
`fdda.analysis.discount`; nothing is typed in by hand. Re-running the analysis
re-renders the memo.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

ALL = "All"


def _pct(x: float | None, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}%}"


def _num(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _join(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _band_row(econ: pd.DataFrame, baseline: str, band: str, category: str = ALL) -> pd.Series:
    sel = econ[
        (econ["baseline"] == baseline)
        & (econ["category"].astype(str) == category)
        & (econ["band"].astype(str) == band)
    ]
    return sel.iloc[0] if len(sel) else pd.Series(dtype=float)


def _category_table(
    thresholds: pd.DataFrame,
    elast: pd.DataFrame,
    baseline: str,
    reference: str,
    margins: list[float],
    central: float,
) -> str:
    th = thresholds[(thresholds["baseline"] == baseline) & (thresholds["reference"] == reference)]
    el = elast[elast["spec"] == "article+week+age FE"].set_index("category")
    cm = f"margin_index_{round(central * 100)}"
    cats = th[th["measure"] == cm].sort_values("ceiling_discount", ascending=False)
    head = (
        f"| Category | Ceiling ({central:.0%} margin) | Margin index at ceiling | "
        + " | ".join(f"Ceiling at {m:.0%}" for m in margins if m != central)
        + " | Elasticity β [95% CI] |"
    )
    sep = "|" + "---|" * (4 + len([m for m in margins if m != central]))
    lines = [head, sep]
    for _, r in cats.iterrows():
        others = []
        for m in margins:
            if m == central:
                continue
            o = th[
                (th["measure"] == f"margin_index_{round(m * 100)}")
                & (th["category"] == r["category"])
            ]
            others.append(o["ceiling_band"].iloc[0] if len(o) else "n/a")
        e = el.loc[r["category"]] if r["category"] in el.index else None
        beta = (
            f"{_num(e['beta'])} [{_num(e['ci_low'])}, {_num(e['ci_high'])}]"
            if e is not None
            else "n/a"
        )
        name = "**All articles**" if r["category"] == ALL else r["category"]
        lines.append(
            f"| {name} | {r['ceiling_band']} | {_num(r['ceiling_index'])} | "
            + " | ".join(others)
            + f" | {beta} |"
        )
    return "\n".join(lines)


def _band_table(econ: pd.DataFrame, baseline: str, labels: list[str], margins: list[float]) -> str:
    head = (
        "| Band | Articles | Median discount | Units / week | Revenue / week | "
        + " | ".join(f"Margin / week ({m:.0%})" for m in margins)
        + " |"
    )
    lines = [head, "|" + "---|" * (5 + len(margins))]
    for b in labels:
        r = _band_row(econ, baseline, b)
        if r.empty:
            continue
        lines.append(
            f"| {b} | {int(r['n_articles']):,} | {_pct(r['band_discount'])} | "
            f"{_num(r['units_index'])} | {_num(r['revenue_index'])} | "
            + " | ".join(_num(r[f"margin_index_{round(m * 100)}"]) for m in margins)
            + " |"
        )
    return "\n".join(lines)


def render_pricing_memo(
    summary: dict[str, Any],
    econ: pd.DataFrame,
    thresholds: pd.DataFrame,
    elast: pd.DataFrame,
    myntra: pd.DataFrame,
    settings: dict[str, Any],
) -> str:
    """Return the memo as Markdown."""
    labels = summary["bands"]
    margins = settings["pricing"]["margin_scenarios"]
    central = summary["central_margin"]
    baseline = summary["headline_baseline"]
    window = settings["discount_analysis"]["baseline_window_weeks"]
    aw = summary["article_weeks"]
    vf, ad = summary["vs_full_price"], summary["among_discounts"]
    cm = f"margin_index_{round(central * 100)}"
    ceil_band = ad["overall"]["ceiling_band"]
    ceil_d = ad["overall"]["ceiling_discount"]
    nxt_band = ad["overall"].get("first_negative_band")
    shallow = _band_row(econ, baseline, labels[1])
    at_ceiling = _band_row(econ, baseline, ceil_band)
    nxt = _band_row(econ, baseline, nxt_band) if nxt_band else pd.Series(dtype=float)
    deepest = _band_row(econ, baseline, labels[-1])
    el, el0 = summary["elasticity_all"], summary["elasticity_all_no_age"]
    sens = summary["sensitivity_other_baseline"]
    my_peak = summary["myntra_peak_revenue_band"]
    my_all = myntra[myntra["category"].astype(str) == ALL]
    my_all = my_all.set_index(my_all["band"].astype(str))
    no_band_beats_full = vf["overall"]["ceiling_band"] == labels[0]
    implied = el.get(f"implied_optimal_discount_{round(central * 100)}", 0.0)

    lo, hi = ad["range"]["min"], ad["range"]["max"]
    span = f"{_pct(min(margins))}–{_pct(max(margins))} margin assumptions"
    range_text = (
        f"the same under all {span}"
        if math.isclose(lo, hi)
        else f"ranging {_pct(lo)}–{_pct(hi)} across {span}"
    )
    head_rows = econ[(econ["baseline"] == baseline) & (econ["category"].astype(str) == ALL)]
    other_rows = econ[(econ["baseline"] != baseline) & (econ["category"].astype(str) == ALL)]
    u_min, u_max = head_rows["units_index"].min(), head_rows["units_index"].max()
    volume_text = (
        "**Discounts barely lift weekly volume.**"
        if u_max <= 1.2
        else "**Discounts lift weekly volume only modestly.**"
    )
    lifetime_lower = other_rows[cm].mean() < head_rows[cm].mean()
    age_masks = el0.get("beta", 0) > el.get("beta", 0)
    cat_rows = ad["by_category"]
    shallower = sorted(
        r["category"] for r in cat_rows if labels.index(r["ceiling_band"]) < labels.index(ceil_band)
    )
    shallow_bands = [r["ceiling_band"] for r in cat_rows if r["category"] in shallower]
    lead = (
        f"Across {aw['articles']:,} H&M articles and {aw['transactions']:,} transactions, "
        + (
            "**no discount band earned more margin per article-week than full price** under any "
            "of the assumed gross margins"
            if no_band_beats_full
            else f"discounts up to **{vf['overall']['ceiling_band']}** still beat full-price margin"
        )
        + f". Once an article is marked down, **margin per week peaks in the {ceil_band} band "
        f"(median discount {_pct(ceil_d)})"
        + (f" and falls sharply beyond it ({nxt_band} onwards)" if nxt_band else "")
        + f"**, {range_text}. "
        f"Myntra's cross-sectional revenue proxy also peaks in the {my_peak} band."
    )

    recs = [
        f"**Cap routine markdowns at the {ceil_band} band (about {_pct(ceil_d)} off).** Going "
        f"deeper cut margin per article-week from {_num(at_ceiling.get(cm))} to "
        f"{_num(nxt.get(cm))} of full-price margin in the next band ({central:.0%}-margin "
        f"scenario) without a matching lift in units.",
        None
        if ceil_band == labels[1] or shallow.get(cm, 0) >= at_ceiling.get(cm, 0)
        else f"**Avoid token discounts in the {labels[1]} band.** They sold "
        f"{_num(shallow.get('units_index'))}× the units of nearby full-price weeks and kept "
        f"only {_num(shallow.get(cm))} of full-price margin: margin is given away "
        + (
            "without extra volume."
            if shallow.get("units_index", 1) <= 1
            else "for little extra volume."
        ),
        f"**Reserve discounts beyond {ceil_band} for clearance.** At {labels[-1]} the "
        f"margin index is {_num(deepest.get(cm))} ({central:.0%} margin). Deep cuts can still "
        "be right for end-of-season stock with no other exit, but that is an inventory decision "
        "(holding cost, salvage value) this data cannot value, not a demand-generation tool.",
        (
            f"**Use category-specific caps** (table below): cap {_join(shallower)} at a "
            f"shallower band ({', '.join(sorted(set(shallow_bands)))}); the other categories "
            "follow the overall pattern."
            if shallower
            else "**Use the same cap across categories:** every category with enough data "
            f"peaks at {ceil_band} or deeper (table below)."
        ),
        f"**Tie promotions together:** win-back offers in the customer win-back playbook are "
        f"capped at the same {ceil_band} band.",
        "**Validate with a controlled test** before rolling out (design below).",
    ]

    my_rows = "\n".join(
        f"| {b} | {int(my_all.loc[b, 'listings']):,} | {_pct(my_all.loc[b, 'share_rated'])} | "
        f"{_num(my_all.loc[b, 'mean_ratings_index'])} | "
        f"{_num(my_all.loc[b, 'mean_revenue_proxy_index'])} |"
        for b in labels
        if b in my_all.index
    )

    return f"""# Pricing & promotion recommendation

*Produced by `python -m fdda.analysis.discount` ({summary["mode"]} data). Every number is read
from `reports/discount/*_{summary["mode"]}.csv` and `discount_summary_{summary["mode"]}.json`.*

## Summary

{lead}

## Recommendations

{chr(10).join(f"{i}. {r}" for i, r in enumerate([r for r in recs if r], 1))}

## What was measured

* **Data:** H&M transactions, {aw["transactions"]:,} units over {aw["article_weeks"]:,}
  article-weeks ({aw["articles"]:,} articles). Prices are normalised, so revenue is an index.
* **Discount:** 1 − price ÷ reference price, where the reference is the 95th percentile of the
  article's price over a trailing 8-week window ({_pct(aw["share_article_weeks_lifetime_fallback"])}
  of article-weeks fall back to the article's lifetime max because the window is thin).
* **Comparison:** for each article, units, revenue and margin per week in a discount band divided
  by the same article's full-price weeks within ±{window} weeks (a *local* baseline, because
  markdowns come late in an article's life when demand has already faded). Medians across
  articles.
* **Margin:** no cost data exists, so unit cost = (1 − m) × reference price for assumed
  full-price gross margins m = {", ".join(_pct(m) for m in margins)}.
* **Ceiling:** the last band before each extra discount point *lowers* the margin index.

## Results by discount band (all articles, {baseline} baseline)

1.00 = the same per week as the article at full price.

{_band_table(econ, baseline, labels, margins)}

* {volume_text} Units per week stay within {_num(u_min)}–{_num(u_max)}× of full price
  across bands, so revenue and margin fall with depth.
* **Elasticity** (log units on log price ratio, article + week + article-age fixed effects):
  β = {_num(el.get("beta"))} [95% CI {_num(el.get("ci_low"))}, {_num(el.get("ci_high"))}],
  i.e. a price 10% further below the reference price is associated with about
  {_pct(abs(el.get("beta", 0)) * 0.1, 1)} {
        "more" if el.get("beta", 0) < 0 else "fewer"
    } units. Without the age control β =
  {_num(el0.get("beta"))}{
        ": lifecycle decay hides part of the discount effect" if age_masks else ""
    }.
  Under a constant elasticity of this size, the implied margin-optimal discount is
  {_pct(implied)}.
* **Sensitivity:** with a *lifetime* baseline the markdown ceiling is
  {sens.get("ceiling_band", "n/a")}{" (the same)" if sens.get("ceiling_band") == ceil_band else ""}{
        ", and discounted bands sit further below full price, because late-life markdown weeks are "
        "compared with launch weeks"
        if lifetime_lower
        else ""
    }.

## Ceiling by category (once marked down)

{_category_table(thresholds, elast, baseline, f"from {labels[1]}", margins, central)}

Elasticity β < 0 means deeper discounts go with more units. Positive values (seen in seasonal
categories) reflect end-of-season clearance: the items discounted in a given week are the ones
that are not selling.

## Myntra cross-check (directional)

Mean per listing relative to the 0–5% band. Different products at different discounts in one
snapshot; rating count proxies cumulative demand. Direction only, not an estimate.

| Band | Listings | Share with ratings | Rating count index | Price × rating count index |
|---|---|---|---|---|
{my_rows}

## Limitations

* **Not causal.** Markdowns are not random: slow sellers get discounted, and timing follows the
  season and lifecycle. Fixed effects and the local baseline remove the main confounders, and the
  remaining bias most likely makes discounts look *less* effective than they are (weak articles
  are the ones being discounted).
* **No zero-sales weeks.** Weeks in which an article sold nothing are not observed.
* **Normalised prices and assumed margins.** Revenue is an index; margins are scenarios.
* **No inventory data.** The value of clearing stock is not counted, which matters most for
  deep end-of-season discounts.

## What an A/B test would add

Randomise markdown depth for comparable articles at the same lifecycle stage, e.g. 0% / 10% /
20% / 30% assigned by store cluster or customer cohort for 4–6 weeks, holding the article set
fixed. Measure units, margin per article-week and sell-through, and confirm whether the
{ceil_band} ceiling holds once selection and timing are removed by design.

## Reproduce

```bash
make clean-data            # FDDA_MODE=full for all customers
FDDA_MODE=full make discount
FDDA_MODE=full make payback-notebook   # notebooks/02_discount_payback.ipynb
```
"""
