"""Generate README.md and reports/resume_bullets.md from pipeline outputs.

Every number is read from files in `reports/` written by the pipeline; anything not
computed yet renders as [TBD]. Run after `make all` (full data):

    python -m fdda.report.readme
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from fdda.config import load_settings, resolve_path
from fdda.log import get_logger

log = get_logger(__name__)
TBD = "[TBD]"
REPO_URL = "https://github.com/wajeehkhanm-ship-it/fashion-discount-demand-analytics"


# ------------------------------------------------------------------ helpers


def load_json(path: Path) -> dict[str, Any] | None:
    """Parsed JSON, or None if the file does not exist."""
    return json.loads(path.read_text()) if path.exists() else None


def load_csv(path: Path) -> pd.DataFrame | None:
    """CSV as a DataFrame, or None if the file does not exist."""
    return pd.read_csv(path) if path.exists() else None


def dig(obj: Any, *keys: Any) -> Any:
    """Nested lookup that returns None when anything along the path is missing."""
    for k in keys:
        if obj is None:
            return None
        try:
            obj = obj[k]
        except (KeyError, IndexError, TypeError):
            return None
    return obj


def pct(x: Any, digits: int = 0) -> str:
    """Fraction -> '12%' (or [TBD] when missing)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return TBD
    return f"{x:.{digits}%}"


def num(x: Any, digits: int = 0) -> str:
    """Number with thousands separators (or [TBD])."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return TBD
    return f"{x:,.{digits}f}"


def millions(x: Any, digits: int = 1) -> str:
    """Large number -> '31.8M' (or [TBD])."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return TBD
    return f"{x / 1e6:.{digits}f}M"


def join_and(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    items = [str(i) for i in items]
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ------------------------------------------------------------------ facts


def collect(settings: dict[str, Any]) -> dict[str, Any]:
    """Load every pipeline output the README needs (None where missing)."""
    r = resolve_path(settings, "reports")
    root = Path(settings["_root"])
    return {
        "hm": load_json(r / "data_summary_hm_full.json"),
        "myntra": load_json(r / "data_summary_myntra_full.json"),
        "discount": load_json(r / "discount" / "discount_summary_full.json"),
        "forecast": load_json(r / "forecast" / "forecast_summary_full.json"),
        "customers": load_json(r / "customers" / "customers_summary_full.json"),
        "segments": load_csv(r / "customers" / "segment_summary_full.csv"),
        "playbook": load_csv(r / "customers" / "winback_playbook_full.csv"),
        "retention": load_csv(r / "customers" / "retention_curve_full.csv"),
        "tagging": load_json(r / "tagging" / "summary.json"),
        "effort": load_json(r / "tagging" / "effort_study.json"),
        "hero": next(
            (
                p.relative_to(root).as_posix()
                for p in [
                    root / "dashboard" / "screenshots" / "1_executive.png",
                    r / "figures" / "payback_threshold_full.png",
                ]
                if p.exists()
            ),
            None,
        ),
        "has_pdf": (root / "dashboard" / "dashboard.pdf").exists(),
    }


def headline(f: dict[str, Any]) -> dict[str, str]:
    """The key numbers, formatted (each falls back to [TBD])."""
    d, fc, cu, tg = f["discount"], f["forecast"], f["customers"], f["tagging"]
    best = dig(fc, "best_prophet_model")
    seg, pb = f["segments"], f["playbook"]
    loyal = seg[seg["segment"].isin(["Champions", "Loyal"])] if seg is not None else None
    effort = dig(f["effort"], "effort_reduction")
    return {
        "transactions": millions(dig(f["hm"], "raw_rows")),
        "transactions_exact": num(dig(f["hm"], "raw_rows")),
        "customers": millions(dig(f["hm"], "n_customers"), 2),
        "myntra_listings": millions(dig(f["myntra"], "raw_rows"), 2),
        "date_min": dig(f["hm"], "date_min") or TBD,
        "date_max": dig(f["hm"], "date_max") or TBD,
        "ceiling_band": dig(d, "among_discounts", "overall", "ceiling_band") or TBD,
        "ceiling_discount": pct(dig(d, "among_discounts", "overall", "ceiling_discount")),
        "ceiling_index": num(dig(d, "among_discounts", "overall", "ceiling_index"), 2),
        "next_band": dig(d, "among_discounts", "overall", "first_negative_band") or TBD,
        "vs_full_band": dig(d, "vs_full_price", "overall", "ceiling_band") or TBD,
        "ceiling_range_min": pct(dig(d, "among_discounts", "range", "min")),
        "ceiling_range_max": pct(dig(d, "among_discounts", "range", "max")),
        "elasticity": num(dig(d, "elasticity_all", "beta"), 2),
        "elasticity_lo": num(dig(d, "elasticity_all", "ci_low"), 2),
        "elasticity_hi": num(dig(d, "elasticity_all", "ci_high"), 2),
        "myntra_peak": dig(d, "myntra_peak_revenue_band") or TBD,
        "articles_analysed": num(dig(d, "article_weeks", "articles")),
        "fc_best": {"prophet": "Prophet", "prophet_promo": "Prophet + promo"}.get(best, TBD),
        "fc_best_mape": pct(dig(fc, "overall", best, "mape"), 1),
        "fc_best_wape": pct(dig(fc, "overall", best, "wape"), 1),
        "fc_prophet_mape": pct(dig(fc, "overall", "prophet", "mape"), 1),
        "fc_naive_mape": pct(dig(fc, "overall", "seasonal_naive", "mape"), 1),
        "fc_naive_wape": pct(dig(fc, "overall", "seasonal_naive", "wape"), 1),
        "fc_beats": len(dig(fc, "categories_where_best_beats_naive_wape") or []),
        "fc_n": dig(fc, "n_categories") or TBD,
        "fc_beat_list": join_and(dig(fc, "categories_where_best_beats_naive_wape") or []),
        "fc_coverage": pct(dig(fc, "interval_coverage", best)),
        "fc_test_weeks": dig(fc, "test_weeks") or TBD,
        "tag_cat": pct(dig(tg, "fields", "category", "test_accuracy"), 1),
        "tag_sub": pct(dig(tg, "fields", "subcategory", "test_accuracy"), 1),
        "tag_pat": pct(dig(tg, "fields", "pattern", "test_accuracy"), 1),
        "tag_col": pct(dig(tg, "fields", "primary_colour", "test_accuracy"), 1),
        "tag_sub_base": pct(dig(tg, "fields", "subcategory", "baseline_test_accuracy"), 1),
        "tag_cat_base": pct(dig(tg, "fields", "category", "baseline_test_accuracy"), 1),
        "tag_sub_title": pct(dig(tg, "fields", "subcategory", "test_accuracy_title_only"), 1),
        "tag_articles": num(dig(tg, "articles")),
        "tag_test": num(dig(tg, "split_articles", "test")),
        "tag_cat_auto_cov": pct(dig(tg, "auto_accept", "per_field", "category", "coverage")),
        "tag_cat_auto_acc": pct(
            dig(tg, "auto_accept", "per_field", "category", "accuracy_above"), 1
        ),
        "tag_sub_auto_cov": pct(dig(tg, "auto_accept", "per_field", "subcategory", "coverage")),
        "tag_sub_auto_acc": pct(
            dig(tg, "auto_accept", "per_field", "subcategory", "accuracy_above"), 1
        ),
        "tag_model": {"logreg": "logistic regression", "xgboost": "XGBoost"}.get(
            dig(tg, "fields", "subcategory", "chosen_model"), TBD
        ),
        "effort": pct(effort) if effort is not None else TBD,
        "n_customers_scored": num(dig(cu, "customers")),
        "loyal_cust_share": pct(loyal["customer_share"].sum()) if loyal is not None else TBD,
        "loyal_rev_share": pct(loyal["monetary_share"].sum()) if loyal is not None else TBD,
        "winback_customers": num(pb["customers"].sum()) if pb is not None else TBD,
        "winback_share": pct(pb["customer_share"].sum()) if pb is not None else TBD,
        "at_stake": millions(pb["revenue_at_stake"].sum()) if pb is not None else TBD,
        "offer_cap": pct(dig(cu, "offer_cap")),
        "m1_retention": pct(f["retention"].set_index("months_since").loc[1, "retention"])
        if f["retention"] is not None
        else TBD,
        "discounted_share": pct(dig(cu, "totals", "discounted_revenue_share")),
    }


# ------------------------------------------------------------------ render


def render_readme(f: dict[str, Any]) -> str:
    """README.md text (results first)."""
    h = headline(f)
    beats = dig(f["forecast"], "best_beats_naive_overall_wape")
    hero = (
        f"![Payback curve: revenue and margin per article-week by discount band]({f['hero']})"
        if f["hero"] and "payback" in f["hero"]
        else (f"![Dashboard]({f['hero']})" if f["hero"] else "")
    )
    dash_note = (
        "Dashboard PDF: [`dashboard/dashboard.pdf`](dashboard/dashboard.pdf)."
        if f["has_pdf"]
        else "Power BI: star-schema export, DAX measures and a 5-page "
        "[build guide](dashboard/BUILD_GUIDE.md) in [`dashboard/`](dashboard/)."
    )
    # Unfinished optional pieces (timed effort study) are left out rather than shown as gaps.
    has_effort = dig(f["effort"], "effort_reduction") is not None
    effort_kr = (
        f"; timed review vs manual tagging saved **{h['effort']}** of the effort."
        if has_effort
        else "."
    )
    effort_q5 = f" Timed effort saving: {h['effort']}." if has_effort else ""
    fc_line = (
        f"{h['fc_best']} forecasts 12 weeks ahead at **{h['fc_best_mape']} MAPE vs "
        f"{h['fc_naive_mape']}** for a seasonal-naive baseline"
        if beats
        else f"{h['fc_best']} does **not** beat the seasonal-naive baseline overall"
    )
    return f"""# Discount & Demand Analytics for Fashion E-commerce

Pricing, customer and demand analytics on {h["transactions"]} H&M fashion transactions and
{h["myntra_listings"]} Myntra listings: where deeper markdowns stop paying back, who is about to
churn and what to offer them, how demand will move, and how far product tagging can be automated.

> **Key results**
> * **Markdown ceiling: {h["ceiling_band"]} (about {h["ceiling_discount"]} off).** No discount
>   band beat full-price margin per article-week; once an item is marked down, going deeper
>   than {h["ceiling_band"]} cut margin sharply (same band under 50-70% margin assumptions).
> * **Forecasting:** {fc_line} (WAPE {h["fc_best_wape"]} vs {h["fc_naive_wape"]}), winning in
>   {h["fc_beats"]} of {h["fc_n"]} categories.
> * **Product tagging:** {h["tag_cat"]} category and {h["tag_sub"]} product-type accuracy on
>   held-out styles{effort_kr}
> * **Customers:** {h["winback_customers"]} churned or at-risk customers with
>   **{h["at_stake"]} indexed revenue at stake**; win-back offers capped at the markdown ceiling.

{hero}

{dash_note}

---

## Business questions and answers

**1. Discount payback: at what depth do deeper markdowns stop paying back?**
Across {h["articles_analysed"]} articles, discounted weeks sold about as many units as the
same article's full-price weeks nearby (elasticity {h["elasticity"]}, 95% CI
{h["elasticity_lo"]} to {h["elasticity_hi"]}, with article, week and article-age effects), so
no band beat full price on margin (best band vs full price: {h["vs_full_band"]}). Among discount
levels, margin per week peaks in the **{h["ceiling_band"]}** band (index {h["ceiling_index"]}
of full-price margin) and falls from {h["next_band"]} onwards. Myntra's cross-sectional revenue
proxy also peaks at {h["myntra_peak"]}.
→ [notebook](notebooks/02_discount_payback.ipynb) ·
[pricing recommendation](reports/pricing_recommendation.md)

**2. Performance KPIs: how are revenue, orders, AOV and units moving?**
Monthly revenue, orders, AOV and customers with month-on-month change, category mix and the top
20 articles and styles. Weekly sales peak in late June and late November; about
{h["discounted_share"]} of revenue is sold at a discount.
→ [notebook](notebooks/03_kpis_and_rfm.ipynb) · {dash_note}

**3. Customer health: who is high-value, at risk or lost, and what should we do?**
{h["n_customers_scored"]} customers scored on recency, frequency and monetary value into nine
segments. Champions and Loyal customers are {h["loyal_cust_share"]} of customers but
{h["loyal_rev_share"]} of revenue. The churned and at-risk segments hold
{h["winback_customers"]} customers ({h["winback_share"]}) and {h["at_stake"]} indexed revenue at
stake; each gets an action, channel, timing and KPI, with offers capped at {h["offer_cap"]}.
Average month-1 repeat rate: {h["m1_retention"]}.
→ [notebook](notebooks/03_kpis_and_rfm.ipynb) · [win-back playbook](reports/winback_playbook.md)

**4. Demand planning: weekly demand for the top categories over the next 12 weeks.**
Prophet with yearly seasonality (with and without a promo-intensity regressor), backtested over
4 rolling 12-week folds ({h["fc_test_weeks"]}) against a seasonal-naive baseline. Best spec:
{h["fc_best"]}, MAPE {h["fc_best_mape"]} / WAPE {h["fc_best_wape"]} vs {h["fc_naive_mape"]} /
{h["fc_naive_wape"]} for the baseline. It beats the baseline in {h["fc_beat_list"]}, not in the
other categories, and its 80% intervals cover {h["fc_coverage"]} of actuals.
→ [notebook](notebooks/04_forecasting.ipynb)

**5. Catalogue ops: can a model tag products well enough to replace most manual tagging?**
A TF-IDF + {h["tag_model"]} tagger trained on {h["tag_articles"]} H&M articles (XGBoost was
compared and came close), split by style to avoid leakage. On {h["tag_test"]} held-out
articles: category {h["tag_cat"]} (baseline {h["tag_cat_base"]}), product type
{h["tag_sub"]} (baseline {h["tag_sub_base"]}), pattern {h["tag_pat"]}, colour {h["tag_col"]}
(texts rarely name the colour). Auto-accepting suggestions at ≥90% confidence covers
{h["tag_cat_auto_cov"]} of products for category at {h["tag_cat_auto_acc"]} accuracy, and
{h["tag_sub_auto_cov"]} for product type at {h["tag_sub_auto_acc"]}.{effort_q5}
→ [notebook](notebooks/05_product_tagging.ipynb)

---

## Data

| Dataset | Use | Source and licence |
|---|---|---|
| H&M Personalized Fashion Recommendations | {h["transactions_exact"]} transactions, {h["customers"]} customers, {h["date_min"]} to {h["date_max"]}; article labels for tagging | Kaggle competition `h-and-m-personalized-fashion-recommendations`, non-commercial / academic use. Accept the rules on Kaggle once. |
| Myntra products | {h["myntra_listings"]} listings (price, MRP, rating count); discount cross-check and tagging | Kaggle dataset `ronakbokaria/myntra-products-dataset` (May 2023 snapshot). |

No raw data is committed: `make data` downloads it. H&M prices are normalised, so revenue is
**indexed revenue** (price × a fixed factor).

## Methods and limitations

* **Discounts.** Reference price = 95th percentile of an article's price over a trailing 8-week
  window; discount = 1 − price ÷ reference. Band indexes compare each article with its own
  full-price weeks within ±8 weeks (not over the article's whole life,
  because markdowns come late in an article's life; the lifetime version is reported as a
  sensitivity check). Elasticity uses article, week and article-age fixed effects.
* **Customers.** RFM scores by percentile rank (ties share a score); churn = no purchase in
  120 days, with 90 and 180 shown; an order = one customer, one day, one channel.
* **Forecasting.** Rolling-origin backtest, MAPE and WAPE by horizon, always against a
  seasonal-naive baseline.
* **Tagging.** TF-IDF text features with logistic regression and XGBoost, run
  locally at no cost; train / test split by style.
* **Limitations.** Markdowns are not randomised, so discount results are associations, not
  causal effects; there is no cost data (margins are 50/60/70% scenarios) and no inventory
  data; weeks with zero sales are unobserved; under two years of history for yearly
  seasonality, and the forecast test window includes the 2020 store closures; colour cannot be
  tagged from text; Myntra titles are much shorter than H&M descriptions.

## Reproduce

```bash
make setup                      # Python 3.11 venv + dependencies + pre-commit
cp .env.example .env            # Kaggle credentials (or ~/.kaggle/kaggle.json)
make all                        # dev mode: 10% customer sample, minutes
FDDA_MODE=full make all         # final numbers on all data
.venv/bin/python -m fdda.report.readme   # regenerate this README from reports/
make test && make lint
```

`config/settings.yaml` holds every parameter (paths, seed, bands, windows, horizons).
Product tagging also needs the OpenMP runtime on macOS (`brew install libomp`) for XGBoost.

## Repo map

```
config/settings.yaml      all parameters
src/fdda/data/            download + clean (DuckDB -> Parquet)
src/fdda/features/        reference price, discount bands, RFM
src/fdda/analysis/        EDA, discount payback, KPIs, cohorts, win-back, memo
src/fdda/forecast/        weekly series, Prophet, backtest
src/fdda/tagging/         TF-IDF + XGBoost / logistic regression tagger, effort study
src/fdda/export/          Power BI star schema
src/fdda/report/          this README and the resume bullets
notebooks/01-05           narrative notebooks (they only call src/)
reports/                  tables, figures, pricing memo, win-back playbook
dashboard/                DAX measures, build guide, theme, data dictionary
tests/                    pytest suite on a synthetic fixture
```

## What I'd do next

* **A/B test the markdown ceiling:** randomise 0 / 10 / 20 / 30% markdowns across matched stores
  or customer cohorts for articles at the same lifecycle stage.
* **Price optimisation** with inventory and cost data, valuing clearance properly.
* **Hierarchical forecasting** (category → product type → article) with a longer history.
* **Image-based tagging** for colour and pattern.

*I build this README with `python -m fdda.report.readme` from the files in `reports/`, so every
number matches the analysis.* · [Repository]({REPO_URL})
"""


def render_resume(f: dict[str, Any]) -> str:
    """Resume bullets, filled from pipeline outputs."""
    h = headline(f)
    effort = (
        f", cutting\n  manual tagging time by {h['effort']} in a timed test."
        if dig(f["effort"], "effort_reduction") is not None
        else "."
    )
    return f"""# Resume bullets

*Produced by `python -m fdda.report.readme`.*

**Discount & Demand Analytics for Fashion E-commerce** · Python, DuckDB, Prophet, XGBoost,
Power BI · {REPO_URL}

- Analysed {h["transactions"]} H&M fashion transactions and {h["myntra_listings"]} Myntra listings
  to find where deeper markdowns stop paying back: no discount beat full-price margin, and
  beyond ~{h["ceiling_discount"]} ({h["ceiling_band"]}) margin fell sharply (fixed-effects
  elasticity {h["elasticity"]}); wrote a pricing and promotion recommendation.
- Built RFM segments, cohort retention and a win-back plan for {h["winback_customers"]}
  churned / at-risk customers ({h["at_stake"]} indexed revenue at stake) with offers capped at
  the discount ceiling, plus a 5-page Power BI dashboard model (star schema, DAX measures).
- Forecast 12-week demand for the top {h["fc_n"]} categories at {h["fc_best_mape"]} MAPE vs
  {h["fc_naive_mape"]} for a seasonal-naive baseline; built an ML product tagger
  ({h["tag_cat"]} category / {h["tag_sub"]} product-type accuracy on held-out styles){effort}
"""


def main() -> None:
    """Write README.md and reports/resume_bullets.md."""
    settings = load_settings()
    facts = collect(settings)
    root = Path(settings["_root"])
    (root / "README.md").write_text(render_readme(facts), encoding="utf-8")
    (resolve_path(settings, "reports") / "resume_bullets.md").write_text(
        render_resume(facts), encoding="utf-8"
    )
    missing = [k for k, v in facts.items() if v is None and k not in ("hero",)]
    log.info(
        "Wrote README.md and reports/resume_bullets.md (missing inputs: %s)", missing or "none"
    )


if __name__ == "__main__":
    main()
