# Pricing & promotion recommendation

*Produced by `python -m fdda.analysis.discount` (dev data). Every number is read
from `reports/discount/*_dev.csv` and `discount_summary_dev.json`.*

## Summary

Across 86,085 H&M articles and 3,157,777 transactions, **no discount band earned more margin per article-week than full price** under any of the assumed gross margins. Once an article is marked down, **margin per week peaks in the 5-15% band (median discount 10%) and falls sharply beyond it (15-25% onwards)**, the same under all 50%–70% margin assumptions. Myntra's cross-sectional revenue proxy also peaks in the 15-25% band.

## Recommendations

1. **Cap routine markdowns at the 5-15% band (about 10% off).** Going deeper cut margin per article-week from 0.82 to 0.67 of full-price margin in the next band (60%-margin scenario) without a matching lift in units.
2. **Reserve discounts beyond 5-15% for clearance.** At 55%+ the margin index is -0.00 (60% margin). Deep cuts can still be right for end-of-season stock with no other exit, but that is an inventory decision (holding cost, salvage value) this data cannot value, not a demand-generation tool.
3. **Use the same cap across categories:** every category with enough data peaks at 5-15% or deeper (table below).
4. **Tie promotions together:** win-back offers in the customer win-back playbook are capped at the same 5-15% band.
5. **Validate with a controlled test** before rolling out (design below).

## What was measured

* **Data:** H&M transactions, 3,157,777 units over 925,503
  article-weeks (86,085 articles). Prices are normalised, so revenue is an index.
* **Discount:** 1 − price ÷ reference price, where the reference is the 95th percentile of the
  article's price over a trailing 8-week window (30%
  of article-weeks fall back to the article's lifetime max because the window is thin).
* **Comparison:** for each article, units, revenue and margin per week in a discount band divided
  by the same article's full-price weeks within ±8 weeks (a *local* baseline, because
  markdowns come late in an article's life when demand has already faded). Medians across
  articles.
* **Margin:** no cost data exists, so unit cost = (1 − m) × reference price for assumed
  full-price gross margins m = 50%, 60%, 70%.
* **Ceiling:** the last band before each extra discount point *lowers* the margin index.

## Results by discount band (all articles, local baseline)

1.00 = the same per week as the article at full price.

| Band | Articles | Median discount | Units / week | Revenue / week | Margin / week (50%) | Margin / week (60%) | Margin / week (70%) |
|---|---|---|---|---|---|---|---|
| 0-5% | 59,259 | 0% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 5-15% | 40,640 | 10% | 0.98 | 0.88 | 0.79 | 0.82 | 0.84 |
| 15-25% | 34,267 | 20% | 1.00 | 0.80 | 0.61 | 0.67 | 0.71 |
| 25-35% | 18,877 | 29% | 1.00 | 0.70 | 0.41 | 0.50 | 0.57 |
| 35-45% | 17,710 | 40% | 1.00 | 0.68 | 0.25 | 0.40 | 0.50 |
| 45-55% | 17,532 | 50% | 1.00 | 0.57 | -0.00 | 0.21 | 0.34 |
| 55%+ | 16,004 | 60% | 1.00 | 0.42 | -0.25 | -0.00 | 0.14 |

* **Discounts barely lift weekly volume.** Units per week stay within 0.98–1.00× of full price
  across bands, so revenue and margin fall with depth.
* **Elasticity** (log units on log price ratio, article + week + article-age fixed effects):
  β = -0.05 [95% CI -0.06, -0.04],
  i.e. a price 10% further below the reference price is associated with about
  0.5% more units. Without the age control β =
  0.01: lifecycle decay hides part of the discount effect.
  Under a constant elasticity of this size, the implied margin-optimal discount is
  0%.
* **Sensitivity:** with a *lifetime* baseline the markdown ceiling is
  5-15% (the same), and discounted bands sit further below full price, because late-life markdown weeks are compared with launch weeks.

## Ceiling by category (once marked down)

| Category | Ceiling (60% margin) | Margin index at ceiling | Ceiling at 50% | Ceiling at 70% | Elasticity β [95% CI] |
|---|---|---|---|---|---|
| Socks & Tights | 5-15% | 0.85 | 5-15% | 5-15% | 0.04 [-0.01, 0.08] |
| Accessories | 5-15% | 0.84 | 5-15% | 5-15% | -0.06 [-0.08, -0.04] |
| Shoes | 5-15% | 0.82 | 5-15% | 5-15% | -0.09 [-0.13, -0.06] |
| Garment Full body | 5-15% | 0.80 | 5-15% | 5-15% | -0.08 [-0.10, -0.06] |
| Garment Lower body | 5-15% | 0.80 | 5-15% | 5-15% | -0.05 [-0.07, -0.03] |
| **All articles** | 5-15% | 0.82 | 5-15% | 5-15% | -0.05 [-0.06, -0.04] |
| Garment Upper body | 5-15% | 0.82 | 5-15% | 5-15% | -0.10 [-0.11, -0.08] |
| Swimwear | 5-15% | 0.75 | 5-15% | 5-15% | 0.12 [0.08, 0.16] |
| Nightwear | 5-15% | 0.83 | 5-15% | 5-15% | -0.20 [-0.28, -0.11] |
| Underwear | 5-15% | 0.78 | 5-15% | 5-15% | 0.00 [-0.03, 0.04] |

Elasticity β < 0 means deeper discounts go with more units. Positive values (seen in seasonal
categories) reflect end-of-season clearance: the items discounted in a given week are the ones
that are not selling.

## Myntra cross-check (directional)

Mean per listing relative to the 0–5% band. Different products at different discounts in one
snapshot; rating count proxies cumulative demand. Direction only, not an estimate.

| Band | Listings | Share with ratings | Rating count index | Price × rating count index |
|---|---|---|---|---|
| 0-5% | 20,087 | 20% | 1.00 | 1.00 |
| 5-15% | 5,848 | 22% | 0.81 | 0.32 |
| 15-25% | 6,609 | 26% | 1.96 | 1.15 |
| 25-35% | 8,150 | 25% | 1.29 | 0.73 |
| 35-45% | 9,994 | 27% | 1.84 | 0.90 |
| 45-55% | 14,863 | 30% | 0.79 | 0.44 |
| 55%+ | 40,538 | 27% | 0.64 | 0.31 |

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
5-15% ceiling holds once selection and timing are removed by design.

## Reproduce

```bash
make clean-data            # FDDA_MODE=full for all customers
FDDA_MODE=full make discount
FDDA_MODE=full make payback-notebook   # notebooks/02_discount_payback.ipynb
```
