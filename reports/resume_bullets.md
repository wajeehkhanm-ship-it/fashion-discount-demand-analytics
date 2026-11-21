# Resume bullets

*Produced by `python -m fdda.report.readme`.*

**Discount & Demand Analytics for Fashion E-commerce** · Python, DuckDB, Prophet, XGBoost,
Power BI · https://github.com/wajeehkhanm-ship-it/fashion-discount-demand-analytics

- Analysed 31.8M H&M fashion transactions and 1.06M Myntra listings
  to find where deeper markdowns stop paying back: no discount beat full-price margin, and
  beyond ~20% (15-25%) margin fell sharply (fixed-effects
  elasticity -0.08); wrote a pricing and promotion recommendation.
- Built RFM segments, cohort retention and a win-back plan for 546,270
  churned / at-risk customers (113.8M indexed revenue at stake) with offers capped at
  the discount ceiling, plus a 5-page Power BI dashboard model (star schema, DAX measures).
- Forecast 12-week demand for the top 8 categories at 23.3% MAPE vs
  30.4% for a seasonal-naive baseline; built an ML product tagger
  (97.9% category / 89.1% product-type accuracy on held-out styles).
