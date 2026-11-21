# Power BI data dictionary

*Produced by `python -m fdda.export.powerbi` (full data, parquet). Files are written to `data/processed/powerbi/` (git-ignored).*

| Table | Rows | Size (MB) | Description |
|---|---|---|---|
| `fact_sales` | 4,877,177 | 72.44 | Units and indexed revenue by week x article x channel x discount band. |
| `dim_article` | 105,542 | 2.53 | H&M article attributes (one row per article_id). |
| `dim_product_group` | 19 | 0.0 | Product groups; links dim_article and fact_forecast. |
| `dim_channel` | 2 | 0.0 | Sales channels. H&M does not name them; channel 2 is commonly read as online. |
| `dim_discount_band` | 7 | 0.0 | Discount bands; is_ceiling_band marks the markdown ceiling. |
| `dim_customer_segment` | 9 | 0.01 | RFM segments with their R/F rules, size and win-back playbook. |
| `fact_kpi_monthly` | 25 | 0.01 | Monthly orders, active customers and AOV (not additive across months). |
| `fact_customer` | 1,362,281 | 31.08 | One row per customer: RFM, segment, churn flags (customer_key is surrogate). |
| `fact_cohort_retention` | 325 | 0.01 | Share of each acquisition cohort buying k months later. |
| `fact_forecast` | 2,272 | 0.05 | Weekly units: actuals, backtest forecasts and the 12-week forward forecast. |
| `fact_forecast_metrics` | 108 | 0.01 | Backtest MAPE and WAPE by category, model and horizon bucket. |
| `fact_discount_bands` | 140 | 0.02 | Within-article indexes by category and discount band. |
| `fact_discount_thresholds` | 200 | 0.01 | Payback ceilings by category, measure and reference. |
| `dim_date` | 835 | 0.02 | Calendar, one row per day. Weekly facts use the week's Monday; monthly facts the 1st. |

## Check totals

With no filters applied, the dashboard's cards should show these values.

| Measure | Expected |
|---|---|
| Units (sum of fact_sales[units]) | 31,788,324 |
| Revenue (indexed) | 884,645,974 |
| Full-price value | 1,015,920,777 |
| Avg Discount % (1 - revenue / full-price value) | 12.9% |
| Discounted Revenue Share | 32.3% |
| Customers (rows of fact_customer) | 1,362,281 |

## fact_sales

| Column | Type |
|---|---|
| `date_key` | int64 |
| `article_id` | str |
| `channel_id` | int32 |
| `discount_band` | str |
| `units` | int64 |
| `indexed_revenue` | float64 |
| `full_price_value` | float64 |
| `discounted_units` | int64 |
| `discounted_revenue` | float64 |

## dim_article

| Column | Type |
|---|---|
| `article_id` | str |
| `product_code` | str |
| `prod_name` | str |
| `product_type_name` | str |
| `product_group_name` | str |
| `garment_group_name` | str |
| `section_name` | str |
| `colour_group_name` | str |
| `graphical_appearance_name` | str |

## dim_product_group

| Column | Type |
|---|---|
| `product_group_name` | str |
| `units_rank` | int64 |
| `is_forecast_category` | bool |

## dim_channel

| Column | Type |
|---|---|
| `channel_id` | int64 |
| `channel_name` | str |

## dim_discount_band

| Column | Type |
|---|---|
| `discount_band` | str |
| `band_order` | int64 |
| `lower_edge` | float64 |
| `upper_edge` | float64 |
| `is_full_price` | bool |
| `is_ceiling_band` | bool |

## dim_customer_segment

| Column | Type |
|---|---|
| `segment` | str |
| `segment_order` | int64 |
| `r_score_range` | str |
| `f_score_range` | str |
| `customers` | int64 |
| `customer_share` | float64 |
| `median_recency_days` | float64 |
| `avg_frequency` | float64 |
| `avg_monetary` | float64 |
| `monetary_share` | float64 |
| `revenue_at_stake` | float64 |
| `churned_share` | float64 |
| `offer_discount` | float64 |
| `responds_to_band` | str |
| `favourite_categories` | str |
| `action` | str |
| `channel` | str |
| `timing` | str |
| `kpi` | str |
| `is_winback_segment` | bool |

## fact_kpi_monthly

| Column | Type |
|---|---|
| `date_key` | int64 |
| `month` | str |
| `indexed_revenue` | float64 |
| `orders` | int64 |
| `units` | int64 |
| `customers` | int64 |
| `discounted_revenue_share` | float64 |
| `discounted_unit_share` | float64 |
| `aov` | float64 |
| `units_per_order` | float64 |
| `is_partial_month` | bool |
| `revenue_mom` | float64 |
| `orders_mom` | float64 |
| `units_mom` | float64 |
| `aov_mom` | float64 |
| `customers_mom` | float64 |

## fact_customer

| Column | Type |
|---|---|
| `customer_key` | int64 |
| `segment` | str |
| `r_score` | int32 |
| `f_score` | int32 |
| `m_score` | int32 |
| `recency_days` | int64 |
| `frequency` | int64 |
| `orders` | int64 |
| `units` | int64 |
| `indexed_revenue` | float64 |
| `revenue_at_stake` | float64 |
| `channel2_share` | float64 |
| `first_purchase_month` | str |
| `last_purchase_date_key` | int64 |
| `churned_90d` | bool |
| `churned_120d` | bool |
| `churned_180d` | bool |

## fact_cohort_retention

| Column | Type |
|---|---|
| `cohort` | str |
| `months_since` | int64 |
| `customers` | int64 |
| `cohort_size` | int64 |
| `retention` | float64 |

## fact_forecast

| Column | Type |
|---|---|
| `date_key` | int64 |
| `category` | str |
| `kind` | str |
| `model` | str |
| `units` | object |
| `yhat` | float64 |
| `yhat_lower` | float64 |
| `yhat_upper` | float64 |
| `fold` | float64 |
| `h` | float64 |
| `promo_share` | float64 |

## fact_forecast_metrics

| Column | Type |
|---|---|
| `category` | str |
| `model` | str |
| `bucket` | str |
| `mape` | float64 |
| `wape` | float64 |
| `n` | int64 |

## fact_discount_bands

| Column | Type |
|---|---|
| `category` | str |
| `band` | str |
| `n_articles` | int64 |
| `band_discount` | float64 |
| `units_index` | float64 |
| `revenue_index` | float64 |
| `margin_index_50` | float64 |
| `margin_index_60` | float64 |
| `margin_index_70` | float64 |
| `units_index_p25` | float64 |
| `units_index_p75` | float64 |
| `baseline` | str |

## fact_discount_thresholds

| Column | Type |
|---|---|
| `category` | str |
| `measure` | str |
| `reference` | str |
| `ceiling_band` | str |
| `ceiling_discount` | float64 |
| `ceiling_index` | float64 |
| `turns_negative` | bool |
| `first_negative_band` | str |
| `marginal_at_turn` | float64 |
| `peak_band` | str |
| `peak_index` | float64 |
| `baseline` | str |

## dim_date

| Column | Type |
|---|---|
| `date` | datetime64[us] |
| `date_key` | int64 |
| `year` | int32 |
| `quarter` | str |
| `month_num` | int32 |
| `month_name` | str |
| `year_month` | str |
| `month_start` | datetime64[us] |
| `week_start` | datetime64[us] |
| `iso_week` | int64 |
| `is_week_start` | bool |
| `in_sales_data` | bool |
| `is_forecast_period` | bool |
