# Discount & Demand Analytics for Fashion E-commerce

Discount payback, RFM/churn, demand forecasting and LLM product tagging for fashion e-commerce, using H&M transactions and Myntra listings (Python, DuckDB, Prophet, Power BI).

🚧 In progress. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full plan.

## Data

No raw data is committed. `make data` downloads it from Kaggle:

| Dataset | Source | License / note |
|---|---|---|
| H&M Personalized Fashion Recommendations | Kaggle competition `h-and-m-personalized-fashion-recommendations` (tabular files only) | Non-commercial / academic use. Accept the competition rules once on Kaggle before downloading. |
| Myntra products | Kaggle dataset `ronakbokaria/myntra-products-dataset` | Listing snapshot (price, MRP, rating, rating count); no transactions or dates. |

H&M prices are normalized, so revenue is reported as **indexed revenue** (price × a fixed factor set in `config/settings.yaml`).

## Reproduce

Requires Python 3.11 and a Kaggle account.

```bash
make setup                 # create .venv, install deps, pre-commit hooks
cp .env.example .env       # then add KAGGLE_USERNAME and KAGGLE_KEY
make data                  # download + clean -> data/interim/<dev|full>/
make test                  # tests run on a small synthetic fixture
```

`dev_mode: true` in `config/settings.yaml` (the default) samples 10% of customers so every step runs in minutes. Set it to `false`, or run `python -m fdda.data.clean --full`, for final numbers. Cleaning row counts and date ranges are written to `reports/data_summary_<dataset>_<mode>.json`.
