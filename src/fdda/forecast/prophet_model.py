"""Prophet wrapper for weekly category demand.

Yearly seasonality only (weekly data), optional `promo_share` regressor. Predictions are
clipped at zero. Prophet and cmdstanpy log a lot; this module quiets them.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

for _name in ("cmdstanpy", "prophet", "prophet.plot"):
    logging.getLogger(_name).setLevel(logging.ERROR)


def fit_predict(
    train: pd.DataFrame,
    future: pd.DataFrame,
    params: dict[str, Any],
    use_promo: bool = False,
) -> pd.DataFrame:
    """Fit Prophet on `train` (ds, y[, promo_share]) and predict for `future` (ds[, promo_share]).

    Returns ds, yhat, yhat_lower, yhat_upper (non-negative).
    """
    from prophet import Prophet  # imported lazily: heavy, and only needed here

    model = Prophet(
        yearly_seasonality=params.get("yearly_seasonality", 6),
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=params.get("seasonality_mode", "multiplicative"),
        changepoint_prior_scale=params.get("changepoint_prior_scale", 0.05),
        interval_width=params.get("interval_width", 0.8),
        uncertainty_samples=params.get("uncertainty_samples", 300),
    )
    cols = ["ds", "y"]
    if use_promo:
        model.add_regressor("promo_share", mode="multiplicative")
        cols.append("promo_share")
    model.fit(train[cols])
    fut_cols = ["ds", "promo_share"] if use_promo else ["ds"]
    pred = model.predict(future[fut_cols])[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    num = ["yhat", "yhat_lower", "yhat_upper"]
    pred[num] = pred[num].clip(lower=0)
    return pred
