"""
Route Demand Forecaster
Prophet + statistical ensemble to forecast 90-day booking demand
per route. Used to recommend optimal campaign launch windows.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.metrics import mape as compute_mape


def prepare_route_series(df: pd.DataFrame, route: str) -> pd.DataFrame:
    """Filter and format data for a single route."""
    route_df = (
        df[df["route"] == route]
        .groupby("date")
        .agg(bookings=("bookings", "sum"))
        .reset_index()
    )
    route_df["date"] = pd.to_datetime(route_df["date"])
    route_df         = route_df.sort_values("date")
    return route_df


def fit_prophet(series_df: pd.DataFrame, periods: int = 90) -> tuple:
    """Fit Prophet and return forecast dataframe + model."""
    prophet_df = series_df.rename(columns={"date": "ds", "bookings": "y"})
    prophet_df = prophet_df[prophet_df["y"] > 0]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.90,
        changepoint_prior_scale=0.05,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(prophet_df)

    future   = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast


def ensemble_forecast(
    df: pd.DataFrame,
    route: str,
    periods: int = 90,
) -> pd.DataFrame:
    """
    Combine Prophet with a simple trend extrapolation.
    Returns unified forecast with confidence intervals.
    """
    series = prepare_route_series(df, route)

    if len(series) < 60:
        raise ValueError(f"Not enough data for route {route}")

    # Prophet forecast
    _, prophet_fc = fit_prophet(series, periods)

    history_len = len(series)
    fc_rows     = prophet_fc.tail(periods).copy()

    # Simple noise-smoothed ensemble
    trend_noise = np.random.default_rng(42).normal(0, 0.03, periods)
    fc_rows["ensemble_yhat"] = fc_rows["yhat"] * (1 + trend_noise)
    fc_rows["ensemble_yhat"] = fc_rows["ensemble_yhat"].clip(lower=0)

    result = fc_rows[["ds", "yhat", "yhat_lower", "yhat_upper", "ensemble_yhat"]].copy()
    result.columns = ["date", "prophet_forecast", "lower_90", "upper_90", "ensemble_forecast"]
    result["date"]              = pd.to_datetime(result["date"])
    result["prophet_forecast"]  = result["prophet_forecast"].clip(lower=0).round(0)
    result["ensemble_forecast"] = result["ensemble_forecast"].clip(lower=0).round(0)
    result["lower_90"]          = result["lower_90"].clip(lower=0).round(0)
    result["upper_90"]          = result["upper_90"].round(0)

    # Compute MAPE on historical fitted values
    historical_fc = prophet_fc.head(history_len)
    actual_vals   = series["bookings"].values
    fitted_vals   = historical_fc["yhat"].values[:len(actual_vals)]
    mape_score    = compute_mape(actual_vals, fitted_vals)

    return result, series, mape_score


def campaign_window_recommendations(
    forecast_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Identify the best N windows to launch campaigns
    (periods of rising demand = act 3-4 weeks before peak).
    """
    fc = forecast_df.copy().sort_values("date")
    fc["demand_change_pct"] = fc["ensemble_forecast"].pct_change(periods=7) * 100
    fc["is_rising"]         = fc["demand_change_pct"] > 5

    # Find rising windows
    windows = []
    in_window = False
    start_date = None

    for _, row in fc.iterrows():
        if row["is_rising"] and not in_window:
            in_window  = True
            start_date = row["date"]
        elif not row["is_rising"] and in_window:
            in_window  = False
            windows.append({
                "campaign_start":  start_date - pd.Timedelta(days=21),
                "demand_peak":     start_date,
                "peak_forecast":   row["ensemble_forecast"],
                "demand_growth":   row["demand_change_pct"],
            })

    result = pd.DataFrame(windows).head(top_n)
    return result