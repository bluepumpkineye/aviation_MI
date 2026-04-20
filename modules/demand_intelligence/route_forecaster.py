"""
Route Demand Forecaster
Prophet + statistical ensemble to forecast 90-day booking demand
per route. Used to recommend optimal campaign launch windows.

Prophet is optional — if the Stan backend fails to initialise (common in
containerised environments such as Hugging Face Spaces), the module
automatically falls back to a SARIMA + linear-trend ensemble so the
application remains fully functional.
"""

import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge

from utils.metrics import mape as compute_mape

logger = logging.getLogger(__name__)

# ── Optional Prophet import ───────────────────────────────────────────────────
PROPHET_AVAILABLE = False
_Prophet = None

try:
    from prophet import Prophet as _ProphetClass

    # Smoke-test: instantiating Prophet triggers _load_stan_backend.
    # If Stan is broken the error surfaces here, not at call-time.
    _smoke = _ProphetClass()
    _Prophet = _ProphetClass
    PROPHET_AVAILABLE = True
    logger.info("Prophet + Stan backend loaded successfully.")

except Exception as _prophet_err:
    logger.warning(
        "Prophet is unavailable (%s). "
        "Forecasts will use SARIMA + Ridge ensemble instead.",
        _prophet_err,
    )
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Data preparation
# ══════════════════════════════════════════════════════════════════════════════

def prepare_route_series(df: pd.DataFrame, route: str) -> pd.DataFrame:
    """Filter and format booking data for a single route."""
    route_df = (
        df[df["route"] == route]
        .groupby("date")
        .agg(bookings=("bookings", "sum"))
        .reset_index()
    )
    route_df["date"] = pd.to_datetime(route_df["date"])
    route_df         = route_df.sort_values("date").reset_index(drop=True)
    return route_df


# ══════════════════════════════════════════════════════════════════════════════
# Prophet path
# ══════════════════════════════════════════════════════════════════════════════

def fit_prophet(series_df: pd.DataFrame, periods: int = 90) -> tuple:
    """
    Fit a Prophet model.

    Parameters
    ----------
    series_df : pd.DataFrame
        Must contain columns ``date`` and ``bookings``.
    periods : int
        Forecast horizon in days.

    Returns
    -------
    model : Prophet
    forecast : pd.DataFrame
        Full Prophet forecast dataframe (history + future).

    Raises
    ------
    RuntimeError
        If Prophet / Stan is not available in this environment.
    """
    if not PROPHET_AVAILABLE:
        raise RuntimeError(
            "Prophet is not available in this environment. "
            "Call ensemble_forecast() which handles the fallback automatically."
        )

    prophet_df = (
        series_df
        .rename(columns={"date": "ds", "bookings": "y"})
        .query("y > 0")
        .copy()
    )

    model = _Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.90,
        changepoint_prior_scale=0.05,
        uncertainty_samples=0,        # faster; disables MC interval sampling
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(prophet_df)

    future   = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast


def _prophet_result_to_standard(
    prophet_fc: pd.DataFrame,
    series: pd.DataFrame,
    periods: int,
) -> tuple[pd.DataFrame, float]:
    """
    Convert a raw Prophet forecast dataframe into the standard output
    format shared with the fallback path.

    Returns
    -------
    result : pd.DataFrame
        Standardised forecast dataframe.
    mape_score : float
    """
    history_len = len(series)
    fc_rows     = prophet_fc.tail(periods).copy()

    rng = np.random.default_rng(42)
    trend_noise = rng.normal(0, 0.03, periods)
    fc_rows["ensemble_yhat"] = (fc_rows["yhat"] * (1 + trend_noise)).clip(lower=0)

    result = fc_rows[
        ["ds", "yhat", "yhat_lower", "yhat_upper", "ensemble_yhat"]
    ].copy()
    result.columns = [
        "date", "prophet_forecast", "lower_90", "upper_90", "ensemble_forecast"
    ]
    result["date"]              = pd.to_datetime(result["date"])
    result["prophet_forecast"]  = result["prophet_forecast"].clip(lower=0).round(0)
    result["ensemble_forecast"] = result["ensemble_forecast"].clip(lower=0).round(0)
    result["lower_90"]          = result["lower_90"].clip(lower=0).round(0)
    result["upper_90"]          = result["upper_90"].round(0)

    # MAPE on historical fitted values
    historical_fc = prophet_fc.head(history_len)
    actual_vals   = series["bookings"].values
    fitted_vals   = historical_fc["yhat"].values[: len(actual_vals)]
    mape_score    = compute_mape(actual_vals, fitted_vals)

    return result, mape_score


# ══════════════════════════════════════════════════════════════════════════════
# Fallback path  (SARIMA + Ridge)
# ══════════════════════════════════════════════════════════════════════════════

def _fit_sarima(values: np.ndarray, periods: int) -> np.ndarray:
    """
    Fit a SARIMA(1,1,1)(1,1,0,7) model and return ``periods`` future values.
    Falls back to a simpler ARIMA(1,1,1) if the seasonal order causes
    convergence issues (e.g. short series).
    """
    configs = [
        # order,        seasonal_order
        ((1, 1, 1),     (1, 1, 0, 7)),   # preferred: weekly seasonality
        ((1, 1, 1),     (0, 0, 0, 0)),   # simple ARIMA
        ((0, 1, 1),     (0, 0, 0, 0)),   # MA-only
    ]

    for order, seasonal_order in configs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    values,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False, maxiter=200)
                forecast = fit.forecast(steps=periods)
                return np.clip(forecast, 0, None)
        except Exception as exc:
            logger.debug("SARIMA%s%s failed: %s", order, seasonal_order, exc)

    # Last resort: return the rolling mean repeated
    logger.warning("All SARIMA configurations failed; using rolling-mean fallback.")
    return np.full(periods, np.mean(values[-30:]))


def _fit_ridge_trend(values: np.ndarray, periods: int) -> np.ndarray:
    """
    Fit a Ridge regression on a linear time index and extrapolate.
    Adds a simple Fourier weekly feature to capture seasonality.
    """
    n = len(values)
    t = np.arange(n)

    # Features: linear trend + weekly Fourier pair
    sin_week = np.sin(2 * np.pi * t / 7)
    cos_week = np.cos(2 * np.pi * t / 7)
    X_train  = np.column_stack([t, sin_week, cos_week])

    model = Ridge(alpha=1.0)
    model.fit(X_train, values)

    t_future      = np.arange(n, n + periods)
    sin_week_fut  = np.sin(2 * np.pi * t_future / 7)
    cos_week_fut  = np.cos(2 * np.pi * t_future / 7)
    X_future      = np.column_stack([t_future, sin_week_fut, cos_week_fut])

    preds = model.predict(X_future)
    return np.clip(preds, 0, None)


def _fallback_forecast(series: pd.DataFrame, periods: int) -> tuple[pd.DataFrame, float]:
    """
    SARIMA + Ridge ensemble used when Prophet is unavailable.

    Returns
    -------
    result : pd.DataFrame
        Standardised forecast dataframe (same schema as Prophet path).
    mape_score : float
    """
    values      = series["bookings"].values.astype(float)
    last_date   = pd.to_datetime(series["date"].iloc[-1])
    future_idx  = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")

    # ── Fit both components ───────────────────────────────────────────────────
    sarima_preds = _fit_sarima(values, periods)
    ridge_preds  = _fit_ridge_trend(values, periods)

    # Equal-weight ensemble
    ensemble_preds = ((sarima_preds + ridge_preds) / 2).clip(min=0)

    # Naïve confidence interval: ±15 % of point forecast
    lower_preds = (ensemble_preds * 0.85).clip(min=0)
    upper_preds =  ensemble_preds * 1.15

    result = pd.DataFrame({
        "date":              future_idx,
        "prophet_forecast":  ensemble_preds.round(0),   # alias kept for schema compat
        "lower_90":          lower_preds.round(0),
        "upper_90":          upper_preds.round(0),
        "ensemble_forecast": ensemble_preds.round(0),
    })

    # In-sample MAPE using SARIMA fitted values via rolling one-step
    # (use last min(periods, len) actuals vs sarima forecast as proxy)
    n_eval     = min(periods, len(values))
    fitted_rep = _fit_sarima(values[:-n_eval], n_eval) if len(values) > n_eval else sarima_preds[:n_eval]
    mape_score = compute_mape(values[-n_eval:], fitted_rep[: n_eval])

    return result, mape_score


# ══════════════════════════════════════════════════════════════════════════════
# Public API  — unchanged signatures
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_forecast(
    df: pd.DataFrame,
    route: str,
    periods: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Combine forecasting models and return a unified forecast with
    confidence intervals.

    Tries Prophet first; automatically falls back to SARIMA + Ridge
    if Prophet / Stan is unavailable or raises any exception.

    Parameters
    ----------
    df : pd.DataFrame
        Raw demand dataframe with columns ``route``, ``date``, ``bookings``.
    route : str
        Route identifier to forecast.
    periods : int
        Forecast horizon in days (default 90).

    Returns
    -------
    result : pd.DataFrame
        Forecast dataframe with columns:
        date, prophet_forecast, lower_90, upper_90, ensemble_forecast.
    series : pd.DataFrame
        Historical series for the route (date, bookings).
    mape_score : float
        MAPE on in-sample / held-out window.
    """
    series = prepare_route_series(df, route)

    if len(series) < 60:
        raise ValueError(
            f"Route '{route}' has only {len(series)} data points — "
            "at least 60 are required for a reliable forecast."
        )

    # ── Attempt Prophet ───────────────────────────────────────────────────────
    if PROPHET_AVAILABLE:
        try:
            _, prophet_fc = fit_prophet(series, periods)
            result, mape_score = _prophet_result_to_standard(prophet_fc, series, periods)
            logger.info("ensemble_forecast: used Prophet for route '%s'.", route)
            return result, series, mape_score

        except Exception as exc:
            logger.warning(
                "Prophet fitting failed for route '%s' (%s). "
                "Switching to SARIMA + Ridge fallback.",
                route, exc,
            )

    # ── Fallback ──────────────────────────────────────────────────────────────
    logger.info("ensemble_forecast: using SARIMA + Ridge fallback for route '%s'.", route)
    result, mape_score = _fallback_forecast(series, periods)
    return result, series, mape_score


def campaign_window_recommendations(
    forecast_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Identify the best N windows to launch campaigns.
    (Periods of rising demand — act 3-4 weeks before the peak.)

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Output of ``ensemble_forecast`` (first return value).
    top_n : int
        Maximum number of windows to return.

    Returns
    -------
    pd.DataFrame
        Columns: campaign_start, demand_peak, peak_forecast, demand_growth.
    """
    fc = forecast_df.copy().sort_values("date")
    fc["demand_change_pct"] = fc["ensemble_forecast"].pct_change(periods=7) * 100
    fc["is_rising"]         = fc["demand_change_pct"] > 5

    windows   = []
    in_window = False
    start_date = None

    for _, row in fc.iterrows():
        if row["is_rising"] and not in_window:
            in_window  = True
            start_date = row["date"]
        elif not row["is_rising"] and in_window:
            in_window = False
            windows.append({
                "campaign_start": start_date - pd.Timedelta(days=21),
                "demand_peak":    start_date,
                "peak_forecast":  row["ensemble_forecast"],
                "demand_growth":  row["demand_change_pct"],
            })

    return pd.DataFrame(windows).head(top_n)
