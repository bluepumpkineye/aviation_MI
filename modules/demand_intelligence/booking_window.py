"""
Booking Window Analyser
Analyses how far in advance customers book per route and cabin.
Identifies booking lead time patterns used to time campaigns optimally.

Key insight: if most customers on HKG-LHR Business book 45-60 days
out, your campaign must launch at least 50 days before departure.
"""

import numpy as np
import pandas as pd
from config.settings import ROUTES, CABIN_CLASSES

rng = np.random.default_rng(42)


# ── Synthetic Booking Window Generator ───────────────────────────

# Realistic booking lead times per route type
ROUTE_BOOKING_PROFILES = {
    # Long haul — book well in advance
    "HKG-LHR": {"mean": 52, "std": 18, "min": 3,  "max": 180},
    "HKG-JFK": {"mean": 55, "std": 20, "min": 3,  "max": 180},
    "HKG-LAX": {"mean": 50, "std": 18, "min": 3,  "max": 180},
    "HKG-YVR": {"mean": 48, "std": 17, "min": 3,  "max": 180},
    "HKG-FRA": {"mean": 45, "std": 16, "min": 3,  "max": 150},
    "HKG-CDG": {"mean": 47, "std": 17, "min": 3,  "max": 150},

    # Medium haul
    "HKG-SYD": {"mean": 38, "std": 15, "min": 2,  "max": 120},
    "HKG-MEL": {"mean": 36, "std": 14, "min": 2,  "max": 120},
    "HKG-DXB": {"mean": 35, "std": 14, "min": 2,  "max": 120},

    # Short haul — book closer to departure
    "HKG-NRT": {"mean": 25, "std": 12, "min": 1,  "max": 90},
    "HKG-ICN": {"mean": 22, "std": 11, "min": 1,  "max": 90},
    "HKG-SIN": {"mean": 20, "std": 10, "min": 1,  "max": 90},
    "HKG-BKK": {"mean": 18, "std": 9,  "min": 1,  "max": 60},
    "HKG-TPE": {"mean": 15, "std": 8,  "min": 1,  "max": 60},
    "HKG-MNL": {"mean": 14, "std": 7,  "min": 1,  "max": 60},
}

# Cabin modifiers — Business/First books earlier
CABIN_MODIFIERS = {
    "Economy":         0.85,   # books closer to departure
    "Premium Economy": 1.00,
    "Business":        1.30,   # books furthest in advance
    "First":           1.45,
}


def generate_booking_window_data(n_records: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic booking window records.
    Each row = one booking with route, cabin, lead time, and
    whether the customer converted (used for conversion analysis).
    """
    records = []

    routes = rng.choice(ROUTES, size=n_records)
    cabins = rng.choice(
        CABIN_CLASSES, size=n_records, p=[0.60, 0.20, 0.15, 0.05]
    )

    for i, (route, cabin) in enumerate(zip(routes, cabins)):
        profile  = ROUTE_BOOKING_PROFILES.get(
            route,
            {"mean": 30, "std": 12, "min": 1, "max": 120}
        )
        modifier = CABIN_MODIFIERS.get(cabin, 1.0)

        # Draw lead time from truncated normal
        mean_days = profile["mean"] * modifier
        std_days  = profile["std"]
        lead_time = int(np.clip(
            rng.normal(mean_days, std_days),
            profile["min"],
            profile["max"]
        ))

        # Fare — higher for longer lead time (advance purchase)
        base_fare_map = {
            "Economy": 2500, "Premium Economy": 6000,
            "Business": 12000, "First": 22000,
        }
        base_fare = base_fare_map.get(cabin, 3000)
        fare_mult = 1.0 + (60 - lead_time) / 200   # last minute = higher fare
        fare      = int(base_fare * fare_mult * rng.uniform(0.85, 1.15))

        # Conversion probability — higher for committed lead times
        if lead_time <= 7:
            conv_prob = 0.85    # last minute = usually committed
        elif lead_time <= 21:
            conv_prob = 0.72
        elif lead_time <= 45:
            conv_prob = 0.58
        elif lead_time <= 90:
            conv_prob = 0.42
        else:
            conv_prob = 0.28    # very early = just browsing

        converted = int(rng.random() < conv_prob)

        records.append({
            "booking_id":      f"BW{str(i).zfill(6)}",
            "route":           route,
            "cabin":           cabin,
            "lead_time_days":  lead_time,
            "fare_hkd":        fare,
            "converted":       converted,
            "lead_time_band":  _band(lead_time),
            "is_long_haul":    int(route in [
                "HKG-LHR", "HKG-JFK", "HKG-LAX",
                "HKG-YVR", "HKG-FRA", "HKG-CDG"
            ]),
        })

    return pd.DataFrame(records)


def _band(days: int) -> str:
    """Categorise lead time into human-readable bands."""
    if days <= 7:    return "0-7 days"
    if days <= 14:   return "8-14 days"
    if days <= 21:   return "15-21 days"
    if days <= 30:   return "22-30 days"
    if days <= 45:   return "31-45 days"
    if days <= 60:   return "46-60 days"
    if days <= 90:   return "61-90 days"
    return "90+ days"


def booking_window_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate booking window stats per route.
    Returns a summary DataFrame used in charts.
    """
    summary = (
        df.groupby("route")
        .agg(
            avg_lead_time  =("lead_time_days", "mean"),
            median_lead_time=("lead_time_days","median"),
            p25_lead_time  =("lead_time_days", lambda x: x.quantile(0.25)),
            p75_lead_time  =("lead_time_days", lambda x: x.quantile(0.75)),
            total_bookings =("converted",      "count"),
            conversion_rate=("converted",      "mean"),
            avg_fare       =("fare_hkd",       "mean"),
        )
        .reset_index()
        .round(1)
    )
    return summary.sort_values("avg_lead_time", ascending=False)


def cabin_window_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate booking window stats per cabin class."""
    summary = (
        df.groupby("cabin")
        .agg(
            avg_lead_time   =("lead_time_days", "mean"),
            median_lead_time=("lead_time_days", "median"),
            conversion_rate =("converted",      "mean"),
            avg_fare        =("fare_hkd",       "mean"),
            total_records   =("converted",      "count"),
        )
        .reset_index()
        .round(1)
    )
    return summary


def optimal_campaign_lead_time(
    df: pd.DataFrame,
    route: str,
    cabin: str,
) -> dict:
    """
    Calculate the optimal campaign launch date for a route/cabin.

    Logic:
      1. Find the P25 booking lead time (when 25% of bookings happen)
      2. Campaign should launch BEFORE this point
      3. Add 7 days buffer for campaign warm-up

    Returns dict with recommendation details.
    """
    filtered = df[
        (df["route"]  == route) &
        (df["cabin"]  == cabin) &
        (df["converted"] == 1)
    ]

    if len(filtered) < 10:
        # Fall back to route-level data
        filtered = df[
            (df["route"] == route) &
            (df["converted"] == 1)
        ]

    if len(filtered) < 5:
        return {
            "route":              route,
            "cabin":              cabin,
            "p25_lead_days":      30,
            "recommended_launch": 37,
            "note":               "Insufficient data — using default",
        }

    p25   = int(filtered["lead_time_days"].quantile(0.25))
    p50   = int(filtered["lead_time_days"].quantile(0.50))
    p75   = int(filtered["lead_time_days"].quantile(0.75))
    mean  = round(float(filtered["lead_time_days"].mean()), 1)

    # Campaign must be live before 25th percentile of bookings
    recommended_launch = p25 + 7   # 7-day warm-up buffer

    return {
        "route":                 route,
        "cabin":                 cabin,
        "mean_lead_days":        mean,
        "p25_lead_days":         p25,
        "p50_lead_days":         p50,
        "p75_lead_days":         p75,
        "recommended_launch_days_before_departure": recommended_launch,
        "insight": (
            f"25% of {cabin} bookings on {route} happen "
            f"{p25} days before departure. "
            f"Launch campaign at least {recommended_launch} days out."
        ),
    }


def conversion_by_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute conversion rate by lead time band.
    Used to show the booking urgency curve.
    """
    band_order = [
        "0-7 days", "8-14 days", "15-21 days", "22-30 days",
        "31-45 days", "46-60 days", "61-90 days", "90+ days",
    ]

    result = (
        df.groupby("lead_time_band")
        .agg(
            total     =("converted", "count"),
            converted =("converted", "sum"),
            avg_fare  =("fare_hkd",  "mean"),
        )
        .reset_index()
    )
    result["conversion_rate"] = (
        result["converted"] / result["total"] * 100
    ).round(1)
    result["avg_fare"] = result["avg_fare"].round(0)

    # Sort by logical band order
    result["band_order"] = result["lead_time_band"].map(
        {b: i for i, b in enumerate(band_order)}
    )
    return result.sort_values("band_order").drop(columns=["band_order"])