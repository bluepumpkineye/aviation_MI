"""
Route Demand Data Generator
Creates daily route-level booking demand with seasonal patterns,
holiday peaks, and route-specific demand levels.
"""

import numpy as np
import pandas as pd

from config.settings import MARKETS, N_ROUTE_DAYS, ROUTES

rng = np.random.default_rng(42)


def _destination_market(route: str) -> str:
    destination = route.split("-")[1]
    mapping = {
        "LHR": "UK",
        "JFK": "US",
        "SYD": "AU",
        "NRT": "JP",
        "SIN": "SG",
        "LAX": "US",
        "YVR": "US",
        "FRA": "UK",
        "CDG": "UK",
        "DXB": "TH",
        "BKK": "TH",
        "TPE": "TW",
        "ICN": "KR",
        "MEL": "AU",
        "MNL": "SG",
    }
    return mapping.get(destination, rng.choice(MARKETS))


def generate_route_demand() -> pd.DataFrame:
    """Generate daily booking demand per route."""
    dates = pd.date_range("2023-01-01", periods=N_ROUTE_DAYS, freq="D")

    route_base = {
        "HKG-LHR": 230,
        "HKG-JFK": 215,
        "HKG-SYD": 185,
        "HKG-NRT": 245,
        "HKG-SIN": 255,
        "HKG-LAX": 205,
        "HKG-YVR": 140,
        "HKG-FRA": 175,
        "HKG-CDG": 165,
        "HKG-DXB": 150,
        "HKG-BKK": 260,
        "HKG-TPE": 275,
        "HKG-ICN": 225,
        "HKG-MEL": 168,
        "HKG-MNL": 238,
    }

    records = []
    for route in ROUTES:
        base = route_base.get(route, 180)
        market = _destination_market(route)
        route_noise = rng.normal(0, 6)

        for day_index, date in enumerate(dates):
            yearly = 1 + 0.18 * np.sin(2 * np.pi * date.dayofyear / 365.25)
            weekly = 1 + 0.07 * np.cos(2 * np.pi * date.dayofweek / 7)
            trend = 1 + day_index / (len(dates) * 18)

            peak_multiplier = 1.0
            if date.month in (7, 8, 12):
                peak_multiplier += 0.16
            if date.month in (1, 2) and route in {"HKG-NRT", "HKG-SYD", "HKG-SIN", "HKG-BKK"}:
                peak_multiplier += 0.10

            bookings = base * yearly * weekly * trend * peak_multiplier + route_noise + rng.normal(0, 14)
            bookings = max(25, int(round(bookings)))

            avg_fare_hkd = np.clip(
                (2800 + base * 10) * (1 + rng.normal(0, 0.08)) * (1 + (peak_multiplier - 1) * 0.7),
                1800,
                9800,
            )
            load_factor = np.clip(
                0.68 + (bookings / (base * 1.6)) * 0.18 + rng.normal(0, 0.03),
                0.52,
                0.97,
            )
            search_volume_index = max(30, int(round(bookings * rng.uniform(5.5, 8.5))))

            records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "route": route,
                    "market": market,
                    "bookings": bookings,
                    "avg_fare_hkd": int(round(avg_fare_hkd)),
                    "load_factor": round(float(load_factor), 3),
                    "search_volume_index": search_volume_index,
                }
            )

    return pd.DataFrame(records)
