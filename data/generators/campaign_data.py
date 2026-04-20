"""
Campaign Data Generator
Produces marketing campaign performance data across channels,
routes, and markets — the raw material for MMM and attribution.
"""

import numpy as np
import pandas as pd
from faker import Faker
from config.settings import (
    N_CAMPAIGNS, CHANNELS, MARKETS, ROUTES, CABIN_CLASSES, CURRENCY_SYMBOL
)

fake = Faker()
rng  = np.random.default_rng(42)


def generate_campaigns() -> pd.DataFrame:
    """Generate campaign performance records."""

    n = N_CAMPAIGNS

    channels   = rng.choice(CHANNELS, size=n)
    markets    = rng.choice(MARKETS, size=n)
    routes     = rng.choice(ROUTES, size=n)
    cabins     = rng.choice(CABIN_CLASSES, size=n)

    # Campaign dates — spread over 2 years
    start_offsets = rng.integers(0, 680, size=n)
    start_dates   = pd.to_datetime("2023-01-01") + pd.to_timedelta(start_offsets, unit="d")
    durations     = rng.integers(7, 60, size=n)
    end_dates     = start_dates + pd.to_timedelta(durations, unit="d")

    # Spend — varies hugely by channel
    channel_spend_map = {
        "Paid Search":           (50_000,  500_000),
        "Paid Social":           (30_000,  300_000),
        "Programmatic Display":  (20_000,  200_000),
        "Email CRM":             (5_000,   50_000),
        "YouTube / Video":       (80_000,  800_000),
        "Affiliate":             (10_000,  100_000),
        "Out-of-Home":           (100_000, 1_000_000),
        "Organic Search":        (0,       10_000),
    }

    spend = np.array([
        rng.integers(*channel_spend_map.get(c, (10_000, 200_000)))
        for c in channels
    ])

    # Performance metrics — derived from spend with realistic noise
    ctr            = np.clip(rng.normal(0.025, 0.012, n), 0.001, 0.15).round(4)
    impressions    = (spend / rng.uniform(0.01, 0.05, n)).astype(int)
    clicks         = (impressions * ctr).astype(int)
    conv_rate      = np.clip(rng.normal(0.018, 0.008, n), 0.001, 0.08).round(4)
    bookings       = np.clip((clicks * conv_rate).astype(int), 1, 5000)

    avg_booking_val = rng.integers(3_000, 25_000, n)
    revenue        = (bookings * avg_booking_val).astype(int)
    roas           = np.where(spend > 0, (revenue / spend).round(2), 0)
    cac            = np.where(bookings > 0, (spend / bookings).round(0), spend).astype(int)

    df = pd.DataFrame({
        "campaign_id":      [f"CAMP{str(i).zfill(5)}" for i in range(n)],
        "campaign_name":    [
            f"{m} {r.split('-')[1]} {cab[:3]} {ch.split()[0]} Q{rng.integers(1,5)}"
            for m, r, cab, ch in zip(markets, routes, cabins, channels)
        ],
        "channel":          channels,
        "market":           markets,
        "route":            routes,
        "cabin_class":      cabins,
        "start_date":       start_dates.strftime("%Y-%m-%d"),
        "end_date":         end_dates.strftime("%Y-%m-%d"),
        "duration_days":    durations,
        "spend_hkd":        spend,
        "impressions":      impressions,
        "clicks":           clicks,
        "ctr":              ctr,
        "conversion_rate":  conv_rate,
        "bookings":         bookings,
        "revenue_hkd":      revenue,
        "roas":             roas,
        "cac_hkd":          cac,
        "avg_booking_value":avg_booking_val,
    })

    return df