"""
Digital Touchpoint Data Generator
Creates synthetic cross-channel interaction data linked to customers
and campaigns for attribution and funnel analysis.
"""

import numpy as np
import pandas as pd

from config.settings import CHANNELS, MARKETS, N_CAMPAIGNS, N_CUSTOMERS, N_TOUCHPOINTS, ROUTES

rng = np.random.default_rng(42)


def generate_digital_touchpoints() -> pd.DataFrame:
    """Generate digital touchpoints across web, CRM, and paid channels."""
    n = N_TOUCHPOINTS

    timestamps = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 730 * 24 * 60, size=n),
        unit="m",
    )
    customer_idx = rng.integers(0, N_CUSTOMERS, size=n)
    campaign_idx = rng.integers(0, N_CAMPAIGNS, size=n)
    channels = rng.choice(
        CHANNELS,
        size=n,
        p=[0.19, 0.18, 0.14, 0.16, 0.10, 0.08, 0.04, 0.11],
    )
    markets = rng.choice(MARKETS, size=n)
    routes = rng.choice(ROUTES, size=n)
    device_type = rng.choice(["Mobile", "Desktop", "Tablet"], size=n, p=[0.62, 0.31, 0.07])
    journey_stage = rng.choice(
        ["Awareness", "Consideration", "Intent", "Purchase"],
        size=n,
        p=[0.33, 0.31, 0.24, 0.12],
    )
    page_type = rng.choice(
        ["Homepage", "Route Search", "Fare Deals", "Destination Guide", "Checkout", "Loyalty"],
        size=n,
        p=[0.14, 0.31, 0.16, 0.13, 0.12, 0.14],
    )

    base_click_prob = pd.Series(channels).map(
        {
            "Paid Search": 0.34,
            "Paid Social": 0.18,
            "Programmatic Display": 0.08,
            "Email CRM": 0.22,
            "YouTube / Video": 0.07,
            "Affiliate": 0.19,
            "Out-of-Home": 0.02,
            "Organic Search": 0.27,
        }
    ).to_numpy()
    clicked = rng.random(n) < base_click_prob

    stage_conversion_bonus = pd.Series(journey_stage).map(
        {"Awareness": 0.01, "Consideration": 0.03, "Intent": 0.08, "Purchase": 0.18}
    ).to_numpy()
    channel_conversion_bonus = pd.Series(channels).map(
        {
            "Paid Search": 0.05,
            "Paid Social": 0.03,
            "Programmatic Display": 0.01,
            "Email CRM": 0.07,
            "YouTube / Video": 0.02,
            "Affiliate": 0.04,
            "Out-of-Home": 0.005,
            "Organic Search": 0.05,
        }
    ).to_numpy()

    conversion_probability = np.clip(
        0.01 + stage_conversion_bonus + channel_conversion_bonus + clicked * 0.05,
        0.005,
        0.42,
    )
    converted = rng.random(n) < conversion_probability

    engagement_seconds = np.clip(
        rng.normal(55, 24, size=n) + clicked * 35 + converted * 80,
        3,
        780,
    ).round(0).astype(int)

    booking_value_hkd = np.where(
        converted,
        np.clip(rng.normal(4800, 1900, size=n), 1200, 24000).round(0).astype(int),
        0,
    )

    df = pd.DataFrame(
        {
            "touchpoint_id": [f"TP{str(i).zfill(7)}" for i in range(n)],
            "customer_id": [f"CUST{str(i).zfill(6)}" for i in customer_idx],
            "campaign_id": [f"CAMP{str(i).zfill(5)}" for i in campaign_idx],
            "session_id": [f"SES{str(i).zfill(8)}" for i in rng.integers(0, n * 3, size=n)],
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "channel": channels,
            "market": markets,
            "route": routes,
            "device_type": device_type,
            "journey_stage": journey_stage,
            "page_type": page_type,
            "clicked": clicked.astype(int),
            "converted": converted.astype(int),
            "engagement_seconds": engagement_seconds,
            "booking_value_hkd": booking_value_hkd,
        }
    )

    return df.sort_values("timestamp").reset_index(drop=True)
