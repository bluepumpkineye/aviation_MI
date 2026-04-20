"""
Customer Data Generator
Creates synthetic but internally consistent customer records for
segmentation, CLV modeling, and churn analysis.
"""

import numpy as np
import pandas as pd
from faker import Faker

from config.settings import (
    AGE_BANDS,
    CABIN_CLASSES,
    LOYALTY_TIERS,
    MARKETS,
    NATIONALITIES,
    N_CUSTOMERS,
)

fake = Faker()
rng = np.random.default_rng(42)


def _weighted_choice(options, probabilities, size):
    return rng.choice(options, size=size, p=probabilities)


def generate_customers() -> pd.DataFrame:
    """Generate customer master data used by customer intelligence models."""
    n = N_CUSTOMERS

    age_band = _weighted_choice(
        AGE_BANDS,
        [0.08, 0.26, 0.24, 0.20, 0.15, 0.07],
        n,
    )
    loyalty_tier = _weighted_choice(
        LOYALTY_TIERS,
        [0.55, 0.23, 0.16, 0.06],
        n,
    )
    preferred_cabin = _weighted_choice(
        CABIN_CLASSES,
        [0.58, 0.18, 0.19, 0.05],
        n,
    )
    home_market = _weighted_choice(
        MARKETS,
        [0.20, 0.10, 0.08, 0.10, 0.11, 0.09, 0.07, 0.10, 0.07, 0.08],
        n,
    )
    nationality = rng.choice(NATIONALITIES, size=n)

    join_date = pd.to_datetime("2019-01-01") + pd.to_timedelta(
        rng.integers(0, 2190, size=n),
        unit="D",
    )
    recency_days = rng.integers(1, 361, size=n)
    last_booking_date = pd.Timestamp("2024-12-31") - pd.to_timedelta(recency_days, unit="D")

    trips_last_12m = np.clip(rng.poisson(3.2, size=n), 0, 18)
    lifetime_bookings = trips_last_12m + rng.integers(0, 24, size=n)
    avg_trip_value_hkd = np.clip(
        rng.normal(5200, 2200, size=n),
        1200,
        28000,
    ).round(0).astype(int)
    ancillary_spend_hkd = np.clip(
        avg_trip_value_hkd * rng.uniform(0.05, 0.22, size=n),
        100,
        4500,
    ).round(0).astype(int)

    tier_multiplier = pd.Series(loyalty_tier).map(
        {"Standard": 0.9, "Silver": 1.0, "Gold": 1.18, "Diamond": 1.4}
    ).to_numpy()
    cabin_multiplier = pd.Series(preferred_cabin).map(
        {"Economy": 0.85, "Premium Economy": 1.0, "Business": 1.45, "First": 2.0}
    ).to_numpy()

    total_revenue_hkd = (
        lifetime_bookings * avg_trip_value_hkd * tier_multiplier * cabin_multiplier
        + lifetime_bookings * ancillary_spend_hkd
    ).round(0).astype(int)

    email_open_rate = np.clip(rng.normal(0.36, 0.16, size=n), 0.02, 0.92).round(3)
    app_engagement_score = np.clip(
        35
        + trips_last_12m * 5
        + email_open_rate * 40
        + (pd.Series(loyalty_tier).map({"Standard": 0, "Silver": 8, "Gold": 15, "Diamond": 24}).to_numpy())
        - recency_days * 0.08,
        1,
        100,
    ).round(1)

    customer_lifetime_value_12m = (
        avg_trip_value_hkd * np.maximum(trips_last_12m, 1) * tier_multiplier
        + ancillary_spend_hkd * np.maximum(trips_last_12m, 1)
    ).round(0).astype(int)

    churn_probability_90d = np.clip(
        0.55
        - trips_last_12m * 0.035
        - email_open_rate * 0.20
        - (app_engagement_score / 100) * 0.20
        + recency_days / 420
        + rng.normal(0, 0.05, size=n),
        0.01,
        0.97,
    ).round(3)

    customer_segment = np.select(
        [
            (loyalty_tier == "Diamond") | (customer_lifetime_value_12m >= 90000),
            (trips_last_12m >= 5) & (churn_probability_90d < 0.30),
            (email_open_rate >= 0.45) & (trips_last_12m <= 2),
            churn_probability_90d >= 0.60,
        ],
        [
            "High Value Loyalists",
            "Frequent Flyers",
            "Growth Opportunity",
            "At Risk",
        ],
        default="Mass Market",
    )

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{str(i).zfill(6)}" for i in range(n)],
            "first_name": [fake.first_name() for _ in range(n)],
            "last_name": [fake.last_name() for _ in range(n)],
            "age_band": age_band,
            "nationality": nationality,
            "home_market": home_market,
            "loyalty_tier": loyalty_tier,
            "preferred_cabin_class": preferred_cabin,
            "join_date": join_date.strftime("%Y-%m-%d"),
            "last_booking_date": last_booking_date.strftime("%Y-%m-%d"),
            "days_since_last_booking": recency_days,
            "trips_last_12m": trips_last_12m,
            "lifetime_bookings": lifetime_bookings,
            "avg_trip_value_hkd": avg_trip_value_hkd,
            "ancillary_spend_hkd": ancillary_spend_hkd,
            "total_revenue_hkd": total_revenue_hkd,
            "email_open_rate": email_open_rate,
            "app_engagement_score": app_engagement_score,
            "customer_lifetime_value_12m": customer_lifetime_value_12m,
            "churn_probability_90d": churn_probability_90d,
            "customer_segment": customer_segment,
        }
    )

    return df
