"""
Customer segmentation helpers for the Streamlit UI.
"""

import numpy as np
import pandas as pd

SEGMENT_ORDER = [
    "High Value Loyalists",
    "Frequent Flyers",
    "Growth Opportunity",
    "Mass Market",
    "At Risk",
]


def prepare_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features used across customer analytics views."""
    enriched = df.copy()
    enriched["last_booking_date"] = pd.to_datetime(enriched["last_booking_date"])
    enriched["join_date"] = pd.to_datetime(enriched["join_date"])
    enriched["tenure_days"] = (pd.Timestamp("2025-01-01") - enriched["join_date"]).dt.days.clip(lower=1)
    enriched["revenue_per_trip_hkd"] = (
        enriched["total_revenue_hkd"] / enriched["lifetime_bookings"].clip(lower=1)
    ).round(0)
    enriched["engagement_band"] = pd.cut(
        enriched["app_engagement_score"],
        bins=[0, 35, 60, 80, 100],
        labels=["Low", "Moderate", "High", "Very High"],
        include_lowest=True,
    )
    return enriched


def add_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple 1-5 recency, frequency, and monetary scores."""
    enriched = prepare_customer_features(df)

    recency_rank = pd.qcut(
        enriched["days_since_last_booking"].rank(method="first", ascending=False),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    frequency_rank = pd.qcut(
        enriched["trips_last_12m"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    monetary_rank = pd.qcut(
        enriched["customer_lifetime_value_12m"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    enriched["r_score"] = recency_rank
    enriched["f_score"] = frequency_rank
    enriched["m_score"] = monetary_rank
    enriched["rfm_score"] = (
        enriched["r_score"].astype(str)
        + enriched["f_score"].astype(str)
        + enriched["m_score"].astype(str)
    )
    return enriched


def segment_profile_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate segment-level counts and commercial indicators."""
    enriched = prepare_customer_features(df)
    summary = (
        enriched.groupby("customer_segment")
        .agg(
            customers=("customer_id", "count"),
            avg_clv_hkd=("customer_lifetime_value_12m", "mean"),
            avg_churn_prob=("churn_probability_90d", "mean"),
            avg_trips=("trips_last_12m", "mean"),
            avg_engagement=("app_engagement_score", "mean"),
        )
        .reset_index()
    )
    summary["customer_segment"] = pd.Categorical(
        summary["customer_segment"],
        categories=SEGMENT_ORDER,
        ordered=True,
    )
    return summary.sort_values("customer_segment").round(2)


def segment_market_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a segment-by-market customer count matrix."""
    enriched = prepare_customer_features(df)
    matrix = (
        enriched.pivot_table(
            index="customer_segment",
            columns="home_market",
            values="customer_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(SEGMENT_ORDER)
        .fillna(0)
    )
    return matrix


def top_customer_opportunities(df: pd.DataFrame, segment: str | None = None, n: int = 20) -> pd.DataFrame:
    """Return highest-value customers, optionally filtered by segment."""
    enriched = prepare_customer_features(df)
    if segment:
        enriched = enriched[enriched["customer_segment"] == segment]
    cols = [
        "customer_id",
        "first_name",
        "last_name",
        "customer_segment",
        "home_market",
        "loyalty_tier",
        "customer_lifetime_value_12m",
        "churn_probability_90d",
        "trips_last_12m",
    ]
    return enriched.sort_values(
        ["customer_lifetime_value_12m", "trips_last_12m"],
        ascending=[False, False],
    )[cols].head(n)
