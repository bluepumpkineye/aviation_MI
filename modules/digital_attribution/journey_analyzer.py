"""
Journey path analysis helpers for digital attribution.
"""

import pandas as pd


def build_conversion_paths(df_digital: pd.DataFrame, lookback_steps: int = 4) -> pd.DataFrame:
    """Build compact customer-channel paths ending in a conversion event."""
    df = df_digital.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["customer_id", "timestamp"])

    records = []
    for customer_id, group in df.groupby("customer_id", sort=False):
        channels = group["channel"].tolist()
        converted_idx = group.index[group["converted"] == 1].tolist()
        if not converted_idx:
            continue
        for idx in converted_idx:
            pos = group.index.get_loc(idx)
            path_channels = channels[max(0, pos - lookback_steps + 1) : pos + 1]
            current_row = group.loc[idx]
            records.append(
                {
                    "customer_id": customer_id,
                    "conversion_timestamp": current_row["timestamp"],
                    "path": " > ".join(path_channels),
                    "steps": len(path_channels),
                    "first_channel": path_channels[0],
                    "last_channel": path_channels[-1],
                    "booking_value_hkd": current_row["booking_value_hkd"],
                }
            )
    return pd.DataFrame(records)


def top_conversion_paths(df_digital: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Return the highest-volume conversion paths."""
    paths = build_conversion_paths(df_digital)
    if paths.empty:
        return pd.DataFrame(columns=["path", "conversions", "avg_value_hkd", "total_value_hkd"])

    summary = (
        paths.groupby("path")
        .agg(
            conversions=("path", "count"),
            avg_value_hkd=("booking_value_hkd", "mean"),
            total_value_hkd=("booking_value_hkd", "sum"),
        )
        .reset_index()
        .sort_values(["conversions", "total_value_hkd"], ascending=[False, False])
        .head(top_n)
        .round(0)
    )
    return summary


def funnel_summary(df_digital: pd.DataFrame) -> pd.DataFrame:
    """Compute basic interaction, click, and conversion funnel metrics."""
    touches = len(df_digital)
    clicks = int(df_digital["clicked"].sum())
    conversions = int(df_digital["converted"].sum())
    return pd.DataFrame(
        {
            "stage": ["Touches", "Clicks", "Conversions"],
            "count": [touches, clicks, conversions],
        }
    )
