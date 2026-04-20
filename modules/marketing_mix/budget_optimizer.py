"""
Budget allocation recommendations for the marketing mix UI.
"""

import pandas as pd

from modules.marketing_mix.mmm_model import channel_efficiency_summary


def recommend_budget_allocation(df_campaigns: pd.DataFrame, total_budget_hkd: float) -> pd.DataFrame:
    """Allocate budget using a blended score of efficiency and scale."""
    summary = channel_efficiency_summary(df_campaigns)
    roas_score = summary["roas"] / summary["roas"].sum()
    conversion_score = summary["conversion_rate"] / summary["conversion_rate"].sum()
    bookings_score = summary["bookings"] / summary["bookings"].sum()

    summary["allocation_score"] = (
        roas_score * 0.45
        + conversion_score * 0.25
        + bookings_score * 0.30
    )
    summary["recommended_budget_hkd"] = (
        summary["allocation_score"] / summary["allocation_score"].sum() * float(total_budget_hkd)
    ).round(0)

    current_total = summary["spend_hkd"].sum()
    summary["current_budget_share"] = (summary["spend_hkd"] / current_total).round(4)
    summary["recommended_share"] = (
        summary["recommended_budget_hkd"] / summary["recommended_budget_hkd"].sum()
    ).round(4)
    summary["budget_change_pct"] = (
        (summary["recommended_budget_hkd"] - summary["spend_hkd"])
        / summary["spend_hkd"].clip(lower=1)
        * 100
    ).round(1)
    return summary.sort_values("recommended_budget_hkd", ascending=False).reset_index(drop=True)
