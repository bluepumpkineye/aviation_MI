"""
Scenario simulation helpers for the marketing mix UI.
"""

import numpy as np
import pandas as pd

from modules.marketing_mix.mmm_model import channel_efficiency_summary


def simulate_budget_scenarios(df_campaigns: pd.DataFrame, base_budget_hkd: float) -> pd.DataFrame:
    """Generate conservative, optimized, and aggressive budget scenarios."""
    summary = channel_efficiency_summary(df_campaigns)
    blended_roas = float((summary["revenue_hkd"].sum() / summary["spend_hkd"].sum()).round(2))
    efficiency_index = summary.set_index("channel")["roas"] / summary["roas"].mean()

    scenarios = []
    for name, multiplier in [
        ("Conservative", 0.90),
        ("Optimized", 1.00),
        ("Aggressive", 1.15),
    ]:
        budget = base_budget_hkd * multiplier
        uplift = np.average(efficiency_index.values, weights=summary["spend_hkd"].values)
        projected_revenue = budget * blended_roas * (0.92 + 0.08 * uplift)
        projected_bookings = projected_revenue / summary["revenue_per_booking"].mean()
        scenarios.append(
            {
                "scenario": name,
                "budget_hkd": round(budget, 0),
                "projected_revenue_hkd": round(projected_revenue, 0),
                "projected_bookings": round(projected_bookings, 0),
                "projected_roas": round(projected_revenue / max(budget, 1), 2),
            }
        )
    return pd.DataFrame(scenarios)


def channel_response_curve(df_campaigns: pd.DataFrame) -> pd.DataFrame:
    """Create a simple response-curve table for channel spend changes."""
    summary = channel_efficiency_summary(df_campaigns)
    records = []
    for _, row in summary.iterrows():
        for pct_change in [-20, -10, 0, 10, 20]:
            spend = row["spend_hkd"] * (1 + pct_change / 100)
            elasticity = 0.55 if row["roas"] >= summary["roas"].median() else 0.35
            projected_revenue = row["revenue_hkd"] * (1 + elasticity * pct_change / 100)
            records.append(
                {
                    "channel": row["channel"],
                    "pct_change": pct_change,
                    "projected_spend_hkd": round(spend, 0),
                    "projected_revenue_hkd": round(projected_revenue, 0),
                }
            )
    return pd.DataFrame(records)
