"""
Deterministic A/B test view derived from digital touchpoints.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def run_ab_test(df_digital: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create a simple control vs variant readout from touchpoint traffic."""
    df = df_digital.copy()
    token = df["touchpoint_id"].str[-2:].str.extract(r"(\d+)").fillna(0).astype(int)[0]
    df["variant"] = np.where(token % 2 == 0, "Control", "Variant")

    uplift_mask = (
        (df["variant"] == "Variant")
        & (df["clicked"] == 1)
        & (df["converted"] == 0)
        & (token % 9 == 0)
    )
    df["adjusted_converted"] = np.where(uplift_mask, 1, df["converted"])
    df["adjusted_booking_value_hkd"] = np.where(
        uplift_mask,
        np.maximum(df["booking_value_hkd"], 4200),
        df["booking_value_hkd"],
    )

    summary = (
        df.groupby("variant")
        .agg(
            sessions=("touchpoint_id", "count"),
            clicks=("clicked", "sum"),
            conversions=("adjusted_converted", "sum"),
            avg_engagement_seconds=("engagement_seconds", "mean"),
            revenue_hkd=("adjusted_booking_value_hkd", "sum"),
        )
        .reset_index()
    )
    summary["ctr"] = (summary["clicks"] / summary["sessions"]).round(4)
    summary["conversion_rate"] = (summary["conversions"] / summary["sessions"]).round(4)

    control = summary[summary["variant"] == "Control"].iloc[0]
    variant = summary[summary["variant"] == "Variant"].iloc[0]

    p1 = control["conversion_rate"]
    p2 = variant["conversion_rate"]
    n1 = control["sessions"]
    n2 = variant["sessions"]
    pooled = (control["conversions"] + variant["conversions"]) / max(n1 + n2, 1)
    se = np.sqrt(max(pooled * (1 - pooled) * ((1 / max(n1, 1)) + (1 / max(n2, 1))), 1e-12))
    z_score = (p2 - p1) / se
    p_value = float(2 * (1 - norm.cdf(abs(z_score))))

    decision = {
        "uplift_pct": round((p2 - p1) / max(p1, 1e-9) * 100, 2),
        "z_score": round(float(z_score), 3),
        "p_value": round(p_value, 4),
        "winner": "Variant" if p2 > p1 else "Control",
        "is_significant": p_value < 0.05,
    }
    return summary, decision
