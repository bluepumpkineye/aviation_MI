"""
Channel contribution summaries derived from MMM SHAP values.
"""

import pandas as pd

from modules.marketing_mix.mmm_model import fit_mmm_model


def channel_contribution_summary(df_campaigns: pd.DataFrame) -> pd.DataFrame:
    """Return normalized SHAP-based contribution shares by channel."""
    artifacts = fit_mmm_model(df_campaigns)
    shap_summary = artifacts["shap_summary"].copy()
    total = shap_summary["mean_abs_shap"].sum()
    shap_summary["contribution_share"] = (
        shap_summary["mean_abs_shap"] / max(total, 1e-9)
    ).round(4)
    shap_summary["contribution_pct"] = (shap_summary["contribution_share"] * 100).round(1)
    return shap_summary
