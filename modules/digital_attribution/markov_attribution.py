"""
Simple path-based attribution proxies for the Streamlit UI.
"""

import pandas as pd

from modules.digital_attribution.journey_analyzer import build_conversion_paths


def compute_channel_attribution(df_digital: pd.DataFrame) -> pd.DataFrame:
    """Estimate channel contribution using path incidence and touch positions."""
    paths = build_conversion_paths(df_digital)
    if paths.empty:
        return pd.DataFrame(
            columns=[
                "channel",
                "path_share",
                "first_touch_share",
                "last_touch_share",
                "removal_effect_score",
            ]
        )

    path_total = len(paths)
    channels = sorted(
        {
            channel
            for path in paths["path"]
            for channel in path.split(" > ")
        }
    )

    records = []
    for channel in channels:
        involved = paths["path"].str.contains(channel, regex=False)
        path_share = involved.mean()
        first_touch_share = (paths["first_channel"] == channel).mean()
        last_touch_share = (paths["last_channel"] == channel).mean()
        removal_effect = path_share * 0.45 + first_touch_share * 0.20 + last_touch_share * 0.35
        records.append(
            {
                "channel": channel,
                "path_share": round(path_share, 4),
                "first_touch_share": round(first_touch_share, 4),
                "last_touch_share": round(last_touch_share, 4),
                "removal_effect_score": round(removal_effect * 100, 2),
                "conversion_paths": int(involved.sum()),
                "total_paths": path_total,
            }
        )
    return pd.DataFrame(records).sort_values("removal_effect_score", ascending=False).reset_index(drop=True)
