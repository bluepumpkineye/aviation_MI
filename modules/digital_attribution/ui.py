"""
Streamlit UI for digital attribution.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config.brand import CHART_COLORS, COLORS
from modules.digital_attribution.ab_testing import run_ab_test
from modules.digital_attribution.journey_analyzer import (
    build_conversion_paths,
    funnel_summary,
    top_conversion_paths,
)
from modules.digital_attribution.markov_attribution import compute_channel_attribution
from utils.chart_helpers import (
    apply_theme,
    format_currency,
    format_dataframe_display,
    format_percentage,
    labeled_divider,
    render_metric_grid,
    render_page_header,
)
from utils.insight_renderer import render_insight_banner, render_what_this_means


@st.cache_data(show_spinner=False)
def _cached_paths(df_digital: pd.DataFrame) -> pd.DataFrame:
    return build_conversion_paths(df_digital)


@st.cache_data(show_spinner=False)
def _cached_top_paths(df_digital: pd.DataFrame) -> pd.DataFrame:
    return top_conversion_paths(df_digital)


@st.cache_data(show_spinner=False)
def _cached_attribution(df_digital: pd.DataFrame) -> pd.DataFrame:
    return compute_channel_attribution(df_digital)


@st.cache_data(show_spinner=False)
def _cached_ab(df_digital: pd.DataFrame):
    return run_ab_test(df_digital)


def render(
    df_customers: pd.DataFrame | None,
    df_campaigns: pd.DataFrame | None,
    df_demand: pd.DataFrame | None,
    df_digital: pd.DataFrame,
    insights: dict | None = None,
):
    """Render the digital attribution dashboard."""
    render_page_header(
        title="Digital Attribution",
        subtitle="Read cross-channel journeys, challenge last-touch bias, and validate optimisation decisions with controlled experiments.",
        pills=["Markov Logic", "Journey Paths", "First vs Last Touch", "A/B Testing"],
        meta=f"Updated {pd.Timestamp.today().strftime('%d %b %Y')}",
    )

    digital_insights = []
    if insights:
        digital_insights = [
            item
            for bucket in ["critical", "opportunity", "on_track", "cross_module"]
            for item in insights.get(bucket, [])
            if "Digital Attribution" in item["module"]
        ][:2]
    if digital_insights:
        labeled_divider("AI Insights")
        for item in digital_insights:
            render_insight_banner(item)

    funnel = funnel_summary(df_digital)
    paths = _cached_paths(df_digital)
    top_paths = _cached_top_paths(df_digital)
    attribution = _cached_attribution(df_digital)
    ab_summary, ab_decision = _cached_ab(df_digital)
    overall_conversion_rate = float(df_digital["converted"].sum() / max(len(df_digital), 1))

    labeled_divider("Performance Analysis")
    render_metric_grid(
        [
            {"label": "Touchpoints", "value": f"{len(df_digital):,}"},
            {"label": "Clicks", "value": f"{int(df_digital['clicked'].sum()):,}", "delta": format_percentage(df_digital["clicked"].mean()), "delta_state": "neutral"},
            {"label": "Conversions", "value": f"{int(df_digital['converted'].sum()):,}", "delta": format_percentage(overall_conversion_rate), "delta_state": "positive"},
            {"label": "Unique Paths", "value": f"{paths['path'].nunique():,}" if not paths.empty else "0"},
        ],
        columns=4,
    )

    journey_tab, attr_tab, ab_tab = st.tabs(["Journey Analysis", "Attribution", "A/B Testing"])

    with journey_tab:
        left, right = st.columns([0.8, 1.2])
        fig = px.funnel(
            funnel,
            x="count",
            y="stage",
            color="stage",
            color_discrete_sequence=CHART_COLORS,
            title="Digital Funnel",
        )
        fig.update_layout(height=320, showlegend=False)
        apply_theme(fig)
        left.plotly_chart(fig, use_container_width=True)

        fig = px.bar(
            top_paths.sort_values("conversions"),
            x="conversions",
            y="path",
            orientation="h",
            color="total_value_hkd",
            color_continuous_scale=["#173B61", COLORS["accent"]],
            title="Top Conversion Paths",
        )
        fig.update_layout(height=360, xaxis_title="Conversions", yaxis_title="")
        apply_theme(fig)
        right.plotly_chart(fig, use_container_width=True)
        render_what_this_means(
            text=(
                f"The platform has analysed {paths['path'].nunique():,} unique converting journeys and found an overall conversion rate of {format_percentage(overall_conversion_rate)}. "
                "Paths that repeatedly appear at the top should influence how you sequence channels, not just how you credit them."
            ),
            action="Focus creative and budget on the highest-volume path combinations first",
        )
        path_display = format_dataframe_display(
            top_paths,
            currency_cols=["avg_value_hkd", "total_value_hkd"],
            rename_map={
                "path": "Journey Path",
                "conversions": "Conversions",
                "avg_value_hkd": "Avg Booking Value",
                "total_value_hkd": "Total Value",
            },
        )
        st.dataframe(path_display, use_container_width=True, hide_index=True)

    with attr_tab:
        fig = px.bar(
            attribution.sort_values("removal_effect_score"),
            x="removal_effect_score",
            y="channel",
            orientation="h",
            color="removal_effect_score",
            color_continuous_scale=["#173B61", COLORS["warning"]],
            title="Channel Removal Effect Proxy",
        )
        fig.update_layout(coloraxis_showscale=False, height=340, xaxis_title="Removal Effect Score", yaxis_title="")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        overcredited = attribution.iloc[0]
        render_what_this_means(
            text=(
                f"{overcredited['channel']} currently carries the highest removal-effect proxy at {overcredited['removal_effect_score']:.1f}. "
                "Channels with high last-touch credit but lower journey presence should be treated carefully before increasing budget."
            ),
            action="Compare journey share and last-touch share before finalising investment shifts",
        )
        attribution_display = format_dataframe_display(
            attribution,
            pct_cols=["path_share", "first_touch_share", "last_touch_share"],
            rename_map={
                "channel": "Channel",
                "path_share": "Journey Share",
                "first_touch_share": "First Touch Share",
                "last_touch_share": "Last Touch Share",
                "removal_effect_score": "Removal Effect Score",
                "conversion_paths": "Conversion Paths",
                "total_paths": "Total Paths",
            },
        )
        st.dataframe(attribution_display, use_container_width=True, hide_index=True)

    with ab_tab:
        left, right = st.columns([0.8, 1.2])
        left.metric("Winner", ab_decision["winner"])
        left.metric("Conversion Uplift", f"{ab_decision['uplift_pct']:.2f}%")
        left.metric("P-value", ab_decision["p_value"])
        left.metric("Significant", "Yes" if ab_decision["is_significant"] else "No")

        fig = px.bar(
            ab_summary,
            x="variant",
            y="conversion_rate",
            color="variant",
            color_discrete_sequence=[COLORS["secondary"], COLORS["success"]],
            title="A/B Conversion Rate",
        )
        fig.update_layout(height=300, showlegend=False, yaxis_title="Conversion Rate")
        apply_theme(fig)
        right.plotly_chart(fig, use_container_width=True)

        render_what_this_means(
            text=(
                f"The {ab_decision['winner']} variant currently leads with a {ab_decision['uplift_pct']:.2f}% conversion uplift and a p-value of {ab_decision['p_value']}. "
                "Use significance and revenue, not just click-through, when deciding whether to roll out a variant."
            ),
            action="Promote the winner only if the revenue uplift justifies full deployment",
        )
        ab_display = format_dataframe_display(
            ab_summary,
            currency_cols=["revenue_hkd"],
            pct_cols=["ctr", "conversion_rate"],
            rename_map={
                "variant": "Variant",
                "sessions": "Sessions",
                "clicks": "Clicks",
                "conversions": "Conversions",
                "avg_engagement_seconds": "Avg Engagement (Sec)",
                "revenue_hkd": "Revenue",
                "ctr": "CTR",
                "conversion_rate": "Conversion Rate",
            },
        )
        st.dataframe(ab_display, use_container_width=True, hide_index=True)
