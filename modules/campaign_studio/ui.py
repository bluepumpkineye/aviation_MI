"""
Streamlit UI for the campaign studio.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config.brand import COLORS
from config.settings import CABIN_CLASSES, ROUTES
from modules.campaign_studio.copy_generator import generate_full_copy_set, score_subject_lines
from utils.chart_helpers import (
    apply_theme,
    format_dataframe_display,
    labeled_divider,
    render_page_header,
)
from utils.insight_renderer import render_decision_card, render_insight_banner, render_what_this_means


def render(
    df_customers=None,
    df_campaigns=None,
    df_demand=None,
    df_digital=None,
    insights: dict | None = None,
):
    """Render the campaign studio dashboard."""
    render_page_header(
        title="Campaign Studio",
        subtitle="Shape route-ready creative, score subject lines, and turn demand signals into campaign-ready execution.",
        pills=["Template Generation", "Subject Line Scoring", "Route Personalisation", "Creative Briefing"],
        meta=f"Updated {pd.Timestamp.today().strftime('%d %b %Y')}",
    )

    studio_insights = []
    if insights:
        studio_insights = [
            item
            for item in insights.get("cross_module", [])
            if "Campaign Studio" in item["module"]
        ][:1]
    if studio_insights:
        labeled_divider("AI Insights")
        for item in studio_insights:
            render_insight_banner(item)

    left, right = st.columns([1.0, 1.2])

    with left:
        route = st.selectbox("Route", ROUTES, index=3, key="studio_route")
        cabin = st.selectbox("Cabin", CABIN_CLASSES, index=1, key="studio_cabin")
        segment = st.selectbox(
            "Audience Segment",
            ["Champions", "Loyal Travellers", "At-Risk Frequent", "Occasional Flyers", "Dormant Members"],
            index=1,
        )
        base_fare = st.number_input("Base Fare HKD", min_value=1000, value=4800, step=250, key="studio_fare")
        n_variations = st.slider("Copy Variations", min_value=2, max_value=6, value=3, key="studio_variations")

    copy_df = generate_full_copy_set(cabin, route, segment, base_fare, n_variations)
    subject_scores = score_subject_lines(copy_df["subject_line"].tolist())

    labeled_divider("Performance Analysis")
    with right:
        best_subject = subject_scores.iloc[0]
        st.metric("Top Subject Score", int(best_subject["score"]))
        st.metric("Predicted Open Rate", f"{best_subject['predicted_open_rate'] * 100:.1f}%")
        st.write(best_subject["subject_line"])
        fig = px.bar(
            subject_scores,
            x="score",
            y="subject_line",
            orientation="h",
            color="predicted_open_rate",
            color_continuous_scale=["#1B4965", COLORS["warning"]],
            title="Subject Line Heuristic Scores",
        )
        fig.update_layout(coloraxis_showscale=False, height=320, xaxis_title="Score", yaxis_title="")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        render_what_this_means(
            text=(
                f"The strongest subject line for {route} {cabin} currently scores {int(best_subject['score'])}/100 with an estimated open rate of "
                f"{best_subject['predicted_open_rate'] * 100:.1f}%. Use the best-performing line as the control for your next live test."
            ),
            action="Brief CRM and paid social teams using the highest-scoring subject line as the first draft",
        )

    copy_display = format_dataframe_display(
        copy_df,
        rename_map={
            "copy_id": "Copy ID",
            "cabin": "Cabin",
            "route": "Route",
            "segment": "Segment",
            "headline": "Headline",
            "body_copy": "Body Copy",
            "subject_line": "Subject Line",
            "cta": "CTA",
            "base_fare": "Base Fare",
        },
    )
    st.dataframe(copy_display, use_container_width=True, hide_index=True)

    labeled_divider("Recommended Actions")
    render_decision_card(
        title="Campaign Creative Next Steps",
        metrics={
            "Route": route,
            "Cabin": cabin,
            "Segment": segment,
            "Best Score": int(best_subject["score"]),
        },
        actions=[
            "Use the top-scoring subject line as the control in the next send",
            "Pair the winning subject line with the strongest headline and CTA combination",
            "Localise the offer and fare anchor before handing off to media teams",
        ],
        impact=f"Sharper creative increases the odds of converting demand already identified in the platform for {route}.",
        urgency="Next campaign brief",
        color="opportunity",
    )
