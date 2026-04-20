"""
Streamlit UI for AI governance and model operations.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config.brand import CHART_COLORS, COLORS
from modules.ai_governance.audit_log import get_audit_log
from modules.ai_governance.model_monitor import (
    current_model_health,
    data_quality_checks,
    generate_model_history,
)
from utils.chart_helpers import (
    apply_theme,
    format_dataframe_display,
    labeled_divider,
    render_metric_grid,
    render_page_header,
)
from utils.insight_renderer import render_decision_card, render_what_this_means


def render(
    df_customers: pd.DataFrame,
    df_campaigns: pd.DataFrame,
    df_demand: pd.DataFrame,
    df_digital: pd.DataFrame,
):
    """Render the AI governance dashboard."""
    render_page_header(
        title="AI Governance",
        subtitle="Monitor model health, control data quality, and keep decision workflows auditable across the platform.",
        pills=["Model Monitoring", "Drift Thresholds", "Data Quality Rules", "Audit Logging"],
        meta=f"Updated {pd.Timestamp.today().strftime('%d %b %Y')}",
    )

    health = current_model_health()
    history = generate_model_history()
    log = get_audit_log()

    labeled_divider("Model Intelligence")
    render_metric_grid(
        [
            {"label": "Models Healthy", "value": f"{int(health['healthy'].sum())}/{len(health)}", "delta": "Current fleet", "delta_state": "positive"},
            {"label": "Models Needing Review", "value": f"{int((~health['healthy']).sum()):,}", "delta": "Review queue", "delta_state": "negative" if int((~health['healthy']).sum()) else "neutral"},
            {"label": "Audit Events Logged", "value": f"{len(log):,}", "delta": "Traceability", "delta_state": "neutral"},
        ],
        columns=3,
    )

    fig = px.line(
        history,
        x="week",
        y="value",
        color="model",
        markers=True,
        color_discrete_sequence=CHART_COLORS,
        title="Model Performance History",
    )
    fig.update_layout(height=380, xaxis_title="Week", yaxis_title="Metric Value")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    alert_models = health.loc[~health["healthy"], "model"].tolist()
    if alert_models:
        render_what_this_means(
            text=(
                f"{', '.join(alert_models)} are trending outside their preferred thresholds. "
                "Governance attention should focus on whether the issue is model drift, data drift, or stale retraining cadence."
            ),
            action="Prioritise retraining and validation for the models flagged above",
        )

    health_display = format_dataframe_display(
        health,
        rename_map={
            "model": "Model",
            "metric": "Primary Metric",
            "value": "Current Value",
            "threshold": "Threshold",
            "healthy": "Healthy",
            "status": "Status",
        },
    )
    st.dataframe(health_display, use_container_width=True, hide_index=True)

    labeled_divider("Data Quality")
    selected_dataset = st.selectbox(
        "Dataset Quality Checks",
        ["Customers", "Campaigns", "Route Demand", "Digital Touchpoints"],
    )
    dataset_map = {
        "Customers": df_customers,
        "Campaigns": df_campaigns,
        "Route Demand": df_demand,
        "Digital Touchpoints": df_digital,
    }
    checks = data_quality_checks(dataset_map[selected_dataset], selected_dataset)
    q1, q2, q3 = st.columns(3)
    q1.metric("Columns Checked", len(checks))
    q2.metric("Passing Checks", int((checks["Status"] == "OK").sum()))
    q3.metric("Issues", int((checks["Status"] == "Issue").sum()))
    checks_display = format_dataframe_display(
        checks,
        rename_map={
            "Dataset": "Dataset",
            "Column": "Column",
            "Null Rate": "Null Rate",
            "Distinct Values": "Distinct Values",
            "Status": "Status",
            "Comment": "Comment",
        },
    )
    st.dataframe(checks_display, use_container_width=True, hide_index=True)

    labeled_divider("Audit Trail")
    st.dataframe(
        format_dataframe_display(
            log,
            rename_map={
                "timestamp": "Timestamp",
                "module": "Module",
                "action": "Action",
                "actor": "Actor",
                "details": "Details",
            },
        ),
        use_container_width=True,
        hide_index=True,
    )

    render_decision_card(
        title="Governance Priority",
        metrics={
            "Healthy Models": f"{int(health['healthy'].sum())}/{len(health)}",
            "Datasets Checked": 4,
            "Audit Events": len(log),
        },
        actions=[
            "Retrain the models marked as needing review before the next planning cycle",
            "Investigate any dataset columns with elevated null rates or low-cardinality issues",
            "Keep audit logging active for every major decision generated by the platform",
        ],
        impact="Tighter governance reduces the risk of acting on stale or misleading model output.",
        urgency="Weekly governance review",
        color="opportunity" if alert_models else "on_track",
    )
