"""
Streamlit UI for the marketing mix module.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config.brand import COLORS
from modules.marketing_mix.budget_optimizer import recommend_budget_allocation
from modules.marketing_mix.mmm_model import channel_efficiency_summary, fit_mmm_model
from modules.marketing_mix.scenario_simulator import channel_response_curve, simulate_budget_scenarios
from modules.marketing_mix.shapley_attribution import channel_contribution_summary
from utils.chart_helpers import (
    apply_theme,
    format_currency,
    format_dataframe_display,
    labeled_divider,
    render_metric_grid,
    render_page_header,
)
from utils.insight_renderer import render_insight_banner, render_what_this_means


@st.cache_data(show_spinner=False)
def _cached_mmm(df_campaigns: pd.DataFrame) -> dict:
    return fit_mmm_model(df_campaigns)


@st.cache_data(show_spinner=False)
def _cached_efficiency(df_campaigns: pd.DataFrame) -> pd.DataFrame:
    return channel_efficiency_summary(df_campaigns)


@st.cache_data(show_spinner=False)
def _cached_contributions(df_campaigns: pd.DataFrame) -> pd.DataFrame:
    return channel_contribution_summary(df_campaigns)


def _decision_label(row: pd.Series, portfolio_avg_roas: float) -> tuple[str, str]:
    spend_share = row["spend_hkd"] / max(row["spend_hkd_total"], 1)
    if row["channel"] == "Email CRM" and spend_share < 0.05:
        return "Underinvested - Increase Now", "#00D4AA"
    if row["roas"] > portfolio_avg_roas * 1.3:
        return "Scale Up - High Efficiency", "#00D4AA"
    if portfolio_avg_roas * 0.8 < row["roas"] <= portfolio_avg_roas * 1.3:
        return "Maintain - On Target", "#00A3A1"
    if row["roas"] < portfolio_avg_roas * 0.8 and spend_share > 0.15:
        return "Reduce - Underperforming at Scale", "#E76F51"
    return "Optimise - Monitor Closely", "#C4973B"


def render(
    df_customers: pd.DataFrame | None,
    df_campaigns: pd.DataFrame,
    df_demand: pd.DataFrame | None = None,
    df_digital: pd.DataFrame | None = None,
    insights: dict | None = None,
):
    """Render the marketing mix dashboard."""
    render_page_header(
        title="Marketing Mix",
        subtitle="Compare channel efficiency, quantify reallocation upside, and pressure-test budget decisions before the next planning cycle.",
        pills=["Media Mix Modelling", "SHAP Attribution", "Scenario Simulation", "Budget Rules"],
        meta=f"Updated {pd.Timestamp.today().strftime('%d %b %Y')}",
    )

    marketing_insights = []
    if insights:
        all_marketing = [
            item
            for bucket in ["critical", "opportunity", "on_track", "cross_module"]
            for item in insights.get(bucket, [])
            if "Marketing Mix" in item["module"]
        ]
        for keyword in ["ROAS", "Email"]:
            match = next(
                (
                    item
                    for item in all_marketing
                    if keyword.lower() in item["title"].lower() or keyword.lower() in item["body"].lower()
                ),
                None,
            )
            if match:
                marketing_insights.append(match)

    if marketing_insights:
        labeled_divider("AI Insights")
        for insight in marketing_insights[:2]:
            render_insight_banner(insight)

    artifacts = _cached_mmm(df_campaigns)
    efficiency = _cached_efficiency(df_campaigns).copy()
    contributions = _cached_contributions(df_campaigns)
    portfolio_avg_roas = float(efficiency["revenue_hkd"].sum() / max(efficiency["spend_hkd"].sum(), 1))
    best_channel = efficiency.sort_values("roas", ascending=False).iloc[0]
    biggest_spend_channel = efficiency.sort_values("spend_hkd", ascending=False).iloc[0]
    worst_channel = efficiency.sort_values("roas", ascending=True).iloc[0]

    labeled_divider("Performance Analysis")
    render_metric_grid(
        [
            {"label": "MMM R²", "value": f"{artifacts['metrics']['r2']:.3f}", "delta": "Model fit", "delta_state": "neutral"},
            {"label": "Portfolio ROAS", "value": f"{portfolio_avg_roas:.1f}x", "delta": f"Best {best_channel['channel']} at {best_channel['roas']:.1f}x", "delta_state": "positive"},
            {"label": "Total Spend", "value": format_currency(efficiency["spend_hkd"].sum()), "delta": format_currency(efficiency["revenue_hkd"].sum()), "delta_state": "positive"},
            {"label": "Channels Analysed", "value": f"{efficiency['channel'].nunique():,}", "delta": f"Worst: {worst_channel['channel']}", "delta_state": "negative"},
        ],
        columns=4,
    )

    mix_tab, budget_tab, scenario_tab, shap_tab = st.tabs(
        ["MMM Performance", "Budget Optimizer", "Scenario Simulator", "Shapley Attribution"]
    )

    with mix_tab:
        left, right = st.columns([1.0, 1.0])
        fig = px.bar(
            efficiency.sort_values("roas"),
            x="roas",
            y="channel",
            orientation="h",
            color="roas",
            color_continuous_scale=["#173B61", COLORS["success"]],
            title="Historical ROAS by Channel",
        )
        fig.update_layout(coloraxis_showscale=False, height=330, xaxis_title="ROAS (x)")
        apply_theme(fig)
        left.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            artifacts["actual_vs_pred"],
            x="actual_revenue_hkd",
            y="predicted_revenue_hkd",
            opacity=0.7,
            title="MMM Actual vs Predicted Revenue",
            color_discrete_sequence=[COLORS["accent"]],
        )
        fig.update_layout(height=330, xaxis_title="Actual Revenue", yaxis_title="Predicted Revenue")
        apply_theme(fig)
        right.plotly_chart(fig, use_container_width=True)

        roas_gap = best_channel["roas"] - biggest_spend_channel["roas"]
        render_what_this_means(
            text=(
                f"Your top channel ({best_channel['channel']}) delivers {best_channel['roas']:.1f}x ROAS while your largest spend channel "
                f"({biggest_spend_channel['channel']}) delivers only {biggest_spend_channel['roas']:.1f}x. "
                f"This gap of {roas_gap:.1f}x represents a significant reallocation opportunity."
            ),
            action="Use the Budget Optimizer tab to quantify the exact reallocation",
        )

        channel_display = format_dataframe_display(
            efficiency[
                ["channel", "spend_hkd", "revenue_hkd", "roas", "conversion_rate", "ctr", "revenue_per_booking"]
            ],
            currency_cols=["spend_hkd", "revenue_hkd", "revenue_per_booking"],
            pct_cols=["conversion_rate", "ctr"],
            rename_map={
                "channel": "Channel",
                "spend_hkd": "Spend",
                "revenue_hkd": "Revenue",
                "roas": "ROAS",
                "conversion_rate": "Conversion Rate",
                "ctr": "CTR",
                "revenue_per_booking": "Revenue / Booking",
            },
        )
        st.dataframe(channel_display, use_container_width=True, hide_index=True)

        labeled_divider("Recommended Actions")
        st.markdown("### Channel Investment Decisions")
        decision_df = efficiency.copy()
        decision_df["spend_hkd_total"] = decision_df["spend_hkd"].sum()
        decisions = decision_df.apply(lambda row: _decision_label(row, portfolio_avg_roas), axis=1)
        decision_df["decision"] = [item[0] for item in decisions]
        decision_df["decision_color"] = [item[1] for item in decisions]
        decision_df["spend_share"] = decision_df["spend_hkd"] / max(decision_df["spend_hkd"].sum(), 1)
        display = format_dataframe_display(
            decision_df[["channel", "spend_hkd", "roas", "spend_share", "decision"]],
            currency_cols=["spend_hkd"],
            pct_cols=["spend_share"],
            rename_map={
                "channel": "Channel",
                "spend_hkd": "Spend",
                "roas": "ROAS",
                "spend_share": "Spend Share",
                "decision": "Decision",
            },
        )
        st.dataframe(
            display.style.apply(
                lambda row: [
                    f"color: {decision_df.loc[row.name, 'decision_color']}; font-weight:700" if col == "Decision" else ""
                    for col in display.columns
                ],
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

    with budget_tab:
        total_budget = st.number_input(
            "Planned Budget HKD",
            min_value=500000,
            value=4000000,
            step=100000,
        )
        allocation = recommend_budget_allocation(df_campaigns, total_budget).copy()
        allocation["delta_hkd"] = allocation["recommended_budget_hkd"] - allocation["spend_hkd"]

        compare = allocation[["channel", "spend_hkd", "recommended_budget_hkd"]].melt(
            id_vars="channel",
            var_name="budget_type",
            value_name="budget_hkd",
        )
        fig = px.bar(
            compare,
            x="channel",
            y="budget_hkd",
            color="budget_type",
            barmode="group",
            color_discrete_sequence=[COLORS["secondary"], COLORS["primary"]],
            title="Current vs Recommended Budget Allocation",
        )
        fig.update_layout(height=360, xaxis_title="", yaxis_title="Budget (HK$)")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        current_projected_revenue = float((allocation["spend_hkd"] * allocation["roas"]).sum())
        optimal_projected_revenue = float((allocation["recommended_budget_hkd"] * allocation["roas"]).sum())
        avg_booking_value = float(df_campaigns["avg_booking_value"].mean())
        revenue_uplift = max(optimal_projected_revenue - current_projected_revenue, 0)
        bookings_uplift = int(round(revenue_uplift / max(avg_booking_value, 1), 0))
        best_delta = allocation.sort_values("delta_hkd", ascending=False).iloc[0]
        worst_delta = allocation.sort_values("delta_hkd", ascending=True).iloc[0]
        amount = abs(float(worst_delta["delta_hkd"]))

        st.markdown(
            f"""
            <div style="background:#0F2040; border-left:6px solid #C4973B; border-radius:10px; padding:16px; margin:10px 0 14px 0;">
                <div style="font-size:0.82rem; text-transform:uppercase; letter-spacing:0.08em; color:#C4973B; margin-bottom:8px;">
                    Reallocation Summary
                </div>
                <div style="color:#FFFFFF; line-height:1.6;">
                    Moving {format_currency(amount)} from {worst_delta['channel']} to {best_delta['channel']}
                    is projected to generate {bookings_uplift:,} additional bookings worth approximately
                    {format_currency(revenue_uplift)} in incremental revenue.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        allocation_display = format_dataframe_display(
            allocation[
                [
                    "channel",
                    "spend_hkd",
                    "recommended_budget_hkd",
                    "delta_hkd",
                    "roas",
                    "conversion_rate",
                ]
            ],
            currency_cols=["spend_hkd", "recommended_budget_hkd", "delta_hkd"],
            pct_cols=["conversion_rate"],
            rename_map={
                "channel": "Channel",
                "spend_hkd": "Current Budget",
                "recommended_budget_hkd": "Recommended Budget",
                "delta_hkd": "Budget Delta",
                "roas": "ROAS",
                "conversion_rate": "Conversion Rate",
            },
        )
        st.dataframe(allocation_display, use_container_width=True, hide_index=True)

    with scenario_tab:
        left, right = st.columns([0.85, 1.15])
        scenarios = simulate_budget_scenarios(df_campaigns, float(total_budget))
        scenario_display = format_dataframe_display(
            scenarios,
            currency_cols=["budget_hkd", "projected_revenue_hkd"],
            rename_map={
                "scenario": "Scenario",
                "budget_hkd": "Budget",
                "projected_revenue_hkd": "Projected Revenue",
                "projected_bookings": "Projected Bookings",
                "projected_roas": "Projected ROAS",
            },
        )
        left.dataframe(scenario_display, use_container_width=True, hide_index=True)

        source_channel = right.selectbox(
            "Shift Budget From",
            efficiency["channel"].tolist(),
            index=efficiency["channel"].tolist().index(biggest_spend_channel["channel"]),
        )
        target_channel = right.selectbox(
            "Shift Budget To",
            efficiency["channel"].tolist(),
            index=efficiency["channel"].tolist().index(best_channel["channel"]),
        )
        shift_amount = right.slider("Budget Shift", min_value=100000, max_value=2000000, value=500000, step=50000)
        response_curve = channel_response_curve(df_campaigns)
        curve = response_curve[response_curve["channel"] == target_channel]
        fig = px.line(
            curve,
            x="pct_change",
            y="projected_revenue_hkd",
            markers=True,
            title=f"Response Curve: {target_channel}",
            color_discrete_sequence=[COLORS["warning"]],
        )
        fig.update_layout(height=330, xaxis_title="Spend Change (%)", yaxis_title="Projected Revenue")
        apply_theme(fig)
        right.plotly_chart(fig, use_container_width=True)

        source_roas = float(efficiency.loc[efficiency["channel"] == source_channel, "roas"].iloc[0])
        target_roas = float(efficiency.loc[efficiency["channel"] == target_channel, "roas"].iloc[0])
        revenue_delta = shift_amount * max(target_roas - source_roas, 0)
        if target_roas > portfolio_avg_roas:
            positive_outcome = f"incremental revenue could reach {format_currency(revenue_delta)} if the stronger channel scales efficiently"
        else:
            positive_outcome = "the move would mainly improve diversification rather than immediate returns"
        source_spend_share = float(
            efficiency.loc[efficiency["channel"] == source_channel, "spend_hkd"].iloc[0] / efficiency["spend_hkd"].sum()
        )
        if source_spend_share > 0.20:
            risk_text = f"reducing {source_channel} too fast may cut reach because it still carries meaningful volume"
        else:
            risk_text = f"{source_channel} is already a smaller line item, so monitor whether the test is large enough to measure cleanly"
        render_what_this_means(
            text=(
                f"Shifting {format_currency(shift_amount)} from {source_channel} to {target_channel}: "
                f"If successful, {positive_outcome}. Risk to monitor: {risk_text}."
            )
        )

    with shap_tab:
        fig = px.bar(
            contributions.sort_values("contribution_pct"),
            x="contribution_pct",
            y="channel",
            orientation="h",
            color="contribution_pct",
            color_continuous_scale=["#173B61", COLORS["accent"]],
            title="SHAP-Based Channel Contribution Share",
        )
        fig.update_layout(coloraxis_showscale=False, height=340, xaxis_title="Contribution (%)", yaxis_title="")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        contribution_display = format_dataframe_display(
            contributions,
            pct_cols=["contribution_share", "contribution_pct"],
            rename_map={
                "channel": "Channel",
                "mean_abs_shap": "Mean |SHAP|",
                "contribution_share": "Contribution Share",
                "contribution_pct": "Contribution %",
            },
        )
        st.dataframe(contribution_display, use_container_width=True, hide_index=True)
