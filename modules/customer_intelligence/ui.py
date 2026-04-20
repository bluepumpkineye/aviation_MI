"""
Streamlit UI for the customer intelligence module.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config.brand import CHART_COLORS, COLORS
from modules.customer_intelligence.churn_model import train_churn_model
from modules.customer_intelligence.clv_model import train_clv_model
from modules.customer_intelligence.segmentation import (
    add_rfm_scores,
    segment_market_matrix,
    segment_profile_summary,
    top_customer_opportunities,
)
from utils.chart_helpers import (
    apply_theme,
    format_currency,
    format_dataframe_display,
    format_percentage,
    labeled_divider,
    render_metric_strip,
    render_page_header,
)
from utils.insight_renderer import (
    render_decision_card,
    render_insight_banner,
    render_what_this_means,
)


@st.cache_data(show_spinner=False)
def _cached_segment_profile(df: pd.DataFrame) -> pd.DataFrame:
    return segment_profile_summary(df)


@st.cache_data(show_spinner=False)
def _cached_segment_market_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return segment_market_matrix(df)


@st.cache_data(show_spinner=False)
def _cached_rfm(df: pd.DataFrame) -> pd.DataFrame:
    return add_rfm_scores(df)


@st.cache_data(show_spinner=False)
def _cached_clv(df: pd.DataFrame) -> dict:
    return train_clv_model(df)


@st.cache_data(show_spinner=False)
def _cached_churn(df: pd.DataFrame) -> dict:
    return train_churn_model(df)


def _segment_action_groups(df_customers: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df_customers.copy()
    remaining = pd.Series(True, index=df.index)

    champions = df[(df["customer_segment"] == "High Value Loyalists") | (df["loyalty_tier"] == "Diamond")]
    remaining.loc[champions.index] = False

    loyal = df[remaining & ((df["customer_segment"] == "Frequent Flyers") | ((df["trips_last_12m"] >= 4) & (df["churn_probability_90d"] < 0.55)))]
    remaining.loc[loyal.index] = False

    at_risk = df[remaining & ((df["churn_probability_90d"] >= 0.70) & (df["trips_last_12m"] >= 2))]
    remaining.loc[at_risk.index] = False

    dormant = df[remaining & ((df["days_since_last_booking"] >= 180) | (df["trips_last_12m"] == 0))]
    remaining.loc[dormant.index] = False

    occasional = df[remaining]

    return {
        "Champions": champions,
        "Loyal Travellers": loyal,
        "At-Risk Frequent": at_risk,
        "Occasional Flyers": occasional,
        "Dormant Members": dormant,
    }


def _card_metrics(segment_df: pd.DataFrame) -> tuple[int, float, float]:
    count = len(segment_df)
    avg_clv = float(segment_df["customer_lifetime_value_12m"].mean()) if count else 0.0
    churn = float(segment_df["churn_probability_90d"].mean()) if count else 0.0
    return count, avg_clv, churn


def render(
    df_customers: pd.DataFrame,
    df_campaigns: pd.DataFrame | None = None,
    df_demand: pd.DataFrame | None = None,
    df_digital: pd.DataFrame | None = None,
    insights: dict | None = None,
):
    """Render the customer intelligence dashboard."""
    render_page_header(
        title="Customer Intelligence",
        subtitle="Segment member value, prioritise churn intervention, and identify where the CRM team should act next.",
        pills=["XGBoost CLV", "LightGBM Churn", "K-Means RFM", "SHAP"],
        meta=f"Updated {pd.Timestamp.today().strftime('%d %b %Y')}",
    )

    customer_insights = []
    if insights:
        all_customer = [
            item
            for bucket in ["critical", "opportunity", "on_track", "cross_module"]
            for item in insights.get(bucket, [])
            if "Customer Intelligence" in item["module"]
        ]
        for keyword in ["Churn", "Miles"]:
            match = next((item for item in all_customer if keyword.lower() in item["title"].lower()), None)
            if match:
                customer_insights.append(match)

    if customer_insights:
        labeled_divider("AI Insights")
        for insight in customer_insights:
            render_insight_banner(insight)

    segment_summary = _cached_segment_profile(df_customers)
    rfm_df = _cached_rfm(df_customers)
    clv_artifacts = _cached_clv(df_customers)
    churn_artifacts = _cached_churn(df_customers)
    scored = churn_artifacts["scored_customers"].copy()
    scored["revenue_at_risk_hkd"] = scored["customer_lifetime_value_12m"] * scored["predicted_churn_risk"]
    high_risk_count = int((scored["predicted_churn_risk"] >= 0.70).sum())
    revenue_at_risk = float(scored.loc[scored["predicted_churn_risk"] >= 0.70, "revenue_at_risk_hkd"].sum())

    avg_clv = float(df_customers["customer_lifetime_value_12m"].mean())
    median_clv = float(df_customers["customer_lifetime_value_12m"].median())
    labeled_divider("Performance Analysis")
    render_metric_strip(
        [
            {"label": "Total Members", "value": f"{len(df_customers):,}"},
            {"label": "Average CLV", "value": format_currency(avg_clv), "delta": f"vs median {format_currency(median_clv)}"},
            {"label": "Revenue at Risk", "value": format_currency(revenue_at_risk), "delta": f"{high_risk_count:,} members >70% risk", "delta_state": "negative"},
            {"label": "High-Risk Members", "value": f"{high_risk_count:,}", "delta": format_percentage(df_customers["churn_probability_90d"].mean()), "delta_state": "negative"},
        ],
        columns=4,
    )

    seg_tab, clv_tab, churn_tab = st.tabs(["Segmentation", "CLV and Value", "Churn Priority"])

    with seg_tab:
        left, right = st.columns([1.05, 0.95])
        fig = px.bar(
            segment_summary,
            x="customer_segment",
            y="customers",
            color="avg_clv_hkd",
            color_continuous_scale=["#173B61", COLORS["primary"]],
            title="Customer Segments by Size and Value",
        )
        fig.update_layout(coloraxis_showscale=False, height=340)
        apply_theme(fig)
        left.plotly_chart(fig, use_container_width=True)

        heatmap_df = _cached_segment_market_matrix(df_customers)
        fig = px.imshow(
            heatmap_df,
            color_continuous_scale=["#10213E", COLORS["accent"]],
            aspect="auto",
            title="Segment Concentration by Home Market",
        )
        fig.update_layout(height=340)
        apply_theme(fig)
        right.plotly_chart(fig, use_container_width=True)

        segment_display = segment_summary.copy()
        segment_display["revenue_at_stake"] = (
            segment_display["customers"] * segment_display["avg_clv_hkd"] * segment_display["avg_churn_prob"]
        )
        segment_display = segment_display[
            [
                "customer_segment",
                "customers",
                "avg_clv_hkd",
                "avg_churn_prob",
                "avg_trips",
                "avg_engagement",
                "revenue_at_stake",
            ]
        ]
        styled_segment = format_dataframe_display(
            segment_display,
            currency_cols=["avg_clv_hkd", "revenue_at_stake"],
            pct_cols=["avg_churn_prob"],
            rename_map={
                "customer_segment": "Segment",
                "customers": "Members",
                "avg_clv_hkd": "Avg CLV (HK$)",
                "avg_churn_prob": "Churn Risk",
                "avg_trips": "Avg Trips/Year",
                "avg_engagement": "Engagement Score",
                "revenue_at_stake": "Revenue at Stake",
            },
        )
        st.dataframe(
            styled_segment.style.set_properties(
                subset=["Revenue at Stake"],
                **{"color": "#C4973B", "font-weight": "700"},
            ),
            use_container_width=True,
            hide_index=True,
        )

        rfm_summary = (
            rfm_df.groupby("customer_segment")[["r_score", "f_score", "m_score"]]
            .mean()
            .round(2)
            .reset_index()
        )
        rfm_display = format_dataframe_display(
            rfm_summary,
            rename_map={
                "customer_segment": "Segment",
                "r_score": "Recency Score",
                "f_score": "Frequency Score",
                "m_score": "Monetary Score",
            },
        )
        st.dataframe(rfm_display, use_container_width=True, hide_index=True)

    with clv_tab:
        top_left, top_right = st.columns([0.85, 1.15])
        metrics = clv_artifacts["metrics"]
        top_left.metric("Model RMSE", format_currency(metrics["rmse"]), delta=f"MAPE {metrics['mape']:.1f}%")
        top_left.metric("Model MAE", format_currency(metrics["mae"]))
        top_left.metric("Model R²", f"{metrics['r2']:.3f}")
        top_left.metric("Avg Member Value", format_currency(avg_clv))

        fig = px.bar(
            clv_artifacts["feature_importance"],
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#1E3A5F", COLORS["warning"]],
            title="CLV Feature Importance",
        )
        fig.update_layout(coloraxis_showscale=False, height=340)
        apply_theme(fig)
        top_right.plotly_chart(fig, use_container_width=True)

        scatter_source = df_customers.copy()
        clv_threshold = float(scatter_source["customer_lifetime_value_12m"].median())
        churn_threshold = float(scatter_source["churn_probability_90d"].median())
        urgent_zone = scatter_source[
            (scatter_source["customer_lifetime_value_12m"] >= clv_threshold)
            & (scatter_source["churn_probability_90d"] >= churn_threshold)
        ].copy()
        urgent_zone["revenue_at_risk_hkd"] = (
            urgent_zone["customer_lifetime_value_12m"] * urgent_zone["churn_probability_90d"]
        )
        fig = px.scatter(
            scatter_source.sample(n=min(1200, len(scatter_source)), random_state=42),
            x="customer_lifetime_value_12m",
            y="churn_probability_90d",
            color="customer_segment",
            color_discrete_sequence=CHART_COLORS,
            opacity=0.68,
            title="CLV vs Churn Risk",
        )
        fig.add_vline(x=clv_threshold, line_dash="dash", line_color=COLORS["muted_text"])
        fig.add_hline(y=churn_threshold, line_dash="dash", line_color=COLORS["muted_text"])
        fig.update_layout(height=380, xaxis_title="CLV (HK$)", yaxis_title="Churn Risk")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        render_what_this_means(
            text=(
                f"Customers in the top-right quadrant carry both high value and high churn risk. "
                f"The {len(urgent_zone):,} customers in this zone represent {format_currency(urgent_zone['revenue_at_risk_hkd'].sum())} in at-risk revenue."
            ),
            action="Filter this list and brief your CRM team today",
        )

    with churn_tab:
        fig = px.histogram(
            scored,
            x="predicted_churn_risk",
            nbins=24,
            color="risk_band",
            color_discrete_sequence=CHART_COLORS,
            title="Predicted Churn Risk Distribution",
        )
        fig.update_layout(height=320, bargap=0.05)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns([0.8, 1.2])
        churn_metrics = churn_artifacts["metrics"]
        left.metric("Accuracy", f"{churn_metrics['accuracy']:.3f}")
        left.metric("Precision", f"{churn_metrics['precision']:.3f}")
        left.metric("Recall", f"{churn_metrics['recall']:.3f}")
        left.metric("ROC AUC", f"{churn_metrics.get('roc_auc', 0):.3f}")

        fig = px.bar(
            churn_artifacts["feature_importance"],
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#173B61", COLORS["danger"]],
            title="Churn Model Feature Importance",
        )
        fig.update_layout(coloraxis_showscale=False, height=320)
        apply_theme(fig)
        right.plotly_chart(fig, use_container_width=True)

        priority = scored.sort_values(
            ["predicted_churn_risk", "customer_lifetime_value_12m"],
            ascending=[False, False],
        )[
            [
                "customer_id",
                "customer_segment",
                "home_market",
                "loyalty_tier",
                "customer_lifetime_value_12m",
                "predicted_churn_risk",
                "risk_band",
            ]
        ].head(25)
        priority_display = format_dataframe_display(
            priority,
            currency_cols=["customer_lifetime_value_12m"],
            pct_cols=["predicted_churn_risk"],
            rename_map={
                "customer_id": "Customer ID",
                "customer_segment": "Segment",
                "home_market": "Home Market",
                "loyalty_tier": "Loyalty Tier",
                "customer_lifetime_value_12m": "CLV (HK$)",
                "predicted_churn_risk": "Predicted Churn Risk",
                "risk_band": "Risk Band",
            },
        )
        st.dataframe(priority_display, use_container_width=True, hide_index=True)

    labeled_divider("Recommended Actions")
    st.markdown("### Recommended Actions by Segment")
    segment_groups = _segment_action_groups(df_customers)
    card_specs = [
        (
            "Champions",
            [
                "Offer early access to new routes or premium experiences",
                "Send personalised milestone recognition (anniversary, tier upgrade)",
                "Invite to feedback panel - they are your brand advocates",
            ],
            "Retaining Champions at current rate protects {impact} annually",
            "Ongoing - quarterly touchpoint minimum",
            "on_track",
            0.20,
        ),
        (
            "Loyal Travellers",
            [
                "Status acceleration campaign - show progress to next tier",
                "Double miles on next 2 bookings promotion",
                "Exclusive preview of new routes before public launch",
            ],
            "Converting 20% to Champions adds {impact} CLV",
            "30 days - before competitors target this segment",
            "opportunity",
            0.20,
        ),
        (
            "At-Risk Frequent",
            [
                "Personalised win-back email - 15% fare discount + miles bonus",
                "Miles expiry extension - automatic 90-day grace period",
                "Personal call from relationship manager for top 50 by CLV",
            ],
            "Recovering 25% prevents {impact} revenue loss",
            "IMMEDIATE - 14 days maximum",
            "critical",
            0.25,
        ),
        (
            "Occasional Flyers",
            [
                "Destination inspiration campaign - YouTube and social",
                "Family package bundle offer - school holiday timing",
                "Miles earning awareness - many do not know their balance",
            ],
            "Increasing frequency by 1 trip adds {impact} revenue",
            "60 days - seasonal opportunity",
            "opportunity",
            0.10,
        ),
        (
            "Dormant Members",
            [
                "\"We miss you\" email - 40% discount, no blackout dates",
                "Miles reactivation - use-or-lose urgency message",
                "Survey - understand why they stopped flying",
            ],
            "Reactivating 10% recovers {impact} in lost revenue",
            "30 days - before miles expire permanently",
            "critical" if len(segment_groups["Dormant Members"]) > 1000 else "opportunity",
            0.10,
        ),
    ]
    card_cols = st.columns(2)
    for idx, (segment_name, actions, impact_template, urgency, color, rate) in enumerate(card_specs):
        segment_df = segment_groups[segment_name]
        count, avg_segment_clv, churn = _card_metrics(segment_df)
        impact_value = count * avg_segment_clv * rate
        metrics = {
            "Members": f"{count:,}",
            "Avg CLV": format_currency(avg_segment_clv),
            "Churn Risk": format_percentage(churn),
        }
        with card_cols[idx % 2]:
            render_decision_card(
                title=segment_name,
                metrics=metrics,
                actions=actions,
                impact=impact_template.format(impact=format_currency(impact_value)),
                urgency=urgency,
                color=color,
            )
