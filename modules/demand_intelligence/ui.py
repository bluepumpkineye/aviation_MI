"""
Streamlit UI for demand intelligence.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.brand import COLORS
from config.settings import CABIN_CLASSES, ROUTES
from modules.demand_intelligence.booking_window import (
    booking_window_summary,
    conversion_by_lead_time,
    generate_booking_window_data,
    optimal_campaign_lead_time,
)
from modules.demand_intelligence.campaign_timing import (
    channel_timing_recommendations,
    generate_campaign_calendar,
    route_timing_benchmarks,
)
from modules.demand_intelligence.route_forecaster import (
    campaign_window_recommendations,
    ensemble_forecast,
)
from utils.chart_helpers import (
    apply_theme,
    format_dataframe_display,
    labeled_divider,
    render_metric_grid,
    render_page_header,
)
from utils.insight_renderer import render_insight_banner, render_what_this_means


@st.cache_data(show_spinner=False)
def _cached_booking_windows() -> pd.DataFrame:
    return generate_booking_window_data()


def render(
    df_customers: pd.DataFrame | None,
    df_campaigns: pd.DataFrame | None,
    df_demand: pd.DataFrame,
    df_digital: pd.DataFrame | None = None,
    insights: dict | None = None,
):
    """Render the demand intelligence dashboard."""
    demand = df_demand.copy()
    demand["date"] = pd.to_datetime(demand["date"])
    bookings_df = _cached_booking_windows()

    render_page_header(
        title="Demand Intelligence",
        subtitle="Forecast route demand, time campaign launches against booking behaviour, and align spend to upcoming peaks.",
        pills=["Prophet Forecasting", "Booking Windows", "Seasonality Benchmarks", "Campaign Timing"],
        meta=f"Updated {pd.Timestamp.today().strftime('%d %b %Y')}",
    )

    demand_insights = []
    if insights:
        demand_insights = [
            item
            for bucket in ["critical", "opportunity", "on_track", "cross_module"]
            for item in insights.get(bucket, [])
            if "Demand Intelligence" in item["module"]
        ][:2]
    if demand_insights:
        labeled_divider("AI Insights")
        for item in demand_insights:
            render_insight_banner(item)

    route_col, cabin_col, horizon_col = st.columns([1.2, 1.0, 0.8])
    route = route_col.selectbox("Route", ROUTES, index=0)
    cabin = cabin_col.selectbox("Cabin", CABIN_CLASSES, index=2)
    horizon = horizon_col.slider("Forecast Days", min_value=30, max_value=90, value=45, step=15)

    forecast_df, history_df, mape_value = ensemble_forecast(demand, route, periods=horizon)
    history_view = history_df.tail(120).copy()
    latest_data_date = pd.to_datetime(demand["date"].max())
    recommendation = optimal_campaign_lead_time(bookings_df, route, cabin)
    peak_row = forecast_df.loc[forecast_df["ensemble_forecast"].idxmax()]
    peak_date = pd.to_datetime(peak_row["date"])
    days_to_peak = int((peak_date - latest_data_date).days)
    launch_lead = int(
        recommendation.get(
            "recommended_launch_days_before_departure",
            recommendation.get("recommended_launch", 30),
        )
    )
    recommended_launch_date = peak_date - pd.Timedelta(days=launch_lead)
    days_until_launch = int((recommended_launch_date - latest_data_date).days)

    labeled_divider("Forecast Intelligence")
    if days_to_peak <= 14:
        st.error(
            f"URGENT: Peak demand for {route} in {days_to_peak} days. Campaign must be live now."
        )
    elif days_to_peak <= 30:
        st.warning(
            f"Window closing: Peak demand in {days_to_peak} days. Launch within {max(days_until_launch, 0)} days."
        )
    elif days_to_peak <= 60:
        st.info(
            f"Plan ahead: Peak demand for {route} expected in {days_to_peak} days. Optimal campaign launch: {recommended_launch_date.strftime('%d %b %Y')}."
        )
    else:
        st.success(
            f"On track: Next demand peak in {days_to_peak} days. Begin campaign planning now."
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_view["date"],
            y=history_view["bookings"],
            mode="lines",
            name="Historical Bookings",
            line={"color": COLORS["secondary"], "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["upper_90"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["lower_90"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(41, 182, 246, 0.15)",
            name="90% Interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["ensemble_forecast"],
            mode="lines",
            name="Ensemble Forecast",
            line={"color": COLORS["primary"], "width": 3},
        )
    )
    apply_theme(fig, f"Route Demand Forecast: {route}")
    fig.update_layout(height=430, xaxis_title="", yaxis_title="Bookings / Day")
    st.plotly_chart(fig, use_container_width=True)

    first7_avg = float(forecast_df["ensemble_forecast"].head(7).mean())
    last7_avg = float(forecast_df["ensemble_forecast"].tail(7).mean())
    trend_pct = ((last7_avg - first7_avg) / max(first7_avg, 1)) * 100
    if trend_pct > 5:
        trend = "increase"
        action_text = "Rising demand signals an opportunity to invest in marketing now before competitors react."
    elif trend_pct < -5:
        trend = "decrease"
        action_text = "Softening demand suggests focusing spend on other routes or considering promotional fares."
    else:
        trend = "remain stable"
        action_text = "Stable demand means timing and message quality matter more than incremental spend."
    render_what_this_means(
        text=(
            f"This forecast shows {route} {cabin} demand is expected to {trend} over the next {horizon} days. "
            f"The model has {mape_value:.1f}% historical accuracy. {action_text}"
        )
    )

    render_metric_grid(
        [
            {"label": "Forecast Horizon", "value": f"{horizon} days", "delta": f"Peak in {days_to_peak} days", "delta_state": "neutral"},
            {"label": "Historical MAPE", "value": f"{mape_value:.1f}%"},
            {"label": "Latest Forecast", "value": f"{int(forecast_df['ensemble_forecast'].iloc[-1]):,}", "delta": f"{trend_pct:+.1f}% trend", "delta_state": "positive" if trend_pct >= 0 else "negative"},
        ],
        columns=3,
    )

    rec_col, win_col = st.columns([1.0, 1.0])
    with rec_col:
        st.markdown("#### Campaign Launch Windows")
        windows = campaign_window_recommendations(forecast_df, top_n=5)
        if windows.empty:
            st.info("No strong rising-demand windows were detected for the selected horizon.")
        else:
            windows["campaign_start"] = pd.to_datetime(windows["campaign_start"]).dt.strftime("%d %b %Y")
            windows["demand_peak"] = pd.to_datetime(windows["demand_peak"]).dt.strftime("%d %b %Y")
            windows["demand_growth"] = windows["demand_growth"].fillna(0).apply(lambda value: f"{value:+.1f}%")
            windows["peak_forecast"] = windows["peak_forecast"].apply(lambda value: f"{int(round(value)):,} bookings/day")
            window_display = format_dataframe_display(
                windows,
                rename_map={
                    "campaign_start": "Campaign Start",
                    "demand_peak": "Demand Peak",
                    "peak_forecast": "Peak Forecast",
                    "demand_growth": "Demand Growth",
                },
            )
            st.dataframe(window_display, use_container_width=True, hide_index=True)

    with win_col:
        st.markdown(f"#### Campaign Launch Intelligence - {route} {cabin}")
        p25 = int(recommendation.get("p25_lead_days", recommendation.get("recommended_launch", 30) - 7))
        p50 = int(recommendation.get("p50_lead_days", p25 + 10))
        p75 = int(recommendation.get("p75_lead_days", p50 + 12))
        cols = st.columns(3)
        with cols[0]:
            st.metric("25% book", f"{p25} days")
            st.metric("50% book", f"{p50} days")
            st.metric("75% book", f"{p75} days")
        with cols[1]:
            st.metric("Campaign must launch", recommended_launch_date.strftime("%d %b %Y"))
            st.metric("Days remaining", f"{days_until_launch:,}")
            total_window = max(launch_lead, 1)
            elapsed = max(total_window - max(days_until_launch, 0), 0)
            st.progress(min(max(elapsed / total_window, 0.0), 1.0))
        with cols[2]:
            if days_until_launch < 0:
                status = "OVERDUE"
                status_color = "#E76F51"
            elif days_until_launch <= 7:
                status = "AT RISK"
                status_color = "#C4973B"
            else:
                status = "ON TRACK"
                status_color = "#00D4AA"
            st.markdown(
                f"""
                <div style="background:#0F2040; border-left:6px solid {status_color}; border-radius:10px; padding:16px;">
                    <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:#607080;">Urgency Level</div>
                    <div style="margin-top:10px; font-size:1.25rem; font-weight:800; color:{status_color};">{status}</div>
                    <div style="margin-top:8px; color:#B9C6DB;">{recommendation.get('insight', recommendation.get('note', ''))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    timing_tab, lead_tab, channel_tab = st.tabs(
        ["Campaign Calendar", "Lead-Time Profile", "Channel Recommendations"]
    )

    with timing_tab:
        cal_col1, cal_col2, cal_col3 = st.columns(3)
        departure_month = cal_col1.selectbox("Departure Month", list(range(1, 13)), index=6)
        departure_year = cal_col2.selectbox("Departure Year", [2025, 2026, 2027], index=1)
        budget_hkd = cal_col3.number_input("Budget HKD", min_value=100000, value=1200000, step=50000)
        calendar_df = generate_campaign_calendar(route, departure_month, departure_year, cabin, budget_hkd).copy()
        calendar_df["start_date"] = pd.to_datetime(calendar_df["start_date"]).dt.strftime("%d %b %Y")
        calendar_df["end_date"] = pd.to_datetime(calendar_df["end_date"]).dt.strftime("%d %b %Y")
        calendar_display = format_dataframe_display(
            calendar_df,
            currency_cols=["budget_hkd"],
            rename_map={
                "phase": "Phase",
                "start_date": "Start Date",
                "end_date": "End Date",
                "duration_days": "Duration",
                "budget_hkd": "Budget",
                "budget_pct": "Budget Share",
                "channels": "Channels",
                "kpi": "Primary KPI",
                "kpi_target": "Target",
                "days_to_depart": "Days to Departure",
            },
        )
        st.dataframe(calendar_display, use_container_width=True, hide_index=True)

        benchmarks = route_timing_benchmarks(demand)
        benchmark_route = benchmarks[benchmarks["route"] == route].sort_values("month")
        fig = px.line(
            benchmark_route,
            x="month",
            y="avg_bookings",
            markers=True,
            title=f"Monthly Demand Benchmark: {route}",
            color_discrete_sequence=[COLORS["primary"]],
        )
        apply_theme(fig)
        fig.update_layout(height=300, xaxis_title="Month", yaxis_title="Average Bookings")
        st.plotly_chart(fig, use_container_width=True)

        heatmap_source = (
            benchmarks.pivot_table(index="route", columns="month", values="avg_bookings", fill_value=0)
            .sort_index()
        )
        fig = px.imshow(
            heatmap_source,
            aspect="auto",
            color_continuous_scale=["#10213E", "#00A3A1", "#C4973B"],
            title="Route Demand Heatmap",
        )
        fig.update_layout(height=420, xaxis_title="Month", yaxis_title="Route")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        route_monthly = benchmarks.groupby("route")["avg_bookings"]
        consistent_routes = route_monthly.mean().sort_values(ascending=False).head(2).index.tolist()
        seasonal_route = route_monthly.var().sort_values(ascending=False).index[0]
        render_what_this_means(
            text=(
                f"Routes showing consistently dark cells across multiple months are your demand anchors - {', '.join(consistent_routes)}. "
                f"Routes with bright cells only in specific months are seasonal - {seasonal_route}. "
                "Align your campaign calendar to these demand patterns."
            ),
            action="Use Campaign Calendar tab to plan route-specific timing",
        )

    with lead_tab:
        route_booking_summary = booking_window_summary(bookings_df)
        lead_curve = conversion_by_lead_time(bookings_df)
        left, right = st.columns(2)
        summary_display = format_dataframe_display(
            route_booking_summary,
            currency_cols=["avg_fare"],
            pct_cols=["conversion_rate"],
            rename_map={
                "route": "Route",
                "avg_lead_time": "Avg Lead Time",
                "median_lead_time": "Median Lead Time",
                "p25_lead_time": "P25 Lead Time",
                "p75_lead_time": "P75 Lead Time",
                "total_bookings": "Bookings Analysed",
                "conversion_rate": "Conversion Rate",
                "avg_fare": "Avg Fare",
            },
        )
        left.dataframe(summary_display, use_container_width=True, hide_index=True)
        fig = px.bar(
            lead_curve,
            x="lead_time_band",
            y="conversion_rate",
            color="conversion_rate",
            color_continuous_scale=["#27496D", COLORS["success"]],
            title="Conversion Rate by Booking Lead Time",
        )
        fig.update_layout(
            coloraxis_showscale=False,
            height=320,
            xaxis_title="Lead Time Band",
            yaxis_title="Conversion Rate (%)",
        )
        apply_theme(fig)
        right.plotly_chart(fig, use_container_width=True)

    with channel_tab:
        phase = st.selectbox("Campaign Phase", ["Awareness", "Consideration", "Conversion"], index=2)
        channel_mix = channel_timing_recommendations(route, phase, float(budget_hkd))
        channel_display = format_dataframe_display(
            channel_mix,
            currency_cols=["spend_hkd"],
            rename_map={
                "channel": "Channel",
                "spend_hkd": "Spend",
                "spend_pct": "Spend Share",
                "est_impressions": "Estimated Impressions",
                "est_bookings": "Estimated Bookings",
                "rationale": "Rationale",
            },
        )
        st.dataframe(channel_display, use_container_width=True, hide_index=True)
