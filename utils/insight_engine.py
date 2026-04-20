"""
Rule-based decision intelligence engine for the platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from modules.demand_intelligence.route_forecaster import ensemble_forecast


THEME = {
    "critical": "#E76F51",
    "opportunity": "#C4973B",
    "on_track": "#00D4AA",
}


def _format_hkd(value: float, decimals: int = 1) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"HK${value / 1_000_000:.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"HK${value:,.0f}"
    return f"HK${value:.0f}"


def _format_pct(value: float, decimals: int = 1, assume_ratio: bool = True) -> str:
    if pd.isna(value):
        return "0.0%"
    pct = float(value) * 100 if assume_ratio and abs(float(value)) <= 1.5 else float(value)
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.{decimals}f}%"


def _today() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def _route_customer_preferences(df_customers: pd.DataFrame, df_digital: pd.DataFrame) -> pd.DataFrame:
    route_pref = (
        df_digital.groupby(["customer_id", "route"])
        .size()
        .reset_index(name="touches")
        .sort_values(["customer_id", "touches"], ascending=[True, False])
        .drop_duplicates("customer_id")
        .rename(columns={"route": "preferred_route"})
    )
    customer_routes = df_customers.merge(route_pref[["customer_id", "preferred_route"]], on="customer_id", how="left")
    fallback_route = (
        df_digital["route"].mode().iloc[0]
        if not df_digital.empty and "route" in df_digital
        else "HKG-LHR"
    )
    customer_routes["preferred_route"] = customer_routes["preferred_route"].fillna(fallback_route)
    return customer_routes


def _derive_customer_fields(df_customers: pd.DataFrame, df_digital: pd.DataFrame) -> pd.DataFrame:
    customers = _route_customer_preferences(df_customers, df_digital).copy()
    customers["revenue_at_risk_hkd"] = (
        customers.get("revenue_at_risk_hkd", customers["customer_lifetime_value_12m"] * customers["churn_probability_90d"])
    )
    customers["avg_miles_balance"] = (
        customers.get(
            "avg_miles_balance",
            ((customers["total_revenue_hkd"] / 12) + (customers["trips_last_12m"] * 850)).round(0),
        )
    ).astype(float)
    customers["miles_expiry_risk"] = customers.get(
        "miles_expiry_risk",
        (
            (customers["days_since_last_booking"] >= 210)
            & (customers["avg_miles_balance"] >= customers["avg_miles_balance"].median())
        ).astype(int),
    )
    return customers


def _channel_roas(df_campaigns: pd.DataFrame) -> pd.DataFrame:
    roas = (
        df_campaigns.groupby("channel")
        .agg(spend_hkd=("spend_hkd", "sum"), revenue_hkd=("revenue_hkd", "sum"))
        .reset_index()
    )
    roas["roas"] = roas["revenue_hkd"] / roas["spend_hkd"].clip(lower=1)
    roas["spend_share"] = roas["spend_hkd"] / max(roas["spend_hkd"].sum(), 1)
    return roas.sort_values("roas", ascending=False).reset_index(drop=True)


def _route_demand_forecasts(df_demand: pd.DataFrame, periods: int = 45) -> pd.DataFrame:
    demand = df_demand.copy()
    demand["date"] = pd.to_datetime(demand["date"])
    route_stats = []
    for route in sorted(demand["route"].unique()):
        try:
            forecast_df, _, mape_value = ensemble_forecast(demand, route, periods=periods)
        except Exception:
            continue
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])
        peak_row = forecast_df.loc[forecast_df["ensemble_forecast"].idxmax()]
        route_stats.append(
            {
                "route": route,
                "peak_date": peak_row["date"],
                "days_to_peak": int((peak_row["date"] - demand["date"].max()).days),
                "peak_forecast": float(peak_row["ensemble_forecast"]),
                "avg_forecast": float(forecast_df["ensemble_forecast"].mean()),
                "mape": float(mape_value),
            }
        )
    return pd.DataFrame(route_stats)


def _conversion_path_shares(df_digital: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    digital = df_digital.copy()
    digital["timestamp"] = pd.to_datetime(digital["timestamp"])
    digital = digital.sort_values(["customer_id", "timestamp"])
    first_touch = []
    last_touch = []
    actual_touches = []

    for _, group in digital.groupby("customer_id", sort=False):
        converted = group[group["converted"] == 1]
        if converted.empty:
            continue
        conv_time = converted.iloc[0]["timestamp"]
        journey = group[group["timestamp"] <= conv_time].tail(4)
        if journey.empty:
            continue
        first_touch.append(journey.iloc[0]["channel"])
        last_touch.append(journey.iloc[-1]["channel"])
        actual_touches.extend(journey["channel"].tolist())

    first_df = pd.Series(first_touch).value_counts(normalize=True).rename_axis("channel").reset_index(name="first_touch_share")
    last_df = pd.Series(last_touch).value_counts(normalize=True).rename_axis("channel").reset_index(name="last_touch_share")
    actual_df = pd.Series(actual_touches).value_counts(normalize=True).rename_axis("channel").reset_index(name="journey_share")
    return first_df, last_df, actual_df


def _severity_insight(
    severity: str,
    module: str,
    icon: str,
    title: str,
    body: str,
    action: str,
    urgency_days: int,
    metric: str,
    nav_target: str,
) -> dict:
    return {
        "severity": severity,
        "module": module,
        "icon": icon,
        "title": title if len(title.split()) <= 10 else " ".join(title.split()[:10]),
        "body": body,
        "action": action,
        "urgency_days": int(urgency_days),
        "metric": metric,
        "nav_target": nav_target,
    }


def _flatten_insights(groups: dict) -> list[dict]:
    items = []
    for key in ["critical", "opportunity", "on_track", "cross_module"]:
        items.extend(groups.get(key, []))
    return items


def generate_all_insights(
    df_customers: pd.DataFrame,
    df_campaigns: pd.DataFrame,
    df_demand: pd.DataFrame,
    df_digital: pd.DataFrame,
) -> dict:
    """Generate decision intelligence using real computed values."""
    critical: list[dict] = []
    opportunity: list[dict] = []
    on_track: list[dict] = []
    cross_module: list[dict] = []

    customers = _derive_customer_fields(df_customers, df_digital)
    campaigns = df_campaigns.copy()
    demand = df_demand.copy()
    digital = df_digital.copy()
    demand["date"] = pd.to_datetime(demand["date"])
    campaigns["start_date"] = pd.to_datetime(campaigns["start_date"])

    channel_roas = _channel_roas(campaigns)
    route_forecasts = _route_demand_forecasts(demand)

    # Check 1 - High churn risk volume
    high_churn = customers[customers["churn_probability_90d"] > 0.70].copy()
    high_churn_count = len(high_churn)
    revenue_at_risk = high_churn["revenue_at_risk_hkd"].sum()
    top_tier = (
        high_churn["loyalty_tier"].value_counts().idxmax()
        if not high_churn.empty
        else customers["loyalty_tier"].value_counts().idxmax()
    )
    if high_churn_count > 500:
        severity = "critical"
        urgency = 14
    elif high_churn_count >= 100:
        severity = "opportunity"
        urgency = 30
    else:
        severity = "on_track"
        urgency = 30

    churn_insight = _severity_insight(
        severity=severity,
        module="Customer Intelligence",
        icon="RISK" if severity != "on_track" else "VALUE",
        title="Churn Risk Concentration",
        body=(
            f"{high_churn_count:,} members sit above 70% churn risk, placing {_format_hkd(revenue_at_risk)} of annual value in play. "
            f"{top_tier} members are the most affected tier, so the retention response should start there."
        ),
        action=f"Launch retention campaign for {top_tier} members within {urgency} days",
        urgency_days=urgency,
        metric=_format_hkd(revenue_at_risk),
        nav_target="Customer Intelligence",
    )
    {"critical": critical, "opportunity": opportunity, "on_track": on_track}[severity].append(churn_insight)

    # Check 2 - Miles expiry risk
    miles_risk = customers[customers["miles_expiry_risk"] == 1]
    miles_metric = float(len(miles_risk) * miles_risk["avg_miles_balance"].mean()) if not miles_risk.empty else 0
    miles_insight = _severity_insight(
        severity="opportunity" if len(miles_risk) > 0 else "on_track",
        module="Customer Intelligence",
        icon="VALUE" if len(miles_risk) > 0 else "VALUE",
        title="Miles Expiry Exposure",
        body=(
            f"{len(miles_risk):,} members are nearing miles expiry, representing roughly {miles_metric:,.0f} miles of latent value. "
            "If they disengage now, you risk losing these members permanently to competitors."
        ),
        action="Send miles expiry alert campaign",
        urgency_days=7,
        metric=f"{miles_metric:,.0f} miles",
        nav_target="Customer Intelligence",
    )
    (opportunity if len(miles_risk) > 0 else on_track).append(miles_insight)

    # Check 3 - Segment health
    segment_health = (
        customers.groupby("customer_segment")
        .agg(avg_churn=("churn_probability_90d", "mean"), avg_clv=("customer_lifetime_value_12m", "mean"))
        .reset_index()
    )
    worst_churn_segment = segment_health.sort_values("avg_churn", ascending=False).iloc[0]
    best_segment = segment_health.sort_values("avg_clv", ascending=False).iloc[0]
    lowest_clv_segment = segment_health.sort_values("avg_clv", ascending=True).iloc[0]
    segment_insight = _severity_insight(
        severity="opportunity",
        module="Customer Intelligence",
        icon="SEG",
        title="Segment Health Divide",
        body=(
            f"{worst_churn_segment['customer_segment']} shows the highest churn risk at {_format_pct(worst_churn_segment['avg_churn'])}, "
            f"while {lowest_clv_segment['customer_segment']} carries the lowest average CLV at {_format_hkd(lowest_clv_segment['avg_clv'])}. "
            f"By comparison, {best_segment['customer_segment']} averages {_format_hkd(best_segment['avg_clv'])} per member."
        ),
        action=f"Prioritise recovery playbooks for {worst_churn_segment['customer_segment']}",
        urgency_days=21,
        metric=_format_pct(worst_churn_segment["avg_churn"]),
        nav_target="Customer Intelligence",
    )
    opportunity.append(segment_insight)

    # Check 4 - Channel ROAS gap
    best_channel = channel_roas.iloc[0]
    worst_channel = channel_roas.sort_values("roas").iloc[0]
    roas_gap = best_channel["roas"] - worst_channel["roas"]
    potential_revenue = worst_channel["spend_hkd"] * max(best_channel["roas"] - worst_channel["roas"], 0)
    if roas_gap > 50:
        opportunity.append(
            _severity_insight(
                severity="opportunity",
                module="Marketing Mix",
                icon="ROAS",
                title="ROAS Reallocation Gap",
                body=(
                    f"{best_channel['channel']} returns {best_channel['roas']:.1f}x ROAS, while {worst_channel['channel']} returns only {worst_channel['roas']:.1f}x. "
                    f"Reallocating the weakest channel's spend would unlock about {_format_hkd(potential_revenue)} in additional revenue."
                ),
                action=f"Run Budget Optimizer - shift spend from {worst_channel['channel']} to {best_channel['channel']}",
                urgency_days=30,
                metric=_format_hkd(potential_revenue),
                nav_target="Marketing Mix",
            )
        )

    # Check 5 - Email CRM underinvestment
    portfolio_roas = campaigns["revenue_hkd"].sum() / max(campaigns["spend_hkd"].sum(), 1)
    email_row = channel_roas[channel_roas["channel"] == "Email CRM"]
    email_underinvested = False
    if not email_row.empty:
        email_spend_share = float(email_row["spend_share"].iloc[0])
        email_roas = float(email_row["roas"].iloc[0])
        if email_spend_share < 0.10 and email_roas > portfolio_roas:
            email_underinvested = True
            opportunity.append(
                _severity_insight(
                    severity="opportunity",
                    module="Marketing Mix",
                    icon="EMAIL",
                    title="Email Budget Gap",
                    body=(
                        f"Email CRM delivers {email_roas:.1f}x ROAS but receives only {_format_pct(email_spend_share)} of budget. "
                        f"That puts one of your most efficient channels below the investment it can absorb."
                    ),
                    action="Increase Email CRM budget by minimum 2x in next planning cycle",
                    urgency_days=45,
                    metric=f"{email_roas:.1f}x ROAS",
                    nav_target="Marketing Mix",
                )
            )

    # Check 6 - Single channel over concentration
    concentration_row = channel_roas.sort_values("spend_share", ascending=False).iloc[0]
    if concentration_row["spend_share"] > 0.40:
        opportunity.append(
            _severity_insight(
                severity="opportunity",
                module="Marketing Mix",
                icon="MIX",
                title="Spend Concentration Risk",
                body=(
                    f"{concentration_row['channel']} absorbs {_format_pct(concentration_row['spend_share'])} of portfolio spend. "
                    f"Its ROAS is {concentration_row['roas']:.1f}x versus a portfolio average of {portfolio_roas:.1f}x, which leaves you over-exposed."
                ),
                action=f"Diversify - reduce {concentration_row['channel']} to max 35% of portfolio",
                urgency_days=30,
                metric=_format_pct(concentration_row["spend_share"]),
                nav_target="Marketing Mix",
            )
        )

    # Check 7 - High load factor routes
    last_30 = demand[demand["date"] >= demand["date"].max() - pd.Timedelta(days=29)]
    high_lf = (
        last_30.groupby("route")["load_factor"]
        .mean()
        .reset_index()
        .sort_values("load_factor", ascending=False)
    )
    high_lf = high_lf[high_lf["load_factor"] > 0.85]
    if high_lf.empty:
        fallback_route = (
            last_30.groupby("route")["load_factor"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        ).iloc[0]
        on_track.append(
            _severity_insight(
                severity="on_track",
                module="Demand Intelligence",
                icon="ROUTE",
                title="Load Factor Strength",
                body=(
                    f"{fallback_route['route']} is your strongest recent route at {_format_pct(fallback_route['load_factor'])} load factor over the latest 30 days. "
                    "Demand is holding firm enough to support disciplined premium pricing."
                ),
                action=f"Consider premium fare strategy on {fallback_route['route']} - demand is strong",
                urgency_days=7,
                metric=_format_pct(fallback_route["load_factor"]),
                nav_target="Demand Intelligence",
            )
        )
    else:
        top_route = high_lf.iloc[0]
        on_track.append(
            _severity_insight(
                severity="on_track",
                module="Demand Intelligence",
                icon="ROUTE",
                title="Premium Pricing Window",
                body=(
                    f"{top_route['route']} is running at {_format_pct(top_route['load_factor'])} load factor in the last 30 days. "
                    "That level of utilisation suggests there is room to test firmer pricing before demand softens."
                ),
                action=f"Consider premium fare strategy on {top_route['route']} - demand is strong",
                urgency_days=7,
                metric=_format_pct(top_route["load_factor"]),
                nav_target="Demand Intelligence",
            )
        )

    # Check 8 - Campaign timing urgency
    if not route_forecasts.empty:
        urgent_routes = route_forecasts[route_forecasts["days_to_peak"] <= 30].sort_values("days_to_peak")
        if not urgent_routes.empty:
            urgent_row = urgent_routes.iloc[0]
            critical.append(
                _severity_insight(
                    severity="critical",
                    module="Demand Intelligence",
                    icon="TIMING",
                    title="Campaign Window Closing",
                    body=(
                        f"Peak demand for {urgent_row['route']} arrives in {int(urgent_row['days_to_peak'])} days. "
                        "Campaign activity needs to go live now or the highest-intent demand will pass without support."
                    ),
                    action=f"Launch {urgent_row['route']} campaign immediately - window closing",
                    urgency_days=0,
                    metric=f"{int(urgent_row['days_to_peak'])} days",
                    nav_target="Demand Intelligence",
                )
            )

    # Check 9 - Undermarketed high-demand routes
    route_demand = demand.groupby("route")["bookings"].mean().reset_index(name="avg_bookings")
    route_spend = campaigns.groupby("route")["spend_hkd"].sum().reset_index(name="mkt_spend_hkd")
    route_mix = route_demand.merge(route_spend, on="route", how="left").fillna(0)
    high_demand_low_spend = route_mix[
        (route_mix["avg_bookings"] > route_mix["avg_bookings"].mean())
        & (route_mix["mkt_spend_hkd"] < route_mix["mkt_spend_hkd"].mean())
    ].sort_values("avg_bookings", ascending=False)
    if not high_demand_low_spend.empty:
        routes = ", ".join(high_demand_low_spend["route"].head(3))
        opportunity.append(
            _severity_insight(
                severity="opportunity",
                module="Demand Intelligence",
                icon="DEMAND",
                title="Underfunded Demand Routes",
                body=(
                    f"{routes} are generating above-average demand while attracting below-average marketing spend. "
                    "That mismatch suggests you have profitable routes being left under-supported."
                ),
                action=f"Increase marketing investment on {routes}",
                urgency_days=21,
                metric=f"{len(high_demand_low_spend)} routes",
                nav_target="Demand Intelligence",
            )
        )

    # Check 10 - Attribution bias detection
    first_df, last_df, actual_df = _conversion_path_shares(digital)
    attribution_compare = last_df.merge(first_df, on="channel", how="outer").merge(actual_df, on="channel", how="outer").fillna(0)
    attribution_compare["over_credit_pct"] = (
        attribution_compare["last_touch_share"] - attribution_compare["journey_share"]
    ) * 100
    if not attribution_compare.empty:
        overcredited = attribution_compare.sort_values("over_credit_pct", ascending=False).iloc[0]
        if overcredited["over_credit_pct"] > 0:
            opportunity.append(
                _severity_insight(
                    severity="opportunity",
                    module="Digital Attribution",
                    icon="ATTR",
                    title="Last-Click Bias",
                    body=(
                        f"Last-click is over-crediting {overcredited['channel']} by approximately {overcredited['over_credit_pct']:.1f}% against actual journey share. "
                        "That can push budget decisions too far toward closing channels and away from journey builders."
                    ),
                    action="Adopt Markov Chain attribution - rebalance channel investment",
                    urgency_days=60,
                    metric=_format_pct(overcredited["over_credit_pct"], assume_ratio=False),
                    nav_target="Digital Attribution",
                )
            )

    # Cross-module Signal 1
    if not high_churn.empty and not route_forecasts.empty:
        top_route = high_churn["preferred_route"].value_counts().idxmax()
        avg_forecast = route_forecasts["avg_forecast"].mean()
        route_row = route_forecasts[route_forecasts["route"] == top_route]
        top_segment = high_churn["customer_segment"].value_counts().idxmax()
        if not route_row.empty and float(route_row["avg_forecast"].iloc[0]) > avg_forecast:
            cross_module.append(
                _severity_insight(
                    severity="opportunity",
                    module="Customer Intelligence + Demand Intelligence",
                    icon="LINK",
                    title="Churn Meets Demand",
                    body=(
                        f"High-churn customers show the strongest route preference for {top_route}, and that route is still forecasting above-average demand. "
                        f"That gives you a rare chance to recover at-risk customers with an offer tied to a route they already want."
                    ),
                    action=f"Create personalised {top_route} win-back campaign targeting {top_segment}",
                    urgency_days=14,
                    metric=_format_hkd(high_churn.loc[high_churn['preferred_route'] == top_route, 'revenue_at_risk_hkd'].sum()),
                    nav_target="Customer Intelligence",
                )
            )

    # Cross-module Signal 2
    top_clv_segment = segment_health.sort_values("avg_clv", ascending=False).iloc[0]
    if email_underinvested and top_clv_segment["avg_clv"] > 0:
        cross_module.append(
            _severity_insight(
                severity="opportunity",
                module="Marketing Mix + Customer Intelligence",
                icon="EMAIL",
                title="High-Value Reach Gap",
                body=(
                    f"Your highest-value segment is {top_clv_segment['customer_segment']} at {_format_hkd(top_clv_segment['avg_clv'])} average CLV, "
                    "yet Email CRM remains under-funded despite outperforming the portfolio. "
                    "Your most valuable audience is under-reached by one of your most efficient channels."
                ),
                action=f"Priority: Email CRM campaign for {top_clv_segment['customer_segment']} - dual quick win",
                urgency_days=21,
                metric=f"{email_row['roas'].iloc[0]:.1f}x ROAS" if not email_row.empty else "N/A",
                nav_target="Marketing Mix",
            )
        )

    # Cross-module Signal 3
    today = _today()
    peak_calendar = [
        ("Chinese New Year", pd.Timestamp(year=today.year + (1 if today.month > 2 else 0), month=2, day=1)),
        ("Summer Peak", pd.Timestamp(year=today.year, month=6, day=1)),
        ("Summer Peak", pd.Timestamp(year=today.year, month=7, day=1)),
        ("Summer Peak", pd.Timestamp(year=today.year, month=8, day=1)),
        ("Christmas", pd.Timestamp(year=today.year, month=12, day=1)),
    ]
    future_peaks = [(name, date) for name, date in peak_calendar if date >= today]
    if future_peaks:
        next_peak_name, next_peak_date = min(future_peaks, key=lambda x: x[1])
        days_to_peak = (next_peak_date - today).days
        recent_campaign_cutoff = today - pd.Timedelta(days=60)
        recent_activity = (campaigns["start_date"] >= recent_campaign_cutoff).any()
        if days_to_peak <= 60 and not recent_activity:
            cross_module.append(
                _severity_insight(
                    severity="opportunity",
                    module="Demand Intelligence + Campaign Studio",
                    icon="SEASON",
                    title="Seasonal Readiness",
                    body=(
                        f"{next_peak_name} is {days_to_peak} days away and there is no recent campaign activity in the planning window. "
                        "Creative, targeting, and media allocation should already be underway for peak routes."
                    ),
                    action="Generate campaign briefs for peak period routes immediately",
                    urgency_days=max(days_to_peak, 1),
                    metric=f"{days_to_peak} days",
                    nav_target="Campaign Studio",
                )
            )

    # Health score
    score = 100
    if high_churn_count > 500:
        score -= 15
    if not route_forecasts.empty and (route_forecasts["days_to_peak"] <= 30).any():
        score -= 10
    if concentration_row["spend_share"] > 0.40:
        score -= 8
    score -= min(len(opportunity) * 5, 20)
    score += min(len(on_track) * 5, 10)
    score = int(np.clip(score, 0, 100))

    if score >= 80:
        label = "Strong"
    elif score >= 60:
        label = "Good"
    elif score >= 40:
        label = "Needs Attention"
    else:
        label = "At Risk"

    insights = {
        "critical": critical,
        "opportunity": opportunity,
        "on_track": on_track,
        "cross_module": cross_module,
        "health_score": score,
    }

    ranked = sorted(_flatten_insights(insights), key=lambda item: item["urgency_days"])
    insights["top_3_actions"] = [item["action"] for item in ranked[:3]]
    return insights
