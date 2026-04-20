import html
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

from config.settings import APP_TITLE, APP_VERSION
from modules.ai_governance.ui import render as render_ai_governance_module
from modules.campaign_studio.ui import render as render_campaign_studio_module
from modules.customer_intelligence.ui import render as render_customer_intelligence_module
from modules.demand_intelligence.route_forecaster import ensemble_forecast
from modules.demand_intelligence.ui import render as render_demand_intelligence_module
from modules.digital_attribution.ui import render as render_digital_attribution_module
from modules.marketing_mix.ui import render as render_marketing_mix_module
from utils.chart_helpers import (
    format_currency,
    format_percentage,
    labeled_divider,
    load_theme_css,
    render_metric_grid,
    render_page_header,
)
from utils.insight_engine import generate_all_insights
from utils.insight_renderer import (
    render_cross_module_signal,
    render_health_score_widget,
    render_insight_banner,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"

DATASETS = {
    "Customers": DATA_DIR / "customers.csv",
    "Campaigns": DATA_DIR / "campaigns.csv",
    "Route Demand": DATA_DIR / "route_demand.csv",
    "Digital Touchpoints": DATA_DIR / "digital_touchpoints.csv",
}

PAGES = [
    ("Overview", "OV"),
    ("Customer Intelligence", "CI"),
    ("Demand Intelligence", "DI"),
    ("Marketing Mix", "MM"),
    ("Digital Attribution", "DA"),
    ("Campaign Studio", "CS"),
    ("AI Governance", "AG"),
]


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_all_data() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for label, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")
        frames[label] = pd.read_csv(path)
    return frames


@st.cache_data(show_spinner=False)
def get_all_insights(
    df_customers: pd.DataFrame,
    df_campaigns: pd.DataFrame,
    df_demand: pd.DataFrame,
    df_digital: pd.DataFrame,
) -> dict:
    return generate_all_insights(df_customers, df_campaigns, df_demand, df_digital)


@st.cache_data(show_spinner=False)
def get_overview_snapshot(
    df_customers: pd.DataFrame,
    df_campaigns: pd.DataFrame,
    df_demand: pd.DataFrame,
    df_digital: pd.DataFrame,
) -> dict:
    customers = df_customers.copy()
    campaigns = df_campaigns.copy()
    demand = df_demand.copy()
    digital = df_digital.copy()
    demand["date"] = pd.to_datetime(demand["date"])
    campaigns["start_date"] = pd.to_datetime(campaigns["start_date"])

    high_churn = customers[customers["churn_probability_90d"] > 0.70].copy()
    high_churn["revenue_at_risk_hkd"] = (
        high_churn["customer_lifetime_value_12m"] * high_churn["churn_probability_90d"]
    )
    revenue_at_risk = float(high_churn["revenue_at_risk_hkd"].sum())

    channel_roas = (
        campaigns.groupby("channel")
        .agg(spend_hkd=("spend_hkd", "sum"), revenue_hkd=("revenue_hkd", "sum"))
        .reset_index()
    )
    channel_roas["roas"] = channel_roas["revenue_hkd"] / channel_roas["spend_hkd"].clip(lower=1)
    best_channel = channel_roas.sort_values("roas", ascending=False).iloc[0]
    worst_channel = channel_roas.sort_values("roas", ascending=True).iloc[0]
    portfolio_roas = float(campaigns["revenue_hkd"].sum() / max(campaigns["spend_hkd"].sum(), 1))

    route_totals = demand.groupby("route", as_index=False)["bookings"].sum().sort_values("bookings", ascending=False)
    top_route = route_totals.iloc[0]
    _, _, mape_value = ensemble_forecast(demand, top_route["route"], periods=30)

    last_30 = demand[demand["date"] >= demand["date"].max() - pd.Timedelta(days=29)]
    high_lf_routes = (
        last_30.groupby("route")["load_factor"]
        .mean()
        .loc[lambda values: values > 0.85]
        .sort_values(ascending=False)
    )
    best_market = (
        campaigns.groupby("market")
        .apply(lambda frame: frame["revenue_hkd"].sum() / max(frame["spend_hkd"].sum(), 1))
        .sort_values(ascending=False)
    )
    converted_pct = float(digital["converted"].sum() / max(len(digital), 1) * 100)

    return {
        "revenue_at_risk_hkd": revenue_at_risk,
        "portfolio_roas": portfolio_roas,
        "top_route": str(top_route["route"]),
        "top_route_bookings": float(top_route["bookings"]),
        "best_channel": str(best_channel["channel"]),
        "best_channel_roas": float(best_channel["roas"]),
        "worst_channel": str(worst_channel["channel"]),
        "forecast_mape": float(mape_value),
        "high_churn_count": int(len(high_churn)),
        "high_lf_routes": high_lf_routes.index.tolist(),
        "best_market": str(best_market.index[0]),
        "converted_pct": converted_pct,
    }


def _flatten_insights(insights: dict) -> list[dict]:
    items: list[dict] = []
    for key in ["critical", "opportunity", "on_track", "cross_module"]:
        items.extend(insights.get(key, []))
    return items


def _health_label(score: int) -> str:
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Needs Attention"
    return "At Risk"


def _page_href(page_name: str) -> str:
    return f"?page={quote(page_name)}"


def _compact_html(value: str) -> str:
    return re.sub(r">\s+<", "><", value.strip())


def _sync_page_state() -> str:
    valid_pages = {name for name, _ in PAGES}
    query_page = st.query_params.get("page", "")
    if isinstance(query_page, list):
        query_page = query_page[0] if query_page else ""
    current_page = query_page or st.session_state.get("page") or st.session_state.get("nav_page") or "Overview"
    if current_page not in valid_pages:
        current_page = "Overview"
    st.session_state["page"] = current_page
    st.session_state["nav_page"] = current_page
    st.query_params["page"] = current_page
    return current_page


def render_sidebar(loaded: dict[str, pd.DataFrame], current_page: str):
    nav_items = "".join(
        _compact_html(
            f"""
            <a class="sidebar-nav-item {'sidebar-nav-item--active' if page_name == current_page else ''}" href="{_page_href(page_name)}" target="_self">
              <span class="sidebar-nav-item__icon">{html.escape(icon)}</span>
              <span>{html.escape(page_name if len(page_name) <= 18 else page_name.replace('Intelligence', 'Intel.'))}</span>
            </a>
            """
        )
        for page_name, icon in PAGES
    )
    sidebar_html = _compact_html(f"""
    <div class="sidebar-shell">
      <div class="sidebar-brand">
        <div class="sidebar-brand__icon">A</div>
        <div>
          <div class="sidebar-brand__title">Aviation MI</div>
          <div class="sidebar-brand__subtext">Intelligence Platform</div>
        </div>
      </div>
      <div class="sidebar-section sidebar-section--first">
        <div class="sidebar-section__label">Modules</div>
      </div>
      <div class="sidebar-nav">
        {nav_items}
      </div>
      <div class="sidebar-section">
        <div class="sidebar-section__label">Data</div>
        <div class="sidebar-data">
          <div class="sidebar-data__row"><span class="sidebar-data__label">Members</span><span class="sidebar-data__value">{len(loaded['Customers']):,}</span></div>
          <div class="sidebar-data__row"><span class="sidebar-data__label">Campaigns</span><span class="sidebar-data__value">{len(loaded['Campaigns']):,}</span></div>
          <div class="sidebar-data__row"><span class="sidebar-data__label">Route-days</span><span class="sidebar-data__value">{len(loaded['Route Demand']):,}</span></div>
          <div class="sidebar-data__row"><span class="sidebar-data__label">Touchpoints</span><span class="sidebar-data__value">{len(loaded['Digital Touchpoints']):,}</span></div>
        </div>
      </div>
      <div class="sidebar-footer">
        <span>v{html.escape(APP_VERSION)}</span>
        <span class="sidebar-status-dot"></span>
        <span>Live</span>
      </div>
    </div>
    """)
    st.sidebar.markdown(sidebar_html, unsafe_allow_html=True)


def _render_action_list(sorted_insights: list[dict]):
    severity_palette = {
        "critical": ("CRITICAL", "#E05A40"),
        "opportunity": ("OPPORTUNITY", "#F0A050"),
        "on_track": ("ON TRACK", "#00A3A1"),
    }
    items_html = []
    number_colors = ["#E05A40", "#F0A050", "#00A3A1"]
    for index, insight in enumerate(sorted_insights[:3], start=1):
        severity_label, severity_color = severity_palette.get(insight["severity"], ("OPPORTUNITY", "#F0A050"))
        if insight["urgency_days"] <= 0:
            urgency_label = "Today"
        elif insight["urgency_days"] <= 14:
            urgency_label = "14 Days"
        else:
            urgency_label = "30 Days"
        items_html.append(
            _compact_html(f"""
            <div class="action-item">
              <span class="action-item__number" style="background:{number_colors[index - 1]};">{index}</span>
              <span class="action-item__severity" style="background:{severity_color}1F;border:1px solid {severity_color}4D;color:{severity_color};">{severity_label}</span>
              <span class="action-item__urgency">{urgency_label}</span>
              <span class="action-item__text">{html.escape(str(insight['action']))}</span>
            </div>
            """)
        )
    st.markdown(
        _compact_html(f'<div class="action-card">{"".join(items_html)}</div>'),
        unsafe_allow_html=True,
    )


def _module_card(title: str, icon_label: str, description: str, insight_line: str, color: str, page_key: str):
    st.markdown(
        _compact_html(f"""
        <a class="module-card-link" href="{_page_href(page_key)}" target="_self" style="text-decoration:none;">
          <div class="module-card">
            <div class="module-card__icon">{html.escape(icon_label)}</div>
            <div class="module-card__title">{html.escape(title)}</div>
            <div class="module-card__description">{html.escape(description)}</div>
            <div class="module-card__insight" style="color:{color};">{html.escape(insight_line)}</div>
          </div>
        </a>
        """),
        unsafe_allow_html=True,
    )


def render_overview(loaded: dict[str, pd.DataFrame], insights: dict):
    demand = loaded["Route Demand"]
    campaigns = loaded["Campaigns"]
    customers = loaded["Customers"]
    digital = loaded["Digital Touchpoints"]
    snapshot = get_overview_snapshot(customers, campaigns, demand, digital)
    all_ranked = sorted(_flatten_insights(insights), key=lambda item: item["urgency_days"])
    today_label = pd.Timestamp.today().strftime("%d %b %Y")
    health_score = int(insights.get("health_score", 0))

    render_page_header(
        title="Marketing Intelligence Briefing",
        subtitle="Platform-wide decision view across customers, demand, channel performance, attribution, campaign planning, and governance.",
        pills=[],
        meta=f"Generated {today_label} | {len(customers):,} members | {len(campaigns):,} campaigns analysed",
    )

    labeled_divider("Portfolio Health")
    render_health_score_widget(health_score, _health_label(health_score))

    labeled_divider("3 Things To Act On Today")
    _render_action_list(all_ranked)

    if insights.get("critical"):
        labeled_divider("Requires Immediate Action")
        for item in insights.get("critical", []):
            render_insight_banner(item)

    if insights.get("opportunity"):
        labeled_divider("Opportunities to Capture")
        opp_cols = st.columns(2)
        for idx, item in enumerate(insights.get("opportunity", [])):
            with opp_cols[idx % 2]:
                render_insight_banner(item)

    if insights.get("cross_module"):
        labeled_divider("Connected Insights")
        for signal in insights.get("cross_module", []):
            render_cross_module_signal(signal)

    labeled_divider("Performance Snapshot")
    render_metric_grid(
        [
            {"label": "Total Members", "value": f"{len(customers):,}"},
            {"label": "Revenue at Risk", "value": format_currency(snapshot["revenue_at_risk_hkd"])},
            {"label": "Portfolio ROAS", "value": f"{snapshot['portfolio_roas']:.1f}x"},
        ],
        columns=3,
    )
    render_metric_grid(
        [
            {"label": "Top Route", "value": snapshot["top_route"], "delta": f"{snapshot['top_route_bookings']:,.0f} bookings", "delta_state": "neutral"},
            {"label": "Best Channel", "value": snapshot["best_channel"], "delta": f"{snapshot['best_channel_roas']:.1f}x ROAS", "delta_state": "positive"},
            {"label": "Forecast Accuracy", "value": format_percentage(snapshot["forecast_mape"])},
        ],
        columns=3,
    )

    labeled_divider("Module Navigation")
    card_cols_top = st.columns(3)
    card_cols_bottom = st.columns(3)
    with card_cols_top[0]:
        churn_line_color = "#E05A40" if snapshot["high_churn_count"] > 500 else ("#F0A050" if snapshot["high_churn_count"] > 100 else "#00C896")
        _module_card(
            "Customer Intelligence",
            "CI",
            "Understand which members create value, which are leaving, and where retention should start.",
            f"{snapshot['high_churn_count']:,} members at high churn risk",
            churn_line_color,
            "Customer Intelligence",
        )
    with card_cols_top[1]:
        _module_card(
            "Demand Intelligence",
            "DI",
            "Forecast route demand, identify campaign windows, and time market activation before peaks.",
            (
                f"{len(snapshot['high_lf_routes'])} routes at >85% load factor"
                if snapshot["high_lf_routes"]
                else "No routes currently above 85% load factor"
            ),
            "#00A3A1",
            "Demand Intelligence",
        )
    with card_cols_top[2]:
        _module_card(
            "Marketing Mix",
            "MM",
            "Compare channel efficiency, shift budget intelligently, and quantify reallocation impact.",
            f"{snapshot['worst_channel']} underperforming - reallocation opportunity",
            "#C4973B",
            "Marketing Mix",
        )
    with card_cols_bottom[0]:
        _module_card(
            "Digital Attribution",
            "DA",
            "Understand journeys, attribution bias, and the experimental evidence behind channel credit.",
            f"{snapshot['converted_pct']:.1f}% journey conversion rate analysed",
            "#00A3A1",
            "Digital Attribution",
        )
    with card_cols_bottom[1]:
        _module_card(
            "Campaign Studio",
            "CS",
            "Generate channel-ready creative and route-specific messaging for the next campaign cycle.",
            f"{snapshot['best_market']} is your highest performing market",
            "#C4973B",
            "Campaign Studio",
        )
    with card_cols_bottom[2]:
        _module_card(
            "AI Governance",
            "AG",
            "Track model health, audit actions, and keep the platform decision-safe over time.",
            "6 models monitored | Platform health tracked",
            "#00C896",
            "AI Governance",
        )


def main():
    load_theme_css()

    missing = [label for label, path in DATASETS.items() if not path.exists()]
    if missing:
        st.error(
            "Missing required CSV files: "
            + ", ".join(missing)
            + ". Run `.\\.venv\\Scripts\\python.exe data\\generate_data.py` first."
        )
        st.stop()

    loaded = load_all_data()
    current_page = _sync_page_state()
    render_sidebar(loaded, current_page)

    insights = get_all_insights(
        loaded["Customers"],
        loaded["Campaigns"],
        loaded["Route Demand"],
        loaded["Digital Touchpoints"],
    )

    if current_page == "Overview":
        render_overview(loaded, insights)
    elif current_page == "Customer Intelligence":
        render_customer_intelligence_module(
            loaded["Customers"],
            loaded["Campaigns"],
            loaded["Route Demand"],
            loaded["Digital Touchpoints"],
            insights,
        )
    elif current_page == "Demand Intelligence":
        render_demand_intelligence_module(
            loaded["Customers"],
            loaded["Campaigns"],
            loaded["Route Demand"],
            loaded["Digital Touchpoints"],
            insights,
        )
    elif current_page == "Marketing Mix":
        render_marketing_mix_module(
            loaded["Customers"],
            loaded["Campaigns"],
            loaded["Route Demand"],
            loaded["Digital Touchpoints"],
            insights,
        )
    elif current_page == "Digital Attribution":
        render_digital_attribution_module(
            loaded["Customers"],
            loaded["Campaigns"],
            loaded["Route Demand"],
            loaded["Digital Touchpoints"],
            insights,
        )
    elif current_page == "Campaign Studio":
        render_campaign_studio_module(
            loaded["Customers"],
            loaded["Campaigns"],
            loaded["Route Demand"],
            loaded["Digital Touchpoints"],
            insights,
        )
    elif current_page == "AI Governance":
        render_ai_governance_module(
            loaded["Customers"],
            loaded["Campaigns"],
            loaded["Route Demand"],
            loaded["Digital Touchpoints"],
        )


if __name__ == "__main__":
    main()
