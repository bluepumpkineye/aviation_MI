"""
Campaign Timing Engine
Combines demand forecasts with booking window analysis
to produce precise campaign launch recommendations.

Answers: "For HKG-NRT in July, when exactly should
          we launch our campaign and on which channels?"
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config.settings import ROUTES, CHANNELS, DIGITAL_CHANNELS

rng = np.random.default_rng(42)

# ── Seasonal Campaign Calendar ────────────────────────────────────

PEAK_PERIODS = {
    "Chinese New Year": {"months": [1, 2], "lead_weeks": 8},
    "Spring Break":     {"months": [3, 4], "lead_weeks": 6},
    "Summer Peak":      {"months": [6, 7, 8], "lead_weeks": 10},
    "Golden Week":      {"months": [10],    "lead_weeks": 5},
    "Christmas/NY":     {"months": [12],    "lead_weeks": 8},
}

# Best channels per campaign objective
OBJECTIVE_CHANNELS = {
    "awareness":    ["YouTube / Video", "Programmatic Display", "Out-of-Home"],
    "consideration":["Paid Search", "Paid Social", "YouTube / Video"],
    "conversion":   ["Paid Search", "Email CRM", "Affiliate"],
    "retention":    ["Email CRM", "Paid Social"],
}

# Route category classification
ROUTE_CATEGORIES = {
    "leisure":  ["HKG-SYD", "HKG-NRT", "HKG-BKK", "HKG-MEL",
                 "HKG-ICN", "HKG-TPE", "HKG-MNL", "HKG-CDG"],
    "business": ["HKG-LHR", "HKG-JFK", "HKG-LAX", "HKG-FRA",
                 "HKG-DXB", "HKG-SIN", "HKG-YVR"],
}


def classify_route(route: str) -> str:
    """Classify a route as leisure or business dominated."""
    if route in ROUTE_CATEGORIES["leisure"]:
        return "leisure"
    if route in ROUTE_CATEGORIES["business"]:
        return "business"
    return "mixed"


def get_peak_period(month: int) -> str:
    """Return the peak period name for a given month, if any."""
    for period, info in PEAK_PERIODS.items():
        if month in info["months"]:
            return period
    return "Standard Period"


def generate_campaign_calendar(
    route:           str,
    departure_month: int,
    departure_year:  int,
    cabin:           str,
    budget_hkd:      float,
) -> pd.DataFrame:
    """
    Generate a week-by-week campaign calendar
    for a given route and departure period.

    Returns a DataFrame with recommended:
      - Campaign phase (Awareness / Consideration / Conversion)
      - Launch date per phase
      - Budget split per phase
      - Recommended channels per phase
      - KPI targets per phase
    """
    # Departure date (1st of target month)
    departure_date = datetime(departure_year, departure_month, 1)

    # Booking window for this route/cabin
    long_haul = route in [
        "HKG-LHR", "HKG-JFK", "HKG-LAX",
        "HKG-YVR", "HKG-FRA", "HKG-CDG"
    ]
    cabin_mult = {
        "Economy": 0.85, "Premium Economy": 1.0,
        "Business": 1.3, "First": 1.45
    }.get(cabin, 1.0)

    base_window = 55 if long_haul else 30
    total_window_days = int(base_window * cabin_mult)

    # Peak period bonus
    peak = get_peak_period(departure_month)
    peak_info = PEAK_PERIODS.get(peak, {})
    if peak_info:
        extra_days = peak_info.get("lead_weeks", 0) * 7
        total_window_days = max(total_window_days, extra_days)

    # Campaign phases
    route_type = classify_route(route)

    if route_type == "business":
        # Business routes: shorter awareness, stronger conversion
        phases = [
            {
                "phase":       "Awareness",
                "duration_days": int(total_window_days * 0.30),
                "budget_pct":  0.20,
                "objective":   "awareness",
                "kpi":         "Impressions & Reach",
                "kpi_target":  f"{int(budget_hkd * 0.20 / 0.04):,} impressions",
            },
            {
                "phase":       "Consideration",
                "duration_days": int(total_window_days * 0.35),
                "budget_pct":  0.35,
                "objective":   "consideration",
                "kpi":         "CTR & Engagement",
                "kpi_target":  "CTR > 2.5%",
            },
            {
                "phase":       "Conversion",
                "duration_days": int(total_window_days * 0.35),
                "budget_pct":  0.45,
                "objective":   "conversion",
                "kpi":         "Bookings & ROAS",
                "kpi_target":  f"ROAS > 4.0x",
            },
        ]
    else:
        # Leisure routes: longer awareness, emotional storytelling
        phases = [
            {
                "phase":       "Inspiration",
                "duration_days": int(total_window_days * 0.35),
                "budget_pct":  0.25,
                "objective":   "awareness",
                "kpi":         "Video Views & Reach",
                "kpi_target":  f"{int(budget_hkd * 0.25 / 0.02):,} views",
            },
            {
                "phase":       "Consideration",
                "duration_days": int(total_window_days * 0.30),
                "budget_pct":  0.30,
                "objective":   "consideration",
                "kpi":         "Site Sessions & Dwell Time",
                "kpi_target":  "CTR > 2.0%, Bounce < 55%",
            },
            {
                "phase":       "Conversion",
                "duration_days": int(total_window_days * 0.35),
                "budget_pct":  0.45,
                "objective":   "conversion",
                "kpi":         "Bookings & CAC",
                "kpi_target":  f"CAC < HK${int(budget_hkd * 0.45 / max(1, int(budget_hkd/8500))):,}",
            },
        ]

    # Build calendar rows
    records   = []
    phase_end = departure_date

    for phase in reversed(phases):
        phase_start = phase_end - timedelta(days=phase["duration_days"])
        channels    = OBJECTIVE_CHANNELS.get(phase["objective"], CHANNELS[:3])
        budget_amt  = budget_hkd * phase["budget_pct"]

        records.append({
            "phase":          phase["phase"],
            "start_date":     phase_start.strftime("%Y-%m-%d"),
            "end_date":       phase_end.strftime("%Y-%m-%d"),
            "duration_days":  phase["duration_days"],
            "budget_hkd":     int(budget_amt),
            "budget_pct":     f"{int(phase['budget_pct']*100)}%",
            "channels":       ", ".join(channels),
            "kpi":            phase["kpi"],
            "kpi_target":     phase["kpi_target"],
            "days_to_depart": (departure_date - phase_start).days,
        })
        phase_end = phase_start

    df = pd.DataFrame(records[::-1])   # chronological order
    return df


def route_timing_benchmarks(df_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical timing benchmarks from demand data.
    Shows which months have highest demand per route.
    Used to prioritise campaign investment calendar.
    """
    df = df_demand.copy()
    if "revenue_hkd" not in df.columns:
        if {"bookings", "avg_fare_hkd"}.issubset(df.columns):
            df["revenue_hkd"] = df["bookings"] * df["avg_fare_hkd"]
        else:
            df["revenue_hkd"] = 0

    df["date"]  = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year

    monthly = (
        df.groupby(["route", "month"])
        .agg(
            avg_bookings  =("bookings",     "mean"),
            avg_load_factor=("load_factor", "mean"),
            avg_fare      =("avg_fare_hkd", "mean"),
            avg_revenue   =("revenue_hkd",  "mean"),
        )
        .reset_index()
        .round(2)
    )

    # Add peak period label
    monthly["peak_period"] = monthly["month"].apply(get_peak_period)

    # Add recommended campaign launch month
    # (campaign should launch 6-8 weeks before peak)
    monthly["campaign_launch_month"] = monthly["month"].apply(
        lambda m: ((m - 2) % 12) + 1
    )

    return monthly.sort_values(
        ["route", "avg_bookings"], ascending=[True, False]
    )


def channel_timing_recommendations(
    route:     str,
    phase:     str,
    budget:    float,
) -> pd.DataFrame:
    """
    Recommend channel mix and spend for a specific campaign phase.
    Returns a DataFrame with channel, spend, and expected KPIs.
    """
    route_type = classify_route(route)

    if phase.lower() in ["awareness", "inspiration"]:
        weights = {
            "YouTube / Video":       0.35,
            "Programmatic Display":  0.25,
            "Out-of-Home":           0.20,
            "Paid Social":           0.15,
            "Paid Search":           0.05,
        }
    elif phase.lower() == "consideration":
        weights = {
            "Paid Search":           0.30,
            "Paid Social":           0.28,
            "YouTube / Video":       0.20,
            "Programmatic Display":  0.12,
            "Email CRM":             0.10,
        }
    else:   # Conversion
        weights = {
            "Paid Search":           0.40,
            "Email CRM":             0.25,
            "Paid Social":           0.15,
            "Affiliate":             0.12,
            "Programmatic Display":  0.08,
        }

    records = []
    for channel, weight in weights.items():
        spend = budget * weight
        # Estimated KPIs based on channel benchmarks
        cpm_map = {
            "YouTube / Video": 45, "Programmatic Display": 18,
            "Out-of-Home": 80, "Paid Social": 35,
            "Paid Search": 0, "Email CRM": 5,
            "Affiliate": 0,
        }
        cpm          = cpm_map.get(channel, 30)
        impressions  = int(spend / cpm * 1000) if cpm > 0 else 0
        est_bookings = int(spend / 8500)

        records.append({
            "channel":       channel,
            "spend_hkd":     int(spend),
            "spend_pct":     f"{int(weight*100)}%",
            "est_impressions": impressions,
            "est_bookings":  est_bookings,
            "rationale":     _channel_rationale(channel, phase),
        })

    return pd.DataFrame(records).sort_values(
        "spend_hkd", ascending=False
    )


def _channel_rationale(channel: str, phase: str) -> str:
    """Return a one-line rationale for using a channel in a phase."""
    rationales = {
        ("YouTube / Video",      "awareness"):    "High reach video storytelling drives brand consideration",
        ("Programmatic Display", "awareness"):    "Broad audience targeting at efficient CPM",
        ("Out-of-Home",          "awareness"):    "Premium placement builds route awareness in key markets",
        ("Paid Social",          "consideration"):"Retargeting warm audiences with route-specific content",
        ("Paid Search",          "consideration"):"Captures active travel intent signals",
        ("Paid Search",          "conversion"):   "Bottom-funnel intent capture at point of booking decision",
        ("Email CRM",            "conversion"):   "Personalised offers to loyalty base drives highest ROAS",
        ("Affiliate",            "conversion"):   "Performance-based spend with guaranteed booking ROI",
    }
    key = (channel, phase.lower())
    return rationales.get(
        key,
        f"{channel} supports {phase} objectives effectively"
    )
