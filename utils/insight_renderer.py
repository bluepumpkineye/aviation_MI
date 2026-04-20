"""
Reusable Streamlit rendering helpers for decision intelligence.
"""

from __future__ import annotations

import html
import re

import streamlit as st


THEME = {
    "primary": "#00A3A1",
    "accent": "#C4973B",
    "danger": "#E05A40",
    "success": "#00C896",
    "text_primary": "#F0F4F8",
    "text_muted": "#4A5A6A",
}

SEVERITY_COLORS = {
    "critical": THEME["danger"],
    "opportunity": THEME["accent"],
    "on_track": THEME["success"],
}

SEVERITY_CLASSES = {
    "critical": "critical",
    "opportunity": "opportunity",
    "on_track": "on-track",
}

MODULE_BADGES = {
    "Customer Intelligence": "CI",
    "Demand Intelligence": "DI",
    "Marketing Mix": "MM",
    "Digital Attribution": "DA",
    "Campaign Studio": "CS",
    "AI Governance": "AG",
}

HEALTH_COLORS = [
    (80, THEME["success"], "Strong"),
    (60, THEME["primary"], "Good"),
    (40, THEME["accent"], "Needs Attention"),
    (0, THEME["danger"], "At Risk"),
]


def _compact_html(value: str) -> str:
    return re.sub(r">\s+<", "><", value.strip())


def _module_badge(module: str) -> str:
    first_module = str(module).split("+")[0].strip()
    return MODULE_BADGES.get(first_module, "MI")


def _severity_style(severity: str) -> tuple[str, str]:
    color = SEVERITY_COLORS.get(severity, THEME["primary"])
    severity_class = SEVERITY_CLASSES.get(severity, "opportunity")
    return color, severity_class


def render_insight_banner(insight: dict):
    """Render a single insight as a styled banner."""
    color, severity_class = _severity_style(insight["severity"])
    urgency_text = "Immediate" if insight["urgency_days"] <= 0 else f"Act within {insight['urgency_days']} days"
    title = html.escape(str(insight.get("title", "")))
    body = html.escape(str(insight.get("body", "")))
    action = html.escape(str(insight.get("action", "")))
    metric = html.escape(str(insight.get("metric", "")))
    badge = _module_badge(str(insight.get("module", "")))
    st.markdown(
        f"""
        <div class="insight-card insight-card--{severity_class}">
          <div class="insight-card__header">
            <div class="insight-card__left">
              <span class="insight-card__module" style="background:{color}1F;border-color:{color}40;color:{color};">{badge}</span>
              <span class="insight-card__title">{title}</span>
            </div>
            <div class="insight-card__metric" style="color:{color};">{metric}</div>
          </div>
          <div class="insight-card__body">{body}</div>
          <div class="insight-card__action">
            <div class="insight-card__action-text">
              <span class="insight-card__action-label">Action:</span>{action}
            </div>
            <div class="insight-card__urgency" style="color:{color};">{html.escape(urgency_text)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_panel(insights: list, max_show: int = 3):
    """Render multiple insight banners with overflow in an expander."""
    if not insights:
        return
    for insight in insights[:max_show]:
        render_insight_banner(insight)
    remaining = insights[max_show:]
    if remaining:
        with st.expander(f"Show {len(remaining)} more"):
            for insight in remaining:
                render_insight_banner(insight)


def render_what_this_means(text: str, action: str | None = None):
    """Render a contextual interpretation box below a chart."""
    action_html = ""
    if action:
        action_html = (
            f'<div class="what-this-means__action"><strong>Recommended Action:</strong> {html.escape(action)}</div>'
        )
    st.markdown(
        f"""
        <div class="what-this-means">
          <div class="what-this-means__label">What This Means</div>
          <div class="what-this-means__body">{html.escape(text)}</div>
          {action_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_card(
    title: str,
    metrics: dict,
    actions: list,
    impact: str,
    urgency: str,
    color: str = "opportunity",
):
    """Render a structured decision card."""
    border_color, severity_class = _severity_style(color)
    metric_html = "".join(
        _compact_html(f"""
        <div class="decision-card__metric">
          <div class="decision-card__metric-label">{html.escape(str(label))}</div>
          <div class="decision-card__metric-value">{html.escape(str(value))}</div>
        </div>
        """)
        for label, value in metrics.items()
    )
    actions_html = "".join(
        f"<li>{html.escape(str(action_item))}</li>"
        for action_item in actions
    )
    st.markdown(
        _compact_html(f"""
        <div class="decision-card decision-card--{severity_class}">
          <div class="decision-card__header">
            <div class="decision-card__title">{html.escape(title)}</div>
            <div class="decision-card__urgency" style="color:{border_color};">{html.escape(urgency)}</div>
          </div>
          <div class="decision-card__metrics">{metric_html}</div>
          <ol class="decision-card__actions">{actions_html}</ol>
          <div class="decision-card__impact"><strong>If we act:</strong> {html.escape(impact)}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_health_score(score: int, label: str):
    """Render a health score with the shared card style."""
    render_health_score_widget(score, label)


def render_cross_module_signal(signal: dict):
    """Render a cross-module signal card."""
    render_insight_banner(signal)


def render_health_score_widget(score: int, label: str):
    """Render the full-width health score widget."""
    color = THEME["danger"]
    status = label.upper()
    for threshold, palette_color, default_label in HEALTH_COLORS:
        if score >= threshold:
            color = palette_color
            status = label.upper() if label else default_label.upper()
            break

    st.markdown(
        f"""
        <div class="health-score-card">
          <div class="health-score-left">
            <div class="health-score-value" style="color:{color};">{score}</div>
            <div class="health-score-max">/ 100</div>
          </div>
          <div class="health-score-center">
            <div class="health-score-title">Portfolio Health Score</div>
            <div class="health-score-track">
              <div class="health-score-fill" style="width:{max(min(score, 100), 0)}%;background:{color};"></div>
            </div>
            <div class="health-score-caption">{html.escape(label)}</div>
          </div>
          <div class="health-score-status" style="color:{color};background:{color}1A;">{html.escape(status)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
