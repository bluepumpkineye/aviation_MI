"""
Shared formatting and presentation helpers for the Streamlit app.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st


THEME = {
    "bg_base": "#070D1A",
    "bg_surface": "#0C1829",
    "bg_elevated": "#111F35",
    "bg_overlay": "#162641",
    "bg_sidebar": "#080F1E",
    "border_subtle": "#1A2F4A",
    "border_default": "#1E3A5F",
    "border_strong": "#2A4A6F",
    "primary": "#00A3A1",
    "primary_dim": "#006E6C",
    "accent": "#C4973B",
    "accent_dim": "#8A6A28",
    "success": "#00C896",
    "warning": "#F0A050",
    "danger": "#E05A40",
    "info": "#4A90D9",
    "text_primary": "#F0F4F8",
    "text_secondary": "#8A9BB0",
    "text_muted": "#4A5A6A",
}

CHART_COLOR_SEQUENCE = [
    "#00A3A1",
    "#C4973B",
    "#4A90D9",
    "#00C896",
    "#F0A050",
    "#E05A40",
    "#9B6DD4",
    "#50C8D4",
]


def _compact_html(value: str) -> str:
    return re.sub(r">\s+<", "><", value.strip())


@st.cache_resource(show_spinner=False)
def _load_theme_css_text() -> str:
    css_path = Path(__file__).resolve().parent.parent / "assets" / "theme.css"
    return css_path.read_text(encoding="utf-8")


def load_theme_css():
    """Inject the shared theme CSS into the app."""
    st.markdown(f"<style>{_load_theme_css_text()}</style>", unsafe_allow_html=True)


def format_currency(value: float | int | str) -> str:
    """Format a numeric value as HKD."""
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"HK${value / 1_000_000:.1f}M"
    return f"HK${value:,.0f}"


def format_percentage(value: float | int | str) -> str:
    """Format a numeric value as a percentage."""
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) <= 1.5:
        value *= 100
    return f"{value:.1f}%"


def snake_to_title(text: str) -> str:
    """Convert snake_case to plain English title case."""
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.title()


def format_dataframe_display(
    df: pd.DataFrame,
    currency_cols: list | None = None,
    pct_cols: list | None = None,
    rename_map: dict | None = None,
) -> pd.DataFrame:
    """Format dataframe values and replace raw snake_case headers."""
    formatted = df.copy()
    currency_cols = currency_cols or []
    pct_cols = pct_cols or []
    rename_map = rename_map or {}

    for column in currency_cols:
        if column in formatted.columns:
            formatted[column] = formatted[column].apply(format_currency)

    for column in pct_cols:
        if column in formatted.columns:
            formatted[column] = formatted[column].apply(format_percentage)

    formatted.columns = [
        rename_map.get(column, snake_to_title(column))
        for column in formatted.columns
    ]
    return formatted


def labeled_divider(text: str):
    """Render the shared editorial section divider."""
    st.markdown(
        f"""
        <div class="section-header">
          <div class="section-line"></div>
          <span class="section-label">{html.escape(text)}</span>
          <div class="section-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tech_pills(techniques: list[str]) -> str:
    """Return HTML for the technique tags."""
    if not techniques:
        return ""
    pills = "".join(
        f'<span class="tech-pill">{html.escape(item)}</span>'
        for item in techniques
    )
    return _compact_html(f'<div class="tech-pills-row">{pills}</div>')


def render_page_header(
    title: str,
    subtitle: str,
    pills: list[str] | None = None,
    meta: str | None = None,
):
    """Render the shared page title block."""
    pills_html = render_tech_pills(pills or [])
    meta_html = f'<div class="page-meta">{html.escape(meta)}</div>' if meta else ""
    st.markdown(
        _compact_html(f"""
        <div class="page-header">
          <div class="page-title-row">
            <h1 class="page-title">{html.escape(title)}</h1>
            {meta_html}
          </div>
          <p class="page-subtitle">{html.escape(subtitle)}</p>
          {pills_html}
        </div>
        """),
        unsafe_allow_html=True,
    )


def _delta_class(delta: str | None, state: str | None = None) -> str:
    if state in {"positive", "negative", "neutral"}:
        return f"delta--{state}"
    if not delta:
        return "delta--neutral"
    stripped = str(delta).strip().lower()
    if stripped.startswith("-"):
        return "delta--negative"
    if stripped.startswith("+") or "up" in stripped or "best" in stripped or "healthy" in stripped:
        return "delta--positive"
    return "delta--neutral"


def render_metric_grid(metrics: list[dict], columns: int = 4):
    """Render standalone metric cards."""
    cards = []
    for item in metrics:
        label = html.escape(str(item.get("label", "")))
        value = html.escape(str(item.get("value", "")))
        delta = str(item.get("delta", "")).strip()
        delta_html = ""
        if delta:
            delta_html = (
                f'<div class="metric-card__delta {_delta_class(delta, item.get("delta_state"))}">'
                f"{html.escape(delta)}</div>"
            )
        cards.append(
            _compact_html(f"""
            <div class="metric-card">
              <div class="metric-card__label">{label}</div>
              <div class="metric-card__value">{value}</div>
              {delta_html}
            </div>
            """)
        )
    grid_class = "metric-grid--4" if columns == 4 else "metric-grid--3"
    st.markdown(
        _compact_html(f'<div class="metric-grid {grid_class}">{"".join(cards)}</div>'),
        unsafe_allow_html=True,
    )


def render_metric_strip(metrics: list[dict], columns: int = 4):
    """Render a unified multi-metric strip."""
    cells = []
    for item in metrics:
        label = html.escape(str(item.get("label", "")))
        value = html.escape(str(item.get("value", "")))
        delta = str(item.get("delta", "")).strip()
        delta_html = ""
        if delta:
            delta_html = (
                f'<div class="metric-strip__delta {_delta_class(delta, item.get("delta_state"))}">'
                f"{html.escape(delta)}</div>"
            )
        cells.append(
            _compact_html(f"""
            <div class="metric-strip__cell">
              <div class="metric-strip__label">{label}</div>
              <div class="metric-strip__value">{value}</div>
              {delta_html}
            </div>
            """)
        )
    strip_class = "metric-strip--4" if columns == 4 else "metric-strip--3"
    st.markdown(
        _compact_html(f'<div class="metric-strip {strip_class}">{"".join(cells)}</div>'),
        unsafe_allow_html=True,
    )


def apply_theme(fig, title: str = ""):
    """Apply the premium dark Plotly theme."""
    resolved_title = title or ""
    if not resolved_title and getattr(fig.layout, "title", None):
        resolved_title = fig.layout.title.text or ""

    fig.update_layout(
        title=None,
        paper_bgcolor=THEME["bg_elevated"],
        plot_bgcolor=THEME["bg_elevated"],
        font={
            "family": "Inter, -apple-system, sans-serif",
            "size": 11,
            "color": THEME["text_secondary"],
        },
        colorway=CHART_COLOR_SEQUENCE,
        margin={"l": 48, "r": 24, "t": 40, "b": 48},
        legend={
            "bgcolor": THEME["bg_surface"],
            "bordercolor": THEME["border_subtle"],
            "borderwidth": 1,
            "font": {
                "family": "Inter, -apple-system, sans-serif",
                "size": 11,
                "color": THEME["text_secondary"],
            },
        },
        hoverlabel={
            "bgcolor": THEME["bg_overlay"],
            "bordercolor": THEME["border_strong"],
            "font": {
                "family": "Inter, -apple-system, sans-serif",
                "size": 11,
                "color": THEME["text_primary"],
            },
        },
        annotations=[
            {
                "text": html.escape(resolved_title.upper()),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "bottom",
                "showarrow": False,
                "font": {
                    "family": "Inter, -apple-system, sans-serif",
                    "size": 12,
                    "color": THEME["text_secondary"],
                },
            }
        ] if resolved_title else [],
    )
    fig.update_xaxes(
        gridcolor=THEME["border_subtle"],
        gridwidth=1,
        showline=True,
        linecolor=THEME["border_default"],
        linewidth=1,
        zeroline=False,
        tickfont={"family": "Inter, -apple-system, sans-serif", "size": 11, "color": THEME["text_secondary"]},
        title_font={"family": "Inter, -apple-system, sans-serif", "size": 11, "color": THEME["text_secondary"]},
    )
    fig.update_yaxes(
        gridcolor=THEME["border_subtle"],
        gridwidth=1,
        showline=True,
        linecolor=THEME["border_default"],
        linewidth=1,
        zeroline=False,
        tickfont={"family": "Inter, -apple-system, sans-serif", "size": 11, "color": THEME["text_secondary"]},
        title_font={"family": "Inter, -apple-system, sans-serif", "size": 11, "color": THEME["text_secondary"]},
    )
    return fig
