"""
Synthetic monitoring outputs used by the AI governance dashboard.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

MODEL_SPECS = [
    ("CLV Predictor", "RMSE", 1180, 1400, False),
    ("Churn Classifier", "AUC-ROC", 0.84, 0.80, True),
    ("RFM Segmentation", "Silhouette", 0.57, 0.50, True),
    ("Demand Forecaster", "MAPE", 4.8, 8.0, False),
    ("Marketing Mix Model", "R-squared", 0.79, 0.72, True),
    ("Markov Attribution", "Removal Effect Stability", 0.88, 0.75, True),
]


def current_model_health() -> pd.DataFrame:
    """Return point-in-time health metrics for each registered model."""
    rows = []
    for model, metric, value, threshold, higher_is_better in MODEL_SPECS:
        healthy = value >= threshold if higher_is_better else value <= threshold
        rows.append(
            {
                "model": model,
                "metric": metric,
                "value": round(value, 3) if value < 10 else round(value, 1),
                "threshold": threshold,
                "healthy": healthy,
                "status": "Healthy" if healthy else "Needs Review",
            }
        )
    return pd.DataFrame(rows)


def generate_model_history(weeks: int = 24) -> pd.DataFrame:
    """Create a short synthetic performance history for dashboard charts."""
    week_range = list(range(1, weeks + 1))
    records = []

    for model, metric, value, threshold, higher_is_better in MODEL_SPECS:
        for week in week_range:
            drift_bias = (week - weeks * 0.55) * (0.012 if higher_is_better else 0.08)
            noise = rng.normal(0, 0.015 if value < 10 else 15)
            hist_value = value - drift_bias + noise if higher_is_better else value + drift_bias + noise
            if metric == "AUC-ROC" or metric == "R-squared" or metric == "Silhouette":
                hist_value = float(np.clip(hist_value, 0.2, 0.99))
            records.append(
                {
                    "week": week,
                    "model": model,
                    "metric": metric,
                    "value": round(hist_value, 3) if hist_value < 10 else round(hist_value, 1),
                    "threshold": threshold,
                }
            )

    return pd.DataFrame(records)


def data_quality_checks(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Run lightweight data-quality checks for a given dataset."""
    checks = []

    for column in df.columns:
        null_rate = float(df[column].isna().mean())
        duplicate_rate = float(df[column].duplicated().mean()) if df[column].dtype == "object" else 0.0
        status = "OK"
        issue = "Within expected bounds"
        if null_rate > 0.05:
            status = "Issue"
            issue = "High null rate"
        elif duplicate_rate > 0.95 and column.endswith("_id") is False:
            status = "Issue"
            issue = "Very low cardinality"

        checks.append(
            {
                "Dataset": dataset_name,
                "Column": column,
                "Null Rate": round(null_rate * 100, 2),
                "Distinct Values": int(df[column].nunique(dropna=True)),
                "Status": "OK" if status == "OK" else "Issue",
                "Comment": issue,
            }
        )

    return pd.DataFrame(checks)
