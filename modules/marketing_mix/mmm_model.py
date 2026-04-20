"""
Marketing mix model helpers.
"""

import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from config.settings import CHANNELS
from utils.metrics import regression_metrics


def prepare_mmm_dataset(df_campaigns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate campaign data into a channel-spend matrix by month and market."""
    campaigns = df_campaigns.copy()
    campaigns["start_date"] = pd.to_datetime(campaigns["start_date"])
    campaigns["month"] = campaigns["start_date"].dt.to_period("M").dt.to_timestamp()

    spend_matrix = (
        campaigns.pivot_table(
            index=["month", "market"],
            columns="channel",
            values="spend_hkd",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=CHANNELS, fill_value=0)
        .reset_index()
    )
    outcomes = (
        campaigns.groupby(["month", "market"])
        .agg(
            revenue_hkd=("revenue_hkd", "sum"),
            bookings=("bookings", "sum"),
            impressions=("impressions", "sum"),
        )
        .reset_index()
    )
    return spend_matrix.merge(outcomes, on=["month", "market"], how="left")


def fit_mmm_model(df_campaigns: pd.DataFrame) -> dict:
    """Train a tree-based MMM and return diagnostics for the UI."""
    dataset = prepare_mmm_dataset(df_campaigns)
    X = dataset[CHANNELS]
    y = dataset["revenue_hkd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=220,
        max_depth=9,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = regression_metrics(y_test.values, y_pred)
    feature_importance = (
        pd.DataFrame({"channel": CHANNELS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_summary = (
        pd.DataFrame(
            {
                "channel": CHANNELS,
                "mean_abs_shap": abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    actual_vs_pred = pd.DataFrame(
        {
            "actual_revenue_hkd": y_test.values,
            "predicted_revenue_hkd": y_pred.round(0),
        }
    )

    return {
        "dataset": dataset,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "shap_summary": shap_summary,
        "actual_vs_pred": actual_vs_pred,
    }


def channel_efficiency_summary(df_campaigns: pd.DataFrame) -> pd.DataFrame:
    """Summarize channel efficiency statistics from campaign history."""
    summary = (
        df_campaigns.groupby("channel")
        .agg(
            spend_hkd=("spend_hkd", "sum"),
            revenue_hkd=("revenue_hkd", "sum"),
            bookings=("bookings", "sum"),
            conversion_rate=("conversion_rate", "mean"),
            ctr=("ctr", "mean"),
        )
        .reset_index()
    )
    summary["roas"] = (summary["revenue_hkd"] / summary["spend_hkd"].clip(lower=1)).round(2)
    summary["revenue_per_booking"] = (
        summary["revenue_hkd"] / summary["bookings"].clip(lower=1)
    ).round(0)
    return summary.sort_values("roas", ascending=False).reset_index(drop=True)
