"""
CLV regression model used by the customer intelligence UI.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from utils.metrics import regression_metrics


FEATURE_COLUMNS = [
    "days_since_last_booking",
    "trips_last_12m",
    "lifetime_bookings",
    "avg_trip_value_hkd",
    "ancillary_spend_hkd",
    "total_revenue_hkd",
    "email_open_rate",
    "app_engagement_score",
    "loyalty_tier",
    "preferred_cabin_class",
    "home_market",
]


def train_clv_model(df: pd.DataFrame) -> dict:
    """Train a lightweight CLV model and return metrics and diagnostics."""
    model_df = df.copy()
    X = pd.get_dummies(model_df[FEATURE_COLUMNS], drop_first=False)
    y = model_df["customer_lifetime_value_12m"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=160,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = regression_metrics(y_test.values, predictions)
    feature_importance = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(12)
        .reset_index(drop=True)
    )
    actual_vs_pred = pd.DataFrame(
        {
            "actual_clv_hkd": y_test.values,
            "predicted_clv_hkd": predictions.round(0),
        }
    )

    return {
        "metrics": metrics,
        "feature_importance": feature_importance,
        "actual_vs_pred": actual_vs_pred,
    }
