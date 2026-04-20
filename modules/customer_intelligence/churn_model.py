"""
Churn classification model used by the customer intelligence UI.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils.metrics import classification_metrics


FEATURE_COLUMNS = [
    "days_since_last_booking",
    "trips_last_12m",
    "lifetime_bookings",
    "avg_trip_value_hkd",
    "ancillary_spend_hkd",
    "email_open_rate",
    "app_engagement_score",
    "customer_lifetime_value_12m",
    "loyalty_tier",
    "preferred_cabin_class",
    "home_market",
]


def train_churn_model(df: pd.DataFrame, threshold: float = 0.55) -> dict:
    """Train a churn classifier based on the synthetic customer dataset."""
    model_df = df.copy()
    model_df["is_high_risk"] = (model_df["churn_probability_90d"] >= threshold).astype(int)

    X = pd.get_dummies(model_df[FEATURE_COLUMNS], drop_first=False)
    y = model_df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=9,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test.values, y_pred, y_prob)
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

    scored_customers = model_df.copy()
    scored_customers["predicted_churn_risk"] = model.predict_proba(X)[:, 1]
    scored_customers["risk_band"] = pd.cut(
        scored_customers["predicted_churn_risk"],
        bins=[0, 0.35, 0.55, 0.75, 1],
        labels=["Low", "Watch", "Elevated", "Critical"],
        include_lowest=True,
    )

    return {
        "metrics": metrics,
        "feature_importance": feature_importance,
        "scored_customers": scored_customers,
    }
