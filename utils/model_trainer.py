"""
Model Trainer — unified wrapper so every module trains consistently.
Handles train/test split, fitting, and metric reporting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from utils.metrics import regression_metrics, classification_metrics


def train_regression_model(model, X: pd.DataFrame, y: pd.Series,
                            test_size: float = 0.2, scale: bool = False):
    """
    Train any sklearn-compatible regression model.
    Returns: trained model, metrics dict, X_test, y_test, y_pred
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    if scale:
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    metrics = regression_metrics(y_test.values, y_pred)

    return model, metrics, X_test, y_test, y_pred


def train_classification_model(model, X: pd.DataFrame, y: pd.Series,
                                test_size: float = 0.2, scale: bool = False):
    """
    Train any sklearn-compatible classification model.
    Returns: trained model, metrics dict, X_test, y_test, y_pred, y_prob
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    if scale:
        scaler  = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test  = pd.DataFrame(scaler.transform(X_test),  columns=X.columns)

    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = classification_metrics(y_test.values, y_pred, y_prob)

    return model, metrics, X_test, y_test, y_pred, y_prob