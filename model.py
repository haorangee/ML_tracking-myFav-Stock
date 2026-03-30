"""
model.py
--------
Random Forest regression model for next-day NVDA close-price prediction.

Public API
----------
train_model(X_train, y_train)  -> sklearn estimator
evaluate_model(model, X_test, y_test) -> dict of metrics
save_model(model, path)
load_model(path)               -> sklearn estimator
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "nvda_rf_model.joblib")


def build_pipeline(model_type: str = "rf", **kwargs) -> Pipeline:
    """Return a sklearn Pipeline with a scaler and regressor.

    Parameters
    ----------
    model_type:
        ``"rf"`` for Random Forest (default) or ``"gbm"`` for Gradient Boosting.
    **kwargs:
        Extra keyword arguments forwarded to the regressor constructor.
    """
    if model_type == "gbm":
        regressor = GradientBoostingRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 4),
            learning_rate=kwargs.get("learning_rate", 0.05),
            subsample=kwargs.get("subsample", 0.8),
            random_state=42,
        )
    else:
        regressor = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", None),
            min_samples_leaf=kwargs.get("min_samples_leaf", 2),
            n_jobs=-1,
            random_state=42,
        )

    return Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])


def train_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    model_type: str = "rf",
    **kwargs,
) -> Pipeline:
    """Fit the model pipeline on training data.

    Parameters
    ----------
    X_train:
        Feature matrix.
    y_train:
        Target values (next-day close prices).
    model_type:
        ``"rf"`` or ``"gbm"``.

    Returns
    -------
    Pipeline
        Fitted sklearn pipeline.
    """
    pipeline = build_pipeline(model_type=model_type, **kwargs)
    pipeline.fit(X_train, y_train)
    return pipeline


def cross_validate_model(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    model_type: str = "rf",
    n_splits: int = 5,
) -> dict:
    """Time-series cross-validation.

    Parameters
    ----------
    X:
        Full feature matrix.
    y:
        Full target vector.
    n_splits:
        Number of time-series splits.

    Returns
    -------
    dict
        Mean and standard deviation of RMSE across folds.
    """
    pipeline = build_pipeline(model_type=model_type)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(
        pipeline, X, y, cv=tscv, scoring="neg_root_mean_squared_error"
    )
    rmse_scores = -scores
    return {"rmse_mean": rmse_scores.mean(), "rmse_std": rmse_scores.std()}


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> dict:
    """Compute regression metrics on the test set.

    Parameters
    ----------
    model:
        Fitted pipeline.
    X_test:
        Test feature matrix.
    y_test:
        True target values.

    Returns
    -------
    dict
        ``mae``, ``rmse``, ``r2`` metrics.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def save_model(model: Pipeline, path: str = DEFAULT_MODEL_PATH) -> None:
    """Persist the trained pipeline to disk.

    Parameters
    ----------
    model:
        Fitted sklearn pipeline.
    path:
        Destination file path (joblib format).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a previously saved pipeline from disk.

    Parameters
    ----------
    path:
        Path to the joblib file.

    Returns
    -------
    Pipeline
        The loaded sklearn pipeline.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{path}' not found. Run train.py first."
        )
    return joblib.load(path)
