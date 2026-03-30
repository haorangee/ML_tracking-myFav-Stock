"""
tests/test_model.py
--------------------
Unit tests for model.py – training, evaluation, and persistence.

All tests use synthetic in-memory data, no network access required.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features import build_features, FEATURE_COLS
from model import (
    build_pipeline,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    cross_validate_model,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def make_feature_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df_raw = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.5, n),
            "High": close + rng.uniform(0, 2, n),
            "Low": close - rng.uniform(0, 2, n),
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=dates,
    )
    df_raw.index.name = "Date"
    return build_features(df_raw)


DF = make_feature_df()
AVAILABLE = [c for c in FEATURE_COLS if c in DF.columns]
SPLIT = int(len(DF) * 0.8)
X_TRAIN = DF.iloc[:SPLIT][AVAILABLE]
Y_TRAIN = DF.iloc[:SPLIT]["target"]
X_TEST = DF.iloc[SPLIT:][AVAILABLE]
Y_TEST = DF.iloc[SPLIT:]["target"]


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_rf_pipeline_has_scaler_and_regressor(self):
        pipe = build_pipeline("rf")
        assert "scaler" in pipe.named_steps
        assert "regressor" in pipe.named_steps

    def test_gbm_pipeline(self):
        pipe = build_pipeline("gbm")
        assert "regressor" in pipe.named_steps

    def test_unknown_type_defaults_to_rf(self):
        # Unknown type falls back to RF branch
        pipe = build_pipeline("unknown")
        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(pipe.named_steps["regressor"], RandomForestRegressor)


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_returns_fitted_pipeline(self):
        model = train_model(X_TRAIN, Y_TRAIN)
        preds = model.predict(X_TEST)
        assert len(preds) == len(X_TEST)

    def test_predictions_are_finite(self):
        model = train_model(X_TRAIN, Y_TRAIN)
        preds = model.predict(X_TEST)
        assert np.all(np.isfinite(preds))

    def test_gbm_model_type(self):
        model = train_model(X_TRAIN, Y_TRAIN, model_type="gbm")
        from sklearn.ensemble import GradientBoostingRegressor
        assert isinstance(model.named_steps["regressor"], GradientBoostingRegressor)


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def setup_method(self):
        self.model = train_model(X_TRAIN, Y_TRAIN)
        self.metrics = evaluate_model(self.model, X_TEST, Y_TEST)

    def test_metrics_keys(self):
        for key in ("mae", "rmse", "r2"):
            assert key in self.metrics

    def test_mae_positive(self):
        assert self.metrics["mae"] > 0

    def test_rmse_positive(self):
        assert self.metrics["rmse"] > 0

    def test_r2_is_float(self):
        assert isinstance(self.metrics["r2"], float)

    def test_rmse_geq_mae(self):
        assert self.metrics["rmse"] >= self.metrics["mae"]


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_predictions_identical_after_round_trip(self):
        model = train_model(X_TRAIN, Y_TRAIN)
        preds_before = model.predict(X_TEST)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            save_model(model, path)
            loaded = load_model(path)
            preds_after = loaded.predict(X_TEST)
            np.testing.assert_allclose(preds_before, preds_after)
        finally:
            os.unlink(path)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.joblib")


# ---------------------------------------------------------------------------
# cross_validate_model
# ---------------------------------------------------------------------------

class TestCrossValidate:
    def test_returns_rmse_stats(self):
        results = cross_validate_model(DF[AVAILABLE], DF["target"], n_splits=3)
        assert "rmse_mean" in results
        assert "rmse_std" in results

    def test_rmse_mean_positive(self):
        results = cross_validate_model(DF[AVAILABLE], DF["target"], n_splits=3)
        assert results["rmse_mean"] > 0
