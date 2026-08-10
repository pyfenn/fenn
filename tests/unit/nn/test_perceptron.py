"""Tests for fenn/nn/models/perceptron.py"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from fenn.nn.models.perceptron import SingleLayerPerceptronClassifier, SingleLayerPerceptronRegressor


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _mock_rich_progress():
    def _fake_progress(*args, **kwargs):
        mock_progress = MagicMock()
        mock_progress.add_task.return_value = MagicMock()
        return mock_progress

    with patch(
        "fenn.nn.trainers.classification_trainer.Progress", side_effect=_fake_progress
    ):
        with patch(
            "fenn.nn.trainers.regression_trainer.Progress", side_effect=_fake_progress
        ):
            yield


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_classification_data(n_samples=40, n_features=4, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=(n_features,))
    scores = X @ weights
    if n_classes == 2:
        y = (scores > np.median(scores)).astype(int)
    else:
        thresholds = np.quantile(scores, np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.digitize(scores, thresholds)
    return X, y


def _make_regression_data(n_samples=40, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=(n_features,))
    y = X @ weights + 0.01 * rng.normal(size=(n_samples,))
    return X, y


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_slp_classifier_init():
    clf = SingleLayerPerceptronClassifier(learning_rate_init=0.01, max_iter=10)
    assert clf.hidden_layer_sizes == ()
    assert clf.activation == "identity"
    assert clf.learning_rate_init == 0.01


def test_slp_regressor_init():
    reg = SingleLayerPerceptronRegressor(learning_rate_init=0.01, max_iter=10)
    assert reg.hidden_layer_sizes == ()
    assert reg.activation == "identity"
    assert reg.learning_rate_init == 0.01


def test_slp_classifier_fit_predict_binary():
    X, y = _make_classification_data(n_classes=2)
    clf = SingleLayerPerceptronClassifier(max_iter=5)
    clf.fit(X, y)
    
    assert clf.n_features_in_ == X.shape[1]
    
    # Check predictions
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    assert set(np.unique(y_pred)).issubset({0, 1})


def test_slp_regressor_fit_predict():
    X, y = _make_regression_data()
    reg = SingleLayerPerceptronRegressor(max_iter=5)
    reg.fit(X, y)
    
    assert reg.n_features_in_ == X.shape[1]
    
    # Check predictions
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape
    assert y_pred.dtype.kind == "f"
