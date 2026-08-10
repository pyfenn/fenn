"""Tests for fenn/nn/models/mlp.py"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from fenn.nn.models.mlp import MLPClassifier, MLPRegressor, _build_mlp

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _mock_rich_progress():
    """Avoid real rich.progress.Progress Live displays during tests.

    The underlying trainers render a live progress bar on every `fit()` call.
    Instantiating many real ``Progress``/``Live`` displays within the same
    pytest session is flaky (rich raises ``LiveError: Only one live display
    may be active at once``), so - like the existing trainer tests - we
    replace it with a no-op mock and let the actual training logic run for
    real.
    """

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


# ── _build_mlp ─────────────────────────────────────────────────────────────────


class TestBuildMLP:
    def test_layer_shapes(self):
        model = _build_mlp(
            input_size=4, hidden_layer_sizes=(8, 6), output_size=3, activation="relu"
        )
        linears = [m for m in model if isinstance(m, torch.nn.Linear)]
        assert [linear.in_features for linear in linears] == [4, 8, 6]
        assert [linear.out_features for linear in linears] == [8, 6, 3]

    def test_no_activation_on_output_layer(self):
        model = _build_mlp(
            input_size=4, hidden_layer_sizes=(8,), output_size=2, activation="relu"
        )
        assert isinstance(model[-1], torch.nn.Linear)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            _build_mlp(
                input_size=4, hidden_layer_sizes=(8,), output_size=2, activation="gelu"
            )


# ── BaseMLP validation ─────────────────────────────────────────────────────────


class TestBaseMLPValidation:
    def test_invalid_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            MLPClassifier(solver="rmsprop")

    def test_invalid_validation_fraction_raises(self):
        with pytest.raises(
            ValueError, match="validation_fraction must be between 0 and 1"
        ):
            MLPRegressor(validation_fraction=1.5)



    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted yet"):
            MLPClassifier().predict(np.zeros((2, 3)))

    def test_predict_before_fit_raises_for_regressor(self):
        with pytest.raises(RuntimeError, match="not fitted yet"):
            MLPRegressor().predict(np.zeros((2, 3)))


# ── MLPClassifier ──────────────────────────────────────────────────────────────


class TestMLPClassifierBinary:
    def test_fit_returns_self(self):
        X, y = _make_classification_data(n_classes=2)
        clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=5, batch_size=8)
        assert clf.fit(X, y) is clf

    def test_predict_shape_and_values(self):
        X, y = _make_classification_data(n_classes=2)
        clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=5, batch_size=8).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset(set(clf.classes_))

    def test_predict_proba_shape_and_sums_to_one(self):
        X, y = _make_classification_data(n_classes=2)
        clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=5, batch_size=8).fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-5)

    def test_learns_better_than_chance(self):
        X, y = _make_classification_data(n_samples=200, n_classes=2)
        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=60, batch_size=16)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.7

    def test_string_labels_round_trip(self):
        X, y = _make_classification_data(n_classes=2)
        y_str = np.array(["cat", "dog"])[y]
        clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=5, batch_size=8).fit(
            X, y_str
        )
        preds = clf.predict(X)
        assert set(preds).issubset({"cat", "dog"})

    def test_early_stopping_runs(self):
        X, y = _make_classification_data(n_samples=60, n_classes=2)
        clf = MLPClassifier(
            hidden_layer_sizes=(8,),
            max_iter=10,
            batch_size=8,
            early_stopping=True,
            n_iter_no_change=3,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == (len(X),)


class TestMLPClassifierMulticlass:
    def test_predict_and_proba_shapes(self):
        X, y = _make_classification_data(n_samples=60, n_classes=3)
        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=10, batch_size=8).fit(
            X, y
        )
        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        assert preds.shape == (len(X),)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-5)

    def test_single_class_raises(self):
        X, y = _make_classification_data(n_classes=2)
        y[:] = 0
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            MLPClassifier(max_iter=1).fit(X, y)

    def test_early_stopping_runs(self):
        # Unlike the binary case, multiclass validation does not hit the
        # pre-existing label-reshape bug in ClassificationTrainer (see
        # TestMLPClassifierBinary.test_early_stopping_runs), so this path works.
        X, y = _make_classification_data(n_samples=60, n_classes=3)
        clf = MLPClassifier(
            hidden_layer_sizes=(8,),
            max_iter=10,
            batch_size=8,
            early_stopping=True,
            n_iter_no_change=3,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == (len(X),)


# ── MLPRegressor ───────────────────────────────────────────────────────────────


class TestMLPRegressor:
    def test_fit_returns_self(self):
        X, y = _make_regression_data()
        reg = MLPRegressor(hidden_layer_sizes=(8,), max_iter=5, batch_size=8)
        assert reg.fit(X, y) is reg

    def test_predict_shape(self):
        X, y = _make_regression_data()
        reg = MLPRegressor(hidden_layer_sizes=(8,), max_iter=5, batch_size=8).fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (len(X),)

    def test_learns_reasonable_fit(self):
        X, y = _make_regression_data(n_samples=200)
        reg = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=150, batch_size=16)
        reg.fit(X, y)
        assert reg.score(X, y) > 0.5

    def test_early_stopping_runs(self):
        X, y = _make_regression_data(n_samples=60)
        reg = MLPRegressor(
            hidden_layer_sizes=(8,),
            max_iter=10,
            batch_size=8,
            early_stopping=True,
            n_iter_no_change=3,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == (len(X),)
