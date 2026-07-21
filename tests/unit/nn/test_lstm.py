"""Tests for fenn/nn/models/lstm.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from fenn.nn.models.lstm import (
    LSTMClassifier,
    LSTMGenerator,
    _LSTMClassificationModel,
    _LSTMGenerationModel,
    _make_loader,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _mock_rich_progress():
    """Avoid real rich progress displays during tests."""

    def _fake_progress(*args, **kwargs):
        mock_progress = MagicMock()
        mock_progress.add_task.return_value = MagicMock()
        return mock_progress

    with patch(
        "fenn.nn.trainers.classification_trainer.Progress",
        side_effect=_fake_progress,
    ):
        yield


def _make_classification_data(
    n_samples=40, seq_len=5, n_features=3, n_classes=2, seed=0
):
    """Create simple synthetic sequence classification data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, seq_len, n_features))
    scores = X.mean(axis=(1, 2))

    if n_classes == 2:
        y = (scores > np.median(scores)).astype(int)
    else:
        thresholds = np.quantile(
            scores,
            np.linspace(0, 1, n_classes + 1)[1:-1],
        )
        y = np.digitize(scores, thresholds)

    return X, y


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_clf(**kwargs):
    defaults = dict(hidden_size=8, max_iter=3, batch_size=8)
    return LSTMClassifier(**{**defaults, **kwargs})


def _make_gen(**kwargs):
    defaults = dict(
        vocab_size=8, embedding_dim=4, hidden_size=8, max_iter=2, batch_size=4
    )
    return LSTMGenerator(**{**defaults, **kwargs})


# ── _LSTMClassificationModel ───────────────────────────────────────────────────


class TestLSTMClassificationModel:
    def test_output_shape(self):
        model = _LSTMClassificationModel(
            input_size=3,
            hidden_size=8,
            output_size=4,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
        )
        X = torch.randn(10, 5, 3)
        output = model(X)
        assert output.shape == (10, 4)

    def test_bidirectional_output_shape(self):
        model = _LSTMClassificationModel(
            input_size=3,
            hidden_size=8,
            output_size=4,
            num_layers=1,
            dropout=0.0,
            bidirectional=True,
        )
        X = torch.randn(10, 5, 3)
        output = model(X)
        assert output.shape == (10, 4)


# ── BaseLSTM validation ────────────────────────────────────────────────────────


class TestBaseLSTMValidation:
    def test_invalid_hidden_size_raises(self):
        with pytest.raises(ValueError, match="hidden_size"):
            LSTMClassifier(hidden_size=0)

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError, match="num_layers"):
            LSTMClassifier(num_layers=0)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            LSTMClassifier(dropout=1.5)

    def test_invalid_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            LSTMClassifier(solver="rmsprop")

    def test_invalid_input_shape_raises(self):
        X = np.zeros((20, 3))
        y = np.zeros(20)
        with pytest.raises(ValueError, match="3D array"):
            _make_clf(max_iter=1).fit(X, y)

    def test_predict_before_fit_raises(self):
        X = np.zeros((2, 5, 3))
        with pytest.raises(RuntimeError, match="not fitted yet"):
            LSTMClassifier().predict(X)


# ── LSTMClassifier ─────────────────────────────────────────────────────────────


class TestLSTMClassifierBinary:
    def test_fit_returns_self(self):
        X, y = _make_classification_data()
        clf = _make_clf()
        assert clf.fit(X, y) is clf

    def test_predict_shape_and_values(self):
        X, y = _make_classification_data()
        clf = _make_clf().fit(X, y)
        predictions = clf.predict(X)
        assert predictions.shape == (len(X),)
        assert set(np.unique(predictions)).issubset(set(clf.classes_))

    def test_predict_proba_shape(self):
        X, y = _make_classification_data()
        clf = _make_clf().fit(X, y)
        probabilities = clf.predict_proba(X)
        assert probabilities.shape == (len(X), 2)
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(len(X)),
            atol=1e-5,
        )

    def test_string_labels_round_trip(self):
        X, y = _make_classification_data()
        y_string = np.array(["negative", "positive"])[y]
        clf = _make_clf().fit(X, y_string)
        assert set(clf.predict(X)).issubset({"negative", "positive"})


# ── LSTMClassifier (multiclass) ────────────────────────────────────────────────


class TestLSTMClassifierMulticlass:
    def test_predict_and_proba_shapes(self):
        X, y = _make_classification_data(n_samples=60, n_classes=3)
        clf = _make_clf().fit(X, y)
        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)
        assert predictions.shape == (len(X),)
        assert probabilities.shape == (len(X), 3)

        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(len(X)),
            atol=1e-5,
        )

    def test_single_class_raises(self):
        X, _ = _make_classification_data()
        y = np.zeros(len(X))
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            _make_clf(max_iter=1).fit(X, y)


# ── _LSTMGenerationModel ───────────────────────────────────────────────────────


class TestLSTMGenerationModel:
    def test_output_shape(self):
        model = _LSTMGenerationModel(
            vocab_size=10, embedding_dim=4, hidden_size=8, num_layers=1, dropout=0.0
        )
        X = torch.randint(0, 10, (3, 5))
        output = model(X)
        assert output.shape == (15, 10)

    def test_generation_loader_flattens_targets(self):
        X = torch.randint(0, 10, (4, 5))
        y = torch.randint(0, 10, (4, 5))
        loader = _make_loader(X, y, batch_size=2, shuffle=False, flatten_targets=True)
        X_batch, y_batch = next(iter(loader))
        assert X_batch.shape == (2, 5)
        assert y_batch.shape == (10,)


# ── LSTMGenerator validation ───────────────────────────────────────────────────


class TestLSTMGeneratorValidation:
    def test_invalid_vocab_size_raises(self):
        with pytest.raises(ValueError, match="vocab_size"):
            LSTMGenerator(vocab_size=1)

    def test_invalid_embedding_dim_raises(self):
        with pytest.raises(ValueError, match="embedding_dim"):
            LSTMGenerator(vocab_size=10, embedding_dim=0)

    def test_invalid_input_dimensions_raises(self):
        X = np.zeros((10, 5, 2))
        y = np.zeros((10, 5))
        with pytest.raises(ValueError, match="2D arrays"):
            LSTMGenerator(vocab_size=10).fit(X, y)

    def test_mismatched_shapes_raise(self):
        X = np.zeros((10, 5))
        y = np.zeros((10, 4))
        with pytest.raises(ValueError, match="same shape"):
            LSTMGenerator(vocab_size=10).fit(X, y)

    def test_invalid_token_ids_raise(self):
        X = np.array([[0, 1, 10]])
        y = np.array([[1, 2, 3]])

        with pytest.raises(ValueError, match="outside"):
            LSTMGenerator(vocab_size=10).fit(X, y)

    def test_generate_before_fit_raises(self):
        generator = LSTMGenerator(vocab_size=10)
        with pytest.raises(RuntimeError, match="not fitted yet"):
            generator.generate([0, 1], length=3)


# ── LSTMGenerator ──────────────────────────────────────────────────────────────


class TestLSTMGenerator:
    @staticmethod
    def _make_data(n_samples=20, seq_len=5, vocab_size=8):
        X = np.random.default_rng(0).integers(0, vocab_size, size=(n_samples, seq_len))
        y = np.roll(X, shift=-1, axis=1)
        return X, y

    def test_fit_returns_self(self):
        X, y = self._make_data()
        generator = _make_gen()
        assert generator.fit(X, y) is generator

    def test_generate_returns_seed_and_new_tokens(self):
        X, y = self._make_data()
        generator = _make_gen().fit(X, y)
        seed = [0, 1, 2]
        generated = generator.generate(seed, length=4)
        assert generated.shape == (7,)
        np.testing.assert_array_equal(generated[:3], seed)

    def test_generated_tokens_are_in_vocabulary(self):
        X, y = self._make_data()
        generator = _make_gen().fit(X, y)
        generated = generator.generate([0, 1], length=5)
        assert np.all(generated >= 0)
        assert np.all(generated < 8)

    def test_zero_length_returns_seed(self):
        X, y = self._make_data()
        generator = _make_gen(max_iter=1).fit(X, y)
        seed = [1, 2, 3]
        generated = generator.generate(seed, length=0)
        np.testing.assert_array_equal(generated, seed)
