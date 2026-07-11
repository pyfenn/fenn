"""Tests for fenn/nn/trainers/regression_trainer.py"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from fenn.nn.trainers.regression_trainer import RegressionTrainer

# ── Helpers ────────────────────────────────────────────────────────────────────


def _linear_model(in_features=4, out_features=1):
    return nn.Linear(in_features, out_features)


def _mock_loss_fn(return_value=0.5):
    loss = MagicMock()
    loss.item.return_value = return_value
    loss.backward = MagicMock()
    return MagicMock(return_value=loss)


def _mock_optimizer():
    optim = MagicMock()
    optim.defaults = {"lr": 1e-3}
    return optim


def _make_dataloader(n_batches=2, n_samples=4, in_features=4):
    batches = []
    for _ in range(n_batches):
        data = torch.randn(n_samples, in_features)
        labels = torch.randn(n_samples, 1)
        batches.append((data, labels))
    return batches


def _make_trainer(**kwargs):
    model = _linear_model()
    loss_fn = _mock_loss_fn()
    optim = _mock_optimizer()
    with patch("fenn.nn.trainers.trainer.ModelPrettyPrinter"):
        with patch("fenn.nn.trainers.trainer.logger"):
            with patch("fenn.nn.trainers.regression_trainer.logger"):
                return RegressionTrainer(
                    model=model,
                    loss_fn=loss_fn,
                    optim=optim,
                    **kwargs,
                )


def _fit(trainer, loader, epochs=1, val_loader=None):
    with patch("fenn.nn.trainers.regression_trainer.Progress") as MockProgress:
        mock_p = MagicMock()
        mock_p.add_task.return_value = MagicMock()
        MockProgress.return_value = mock_p
        with patch("fenn.nn.trainers.regression_trainer.logger"):
            return trainer.fit(loader, epochs=epochs, val_loader=val_loader)


# ── __init__ ───────────────────────────────────────────────────────────────────


class TestRegressionTrainerInit:
    def test_default_return_model_is_last(self):
        assert _make_trainer()._return_model == "last"

    def test_return_model_best(self):
        assert _make_trainer(return_model="best")._return_model == "best"

    def test_return_model_case_insensitive(self):
        assert _make_trainer(return_model="BEST")._return_model == "best"

    def test_invalid_return_model_raises(self):
        with pytest.raises(ValueError, match="return_model must be 'last' or 'best'"):
            _make_trainer(return_model="invalid")


# ── predict ────────────────────────────────────────────────────────────────────


class TestRegressionTrainerPredict:
    def test_predict_from_tensor(self):
        trainer = _make_trainer()
        trainer._model = MagicMock(return_value=torch.tensor([[1.5], [2.5]]))
        preds = trainer.predict(torch.randn(2, 4))
        assert len(preds) == 2
        assert abs(preds[0] - 1.5) < 1e-4


# ── fit ────────────────────────────────────────────────────────────────────────


class TestRegressionTrainerFit:
    def test_runs_without_validation(self):
        trainer = _make_trainer()
        _fit(trainer, _make_dataloader(), epochs=1)
        assert trainer._state.epoch == 1

    def test_returns_last_model(self):
        trainer = _make_trainer(return_model="last")
        result = _fit(trainer, _make_dataloader(), epochs=1)
        assert result is trainer._model

    def test_returns_best_model_when_configured(self):
        trainer = _make_trainer(return_model="best")
        result = _fit(
            trainer, _make_dataloader(), epochs=1, val_loader=_make_dataloader()
        )
        assert result is not None

    def test_empty_loader_raises(self):
        trainer = _make_trainer()
        with pytest.raises(ValueError, match="0 batches"):
            _fit(trainer, [], epochs=1)

    def test_empty_val_loader_raises(self):
        trainer = _make_trainer()
        with pytest.raises(ValueError, match="val_loader produced 0 batches"):
            _fit(trainer, _make_dataloader(), epochs=1, val_loader=[])

    def test_with_validation(self):
        trainer = _make_trainer()
        _fit(trainer, _make_dataloader(), epochs=1, val_loader=_make_dataloader())
        assert trainer._state.val_loss is not None

    def test_early_stopping_triggers(self):
        trainer = _make_trainer(early_stopping_patience=1)
        _fit(trainer, _make_dataloader(n_batches=1), epochs=10)
        assert trainer._state.epoch < 10

    def test_multiple_epochs(self):
        trainer = _make_trainer()
        _fit(trainer, _make_dataloader(), epochs=3)
        assert trainer._state.epoch == 3

    def test_val_epochs_interval(self):
        trainer = _make_trainer()
        _fit(trainer, _make_dataloader(), epochs=4, val_loader=_make_dataloader())
        assert trainer._state.val_loss is not None
