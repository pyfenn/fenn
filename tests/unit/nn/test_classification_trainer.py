"""Tests for fenn/nn/trainers/classification_trainer.py"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from fenn.nn.trainers.classification_trainer import ClassificationTrainer

# ── Helpers ────────────────────────────────────────────────────────────────────


def _linear_model(in_features=4, out_features=2):
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


def _make_dataloader(n_batches=2, n_samples=4, in_features=4, n_classes=2):
    batches = []
    for _ in range(n_batches):
        data = torch.randn(n_samples, in_features)
        labels = torch.randint(0, n_classes, (n_samples,))
        batches.append((data, labels))
    return batches


def _make_trainer(num_classes=2, multi_label=False, **kwargs):
    model = _linear_model(out_features=num_classes if num_classes > 1 else 1)
    loss_fn = _mock_loss_fn()
    optim = _mock_optimizer()
    with patch("fenn.nn.trainers.trainer.ModelPrettyPrinter"):
        with patch("fenn.nn.trainers.trainer.logger"):
            with patch("fenn.nn.trainers.classification_trainer.logger"):
                return ClassificationTrainer(
                    model=model,
                    loss_fn=loss_fn,
                    optim=optim,
                    num_classes=num_classes,
                    multi_label=multi_label,
                    **kwargs,
                )


def _fit(trainer, loader, epochs=1, val_loader=None):
    with patch("fenn.nn.trainers.classification_trainer.Progress") as MockProgress:
        mock_p = MagicMock()
        mock_p.add_task.return_value = MagicMock()
        MockProgress.return_value = mock_p
        with patch("fenn.nn.trainers.classification_trainer.logger"):
            trainer.fit(loader, epochs=epochs, val_loader=val_loader)


# ── __init__ ───────────────────────────────────────────────────────────────────


class TestClassificationTrainerInit:
    def test_binary_task_type(self):
        assert _make_trainer(num_classes=2)._task_type == "binary"

    def test_multiclass_task_type(self):
        assert _make_trainer(num_classes=5)._task_type == "multiclass"

    def test_multilabel_task_type(self):
        assert _make_trainer(num_classes=3, multi_label=True)._task_type == "multilabel"

    def test_num_classes_zero_raises(self):
        with pytest.raises(ValueError, match="num_classes must be >= 1"):
            _make_trainer(num_classes=0)

    def test_multilabel_single_class_raises(self):
        with pytest.raises(
            ValueError, match="Multi-label classification requires num_classes >= 2"
        ):
            _make_trainer(num_classes=1, multi_label=True)

    def test_num_classes_one_valid(self):
        assert _make_trainer(num_classes=1)._num_classes == 1


# ── predict ────────────────────────────────────────────────────────────────────


class TestClassificationTrainerPredict:
    def test_binary_from_tensor(self):
        trainer = _make_trainer(num_classes=2)
        trainer._model = MagicMock(return_value=torch.tensor([[2.0], [-1.0], [3.0]]))
        assert trainer.predict(torch.randn(3, 4)) == [1, 0, 1]

    def test_multiclass_from_tensor(self):
        trainer = _make_trainer(num_classes=3)
        trainer._model = MagicMock(
            return_value=torch.tensor([[0.1, 0.8, 0.1], [0.6, 0.2, 0.2]])
        )
        assert trainer.predict(torch.randn(2, 4)) == [1, 0]

    def test_multilabel_from_tensor(self):
        trainer = _make_trainer(num_classes=3, multi_label=True)
        trainer._model = MagicMock(return_value=torch.tensor([[2.0, -1.0, 3.0]]))
        assert trainer.predict(torch.randn(1, 4)) == [[1, 0, 1]]

    def test_binary_with_proba(self):
        trainer = _make_trainer(num_classes=2)
        trainer._model = MagicMock(return_value=torch.tensor([[5.0], [-5.0]]))
        preds, proba = trainer.predict(torch.randn(2, 4), return_proba=True)
        assert len(preds) == 2
        assert all(0 <= p <= 1 for p in proba)

    def test_multiclass_with_proba(self):
        trainer = _make_trainer(num_classes=3)
        trainer._model = MagicMock(return_value=torch.tensor([[0.1, 0.8, 0.1]]))
        preds, proba = trainer.predict(torch.randn(1, 4), return_proba=True)
        assert preds == [1]
        assert len(proba[0]) == 3

    def test_multilabel_with_proba(self):
        trainer = _make_trainer(num_classes=3, multi_label=True)
        trainer._model = MagicMock(return_value=torch.tensor([[2.0, -1.0, 3.0]]))
        preds, proba = trainer.predict(torch.randn(1, 4), return_proba=True)
        assert preds == [[1, 0, 1]]
        assert len(proba[0]) == 3


# ── fit ────────────────────────────────────────────────────────────────────────


class TestClassificationTrainerFit:
    def test_runs_without_validation(self):
        trainer = _make_trainer(num_classes=3)
        _fit(trainer, _make_dataloader(n_classes=3), epochs=1)
        assert trainer._state.epoch == 1

    def test_updates_train_loss(self):
        trainer = _make_trainer(num_classes=3)
        _fit(trainer, _make_dataloader(n_classes=3), epochs=1)
        assert trainer._state.train_loss is not None

    def test_empty_loader_raises(self):
        trainer = _make_trainer(num_classes=2)
        with pytest.raises(ValueError, match="0 batches"):
            _fit(trainer, [], epochs=1)

    def test_with_validation(self):
        trainer = _make_trainer(num_classes=3)
        _fit(
            trainer,
            _make_dataloader(n_classes=3),
            epochs=1,
            val_loader=_make_dataloader(n_classes=3),
        )
        assert trainer._state.val_loss is not None

    def test_empty_val_loader_raises(self):
        trainer = _make_trainer(num_classes=2)
        with pytest.raises(ValueError, match="val_loader produced 0 batches"):
            _fit(trainer, _make_dataloader(n_classes=2), epochs=1, val_loader=[])

    def test_early_stopping_triggers(self):
        trainer = _make_trainer(num_classes=2, early_stopping_patience=1)
        _fit(trainer, _make_dataloader(n_batches=1, n_classes=2), epochs=10)
        assert trainer._state.epoch < 10

    def test_binary_label_reshaped(self):
        trainer = _make_trainer(num_classes=2)
        trainer._loss_fn = nn.BCEWithLogitsLoss()
        trainer._model = nn.Linear(4, 1)
        _fit(trainer, _make_dataloader(n_batches=1, n_classes=2), epochs=1)
        assert trainer._state.train_loss is not None

    def test_multiple_epochs(self):
        trainer = _make_trainer(num_classes=3)
        _fit(trainer, _make_dataloader(n_classes=3), epochs=3)
        assert trainer._state.epoch == 3

    def test_checkpoint_saved_at_last_epoch(self):
        trainer = _make_trainer(num_classes=2)
        mock_ckpt = MagicMock()
        mock_ckpt.epochs = 999
        trainer._checkpoint = mock_ckpt
        _fit(trainer, _make_dataloader(n_classes=2), epochs=1)
        mock_ckpt.save.assert_called()
