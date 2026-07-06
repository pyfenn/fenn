"""Tests for fenn/nn/trainers/trainer.py"""

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


def _make_classification_trainer(num_classes=2, multi_label=False, **kwargs):
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


# ── _move_to_device ────────────────────────────────────────────────────────────


class TestMoveToDevice:
    def test_tensor(self):
        trainer = _make_classification_trainer()
        t = torch.tensor([1.0, 2.0])
        assert torch.is_tensor(trainer._move_to_device(t, "cpu"))

    def test_list_of_tensors(self):
        trainer = _make_classification_trainer()
        result = trainer._move_to_device(
            [torch.tensor([1.0]), torch.tensor([2.0])], "cpu"
        )
        assert isinstance(result, list)
        assert all(torch.is_tensor(t) for t in result)

    def test_tuple_of_tensors(self):
        trainer = _make_classification_trainer()
        result = trainer._move_to_device(
            (torch.tensor([1.0]), torch.tensor([2.0])), "cpu"
        )
        assert isinstance(result, tuple)

    def test_dict_of_tensors(self):
        trainer = _make_classification_trainer()
        result = trainer._move_to_device(
            {"x": torch.tensor([1.0]), "y": torch.tensor([2.0])}, "cpu"
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"x", "y"}

    def test_passthrough_non_tensor(self):
        trainer = _make_classification_trainer()
        assert trainer._move_to_device("hello", "cpu") == "hello"
        assert trainer._move_to_device(42, "cpu") == 42


# ── _should_save_checkpoint ────────────────────────────────────────────────────


class TestShouldSaveCheckpoint:
    def test_no_checkpoint_returns_false(self):
        trainer = _make_classification_trainer()
        trainer._checkpoint = None
        assert trainer._should_save_checkpoint(1) is False

    def test_last_epoch_always_saves(self):
        trainer = _make_classification_trainer()
        trainer._checkpoint = MagicMock()
        trainer._checkpoint.epochs = 999
        assert trainer._should_save_checkpoint(5, is_last_epoch=True) is True

    def test_epoch_list_saves_on_match(self):
        trainer = _make_classification_trainer()
        trainer._checkpoint = MagicMock()
        trainer._checkpoint.epochs = [2, 4, 6]
        assert trainer._should_save_checkpoint(4) is True
        assert trainer._should_save_checkpoint(3) is False

    def test_epoch_interval_saves_on_multiple(self):
        trainer = _make_classification_trainer()
        trainer._checkpoint = MagicMock()
        trainer._checkpoint.epochs = 3
        assert trainer._should_save_checkpoint(3) is True
        assert trainer._should_save_checkpoint(6) is True
        assert trainer._should_save_checkpoint(4) is False


# ── load_checkpoint / load_best / load_at_epoch ────────────────────────────────


class TestLoadCheckpoint:
    def test_raises_without_config(self):
        trainer = _make_classification_trainer()
        trainer._checkpoint = None
        with pytest.raises(ValueError, match="checkpoint_config is missing"):
            trainer.load_checkpoint("path.pt")

    def test_load_checkpoint_at_epoch_raises_without_config(self):
        trainer = _make_classification_trainer()
        trainer._checkpoint = None
        with pytest.raises(ValueError, match="checkpoint_config is missing"):
            trainer.load_checkpoint_at_epoch(3)

    def test_load_best_checkpoint_raises_without_config(self):
        trainer = _make_classification_trainer()
        trainer._checkpoint = None
        with pytest.raises(ValueError, match="checkpoint_config is missing"):
            trainer.load_best_checkpoint()

    def test_load_checkpoint_delegates_to_ckpt(self):
        trainer = _make_classification_trainer()
        mock_ckpt = MagicMock()
        mock_state = MagicMock()
        mock_state.model_state_dict = None
        mock_state.optimizer_state_dict = None
        mock_ckpt.load.return_value = mock_state
        trainer._checkpoint = mock_ckpt
        trainer.load_checkpoint("path/to/ckpt.pt")
        mock_ckpt.load.assert_called_once_with("path/to/ckpt.pt")

    def test_load_checkpoint_at_epoch_delegates(self):
        trainer = _make_classification_trainer()
        mock_ckpt = MagicMock()
        mock_state = MagicMock()
        mock_state.model_state_dict = None
        mock_state.optimizer_state_dict = None
        mock_ckpt.load_at_epoch.return_value = mock_state
        trainer._checkpoint = mock_ckpt
        trainer.load_checkpoint_at_epoch(5)
        mock_ckpt.load_at_epoch.assert_called_once_with(5)

    def test_load_best_checkpoint_delegates(self):
        trainer = _make_classification_trainer()
        mock_ckpt = MagicMock()
        mock_state = MagicMock()
        mock_state.model_state_dict = None
        mock_state.optimizer_state_dict = None
        mock_ckpt.load_best.return_value = mock_state
        trainer._checkpoint = mock_ckpt
        trainer.load_best_checkpoint()
        mock_ckpt.load_best.assert_called_once()


# ── _replace_state ─────────────────────────────────────────────────────────────


class TestReplaceState:
    def test_sets_state(self):
        trainer = _make_classification_trainer()
        mock_state = MagicMock()
        mock_state.model_state_dict = trainer._model.state_dict()
        mock_state.optimizer_state_dict = None
        trainer._replace_state(mock_state)
        assert trainer._state is mock_state

    def test_loads_optimizer_state(self):
        trainer = _make_classification_trainer()
        mock_state = MagicMock()
        mock_state.model_state_dict = None
        mock_state.optimizer_state_dict = {"lr": 0.01}
        trainer._replace_state(mock_state)
        trainer._optimizer.load_state_dict.assert_called_once_with({"lr": 0.01})


class TestSaveModel:
    def test_calls_torch_save(self, tmp_path):
        trainer = _make_classification_trainer()
        trainer._exporter = MagicMock()
        trainer._exporter.export_dir = tmp_path
        with patch("fenn.nn.trainers.trainer.torch.save") as mock_save:
            trainer.save_model("my_model.pth")
        mock_save.assert_called_once()
