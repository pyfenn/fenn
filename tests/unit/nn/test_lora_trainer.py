"""Tests for fenn/nn/trainers/lora_trainer.py"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from fenn.nn.trainers.lora_trainer import LoRATrainer

# ── Helpers ────────────────────────────────────────────────────────────────────


def _linear_model(in_features=4, out_features=2):
    return nn.Linear(in_features, out_features)


def _make_fake_peft_module(mock_peft_model):
    """Build a fake peft module so the try/except import in lora_trainer succeeds."""
    fake_peft = MagicMock()
    fake_peft.get_peft_model = MagicMock(return_value=mock_peft_model)
    fake_peft.LoraConfig = MagicMock()
    fake_peft.TaskType = MagicMock()
    return fake_peft


def _make_trainer(task_type="SEQ_CLS", **kwargs):
    model = _linear_model()
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    mock_peft_model = MagicMock()
    mock_peft_model.parameters.return_value = list(model.parameters())
    mock_peft_model.to.return_value = mock_peft_model
    mock_peft_model.state_dict.return_value = {}

    fake_peft = _make_fake_peft_module(mock_peft_model)

    with patch.dict("sys.modules", {"peft": fake_peft}):
        # Reload the module-level PEFT_AVAILABLE flag and imports
        import fenn.nn.trainers.lora_trainer as lora_mod

        with patch.object(lora_mod, "PEFT_AVAILABLE", True):
            with patch.object(lora_mod, "LoraConfig", fake_peft.LoraConfig):
                with patch.object(lora_mod, "TaskType", fake_peft.TaskType):
                    with patch.object(
                        lora_mod, "get_peft_model", fake_peft.get_peft_model
                    ):
                        with patch("fenn.nn.trainers.trainer.ModelPrettyPrinter"):
                            with patch("fenn.nn.trainers.trainer.logger"):
                                with patch("fenn.nn.trainers.lora_trainer.logger"):
                                    return LoRATrainer(
                                        model=model,
                                        optim=optim,
                                        task_type=task_type,
                                        **kwargs,
                                    )


def _fit(trainer, loader, epochs=1, val_loader=None):
    with patch("fenn.nn.trainers.lora_trainer.Progress") as MockProgress:
        mock_p = MagicMock()
        mock_p.add_task.return_value = MagicMock()
        MockProgress.return_value = mock_p
        with patch("fenn.nn.trainers.lora_trainer.logger"):
            trainer.fit(loader, epochs=epochs, val_loader=val_loader)


# ── __init__ ───────────────────────────────────────────────────────────────────


class TestLoRATrainerInit:
    def test_raises_without_peft(self):
        model = _linear_model()
        optim = torch.optim.SGD(model.parameters(), lr=1e-3)
        with patch("fenn.nn.trainers.lora_trainer.PEFT_AVAILABLE", False):
            with pytest.raises(ImportError, match="peft is not installed"):
                LoRATrainer(model=model, optim=optim)

    def test_raises_on_invalid_task_type(self):
        model = _linear_model()
        optim = torch.optim.SGD(model.parameters(), lr=1e-3)
        with patch("fenn.nn.trainers.lora_trainer.PEFT_AVAILABLE", True):
            with pytest.raises(ValueError, match="Unsupported task_type"):
                LoRATrainer(model=model, optim=optim, task_type="INVALID")

    def test_task_type_stored_uppercase(self):
        assert _make_trainer(task_type="seq_cls")._task_type == "SEQ_CLS"

    def test_is_generative_causal_lm(self):
        assert _make_trainer(task_type="CAUSAL_LM")._is_generative is True

    def test_is_generative_seq2seq(self):
        assert _make_trainer(task_type="SEQ_2_SEQ_LM")._is_generative is True

    def test_is_not_generative_seq_cls(self):
        assert _make_trainer(task_type="SEQ_CLS")._is_generative is False

    def test_is_not_generative_token_cls(self):
        assert _make_trainer(task_type="TOKEN_CLS")._is_generative is False

    def test_is_not_generative_question_ans(self):
        assert _make_trainer(task_type="QUESTION_ANS")._is_generative is False


# ── _forward ───────────────────────────────────────────────────────────────────


class TestLoRATrainerForward:
    def test_returns_loss_and_logits_from_model(self):
        trainer = _make_trainer()
        loss = torch.tensor(0.5, requires_grad=True)
        logits = torch.randn(2, 3)
        trainer._model = MagicMock(
            return_value=SimpleNamespace(loss=loss, logits=logits)
        )
        returned_loss, returned_logits = trainer._forward(
            {"input_ids": torch.ones(2, 4).long()}
        )
        assert returned_loss is loss
        assert returned_logits is logits

    def test_falls_back_to_loss_fn_when_model_has_no_loss(self):
        trainer = _make_trainer()
        logits = torch.randn(2, 3)
        labels = torch.randint(0, 3, (2,))
        trainer._model = MagicMock(
            return_value=SimpleNamespace(loss=None, logits=logits)
        )
        mock_loss = torch.tensor(0.3, requires_grad=True)
        trainer._loss_fn = MagicMock(return_value=mock_loss)
        returned_loss, _ = trainer._forward(
            {"input_ids": torch.ones(2).long(), "labels": labels}
        )
        assert returned_loss is mock_loss

    def test_returns_none_loss_when_no_loss_fn_and_no_model_loss(self):
        trainer = _make_trainer()
        trainer._model = MagicMock(
            return_value=SimpleNamespace(loss=None, logits=torch.randn(2, 3))
        )
        trainer._loss_fn = None
        loss, _ = trainer._forward({"input_ids": torch.ones(2).long()})
        assert loss is None


# ── predict ────────────────────────────────────────────────────────────────────


class TestLoRATrainerPredict:
    def test_classification_argmax(self):
        trainer = _make_trainer(task_type="SEQ_CLS")
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        trainer._model = MagicMock(return_value=SimpleNamespace(logits=logits))
        assert trainer.predict({"input_ids": torch.ones(2, 4).long()}) == [1, 0]

    def test_generative_returns_logits(self):
        trainer = _make_trainer(task_type="CAUSAL_LM")
        logits = torch.randn(2, 5, 10)
        trainer._model = MagicMock(return_value=SimpleNamespace(logits=logits))
        preds = trainer.predict({"input_ids": torch.ones(2, 5).long()})
        assert len(preds) == 1
        assert preds[0].shape == logits.shape

    def test_strips_labels_from_dict_batch(self):
        trainer = _make_trainer(task_type="SEQ_CLS")
        logits = torch.tensor([[0.1, 0.9]])
        trainer._model = MagicMock(return_value=SimpleNamespace(logits=logits))
        batch = {"input_ids": torch.ones(1, 4).long(), "labels": torch.tensor([1])}
        trainer.predict(batch)
        assert "labels" not in trainer._model.call_args[1]

    def test_predict_from_raw_tensor(self):
        trainer = _make_trainer(task_type="SEQ_CLS")
        logits = torch.tensor([[0.2, 0.8]])
        trainer._model = MagicMock(return_value=logits)
        preds = trainer.predict(torch.ones(1, 4).long())
        assert preds == [1]


# ── fit ────────────────────────────────────────────────────────────────────────


class TestLoRATrainerFit:
    def test_raises_when_no_loss(self):
        trainer = _make_trainer()
        trainer._model = MagicMock(return_value=SimpleNamespace(loss=None, logits=None))
        trainer._loss_fn = None
        with pytest.raises(ValueError, match="No loss was returned"):
            _fit(trainer, [{"input_ids": torch.ones(2, 4).long()}], epochs=1)

    def test_empty_loader_raises(self):
        trainer = _make_trainer()
        with pytest.raises(ValueError, match="0 batches"):
            _fit(trainer, [], epochs=1)

    def test_runs_with_loss_in_outputs(self):
        trainer = _make_trainer()
        loss_val = torch.tensor(0.4, requires_grad=True)
        trainer._model = MagicMock(
            return_value=SimpleNamespace(loss=loss_val, logits=torch.randn(2, 2))
        )
        loader = [
            {"input_ids": torch.ones(2, 4).long(), "labels": torch.tensor([0, 1])}
        ]
        _fit(trainer, loader, epochs=1)
        assert trainer._state.epoch == 1

    def test_early_stopping_triggers(self):
        trainer = _make_trainer(early_stopping_patience=1)
        # loss never improves (constant mock)
        loss_val = torch.tensor(0.5, requires_grad=True)
        trainer._model = MagicMock(
            return_value=SimpleNamespace(loss=loss_val, logits=torch.randn(2, 2))
        )
        loader = [
            {"input_ids": torch.ones(2, 4).long(), "labels": torch.tensor([0, 1])}
        ]
        _fit(trainer, loader, epochs=10)
        assert trainer._state.epoch < 10

    def test_empty_val_loader_raises(self):
        trainer = _make_trainer()
        loss_val = torch.tensor(0.4, requires_grad=True)
        trainer._model = MagicMock(
            return_value=SimpleNamespace(loss=loss_val, logits=torch.randn(2, 2))
        )
        loader = [
            {"input_ids": torch.ones(2, 4).long(), "labels": torch.tensor([0, 1])}
        ]
        with pytest.raises(ValueError, match="val_loader produced 0 batches"):
            _fit(trainer, loader, epochs=1, val_loader=[])

    def test_fit_with_validation(self):
        trainer = _make_trainer()
        loss_val = torch.tensor(0.4, requires_grad=True)
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        trainer._model = MagicMock(
            return_value=SimpleNamespace(loss=loss_val, logits=logits)
        )
        loader = [
            {"input_ids": torch.ones(2, 4).long(), "labels": torch.tensor([0, 1])}
        ]
        _fit(trainer, loader, epochs=1, val_loader=loader)
        assert trainer._state.val_loss is not None
