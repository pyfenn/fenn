from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn
import torch.optim
from sklearn.metrics import (  # noqa: F401
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from fenn.exporter import Exporter
from fenn.logging import logger
from fenn.nn.utils import Checkpoint, ModelPrettyPrinter, TrainingState


class Trainer(ABC):
    """The base Trainer abstract class for classification and regression tasks.

    Provides a common training loop with support for early stopping,
    checkpointing, and validation monitoring. Subclasses must implement
    :meth:`fit` to define the per-epoch training logic and :meth:`predict`
    to generate predictions from a model.

    Subclasses:
        - :class:`ClassificationTrainer` for classification tasks.
        - :class:`RegressionTrainer` for regression tasks.
        - :class:`LoRATrainer` for parameter-efficient fine-tuning.
    """

    @abstractmethod
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optim: torch.optim.Optimizer,
        device: torch.device | str = "cpu",
        early_stopping_patience: int | None = None,
        checkpoint_config: Checkpoint | None = None,
    ):
        """Initialize a Trainer instance to fit a neural network model.

        Args:
            model: The neural network model to train.
            loss_fn: The loss function to use.
            optim: The optimizer to use.
            num_classes: The number of classes to predict.
            device: The device on which the data will be loaded.
            early_stopping_patience: The number of epochs to wait before early stopping.
            checkpoint_config: The checkpoint configuration. If `None`, checkpointing is disabled.
        """

        self._exporter = Exporter()

        self._loss_fn = loss_fn
        self._num_classes = 2
        self._device = torch.device(device)

        # training state at epoch 0
        self._model = model.to(device)
        self._optimizer = optim
        self._state = TrainingState(epoch=0)
        self._log_model_summary()

        self._best_state: TrainingState | None = None
        """Best training state based on validation loss."""

        self._best_model: torch.nn.Module | None = None

        # checkpoint setup
        self._checkpoint = checkpoint_config
        if self._checkpoint is not None:
            self._checkpoint._setup()

        # early stopping setup
        self._early_stopping_patience = early_stopping_patience
        if self._early_stopping_patience is not None:
            logger.info(
                f"Early stopping enabled with patience of {self._early_stopping_patience} epochs."
            )

    def _log_model_summary(self) -> None:
        summary = ModelPrettyPrinter(self._model).render()
        logger.info(summary, extra={"skip_console": True})

    def _move_to_device(self, batch: Any, device: torch.device | str) -> Any:
        """Recursively move tensor data to the specified device.

        Handles tensors, lists, tuples, and dictionaries of tensors.
        Non-tensor values are passed through unchanged.

        Args:
            batch: Input data — a tensor, list, tuple, or dict of tensors.
            device: Target device (``torch.device`` or string like ``'cuda'``).

        Returns:
            The input data with all tensors moved to the target device.
        """
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(x, device) for x in batch)
        if isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        return batch

    def _should_save_checkpoint(self, epoch: int, is_last_epoch: bool = False) -> bool:
        """Check whether a checkpoint should be saved at the given epoch.

        Evaluates the checkpoint configuration to determine if the current
        epoch qualifies for a checkpoint save based on fixed intervals,
        explicit epoch lists, or being the final training epoch.

        Args:
            epoch: The current epoch number (1-indexed).
            is_last_epoch: Whether this is the final training epoch.

        Returns:
            ``True`` if a checkpoint should be saved, ``False`` otherwise.
        """

        if self._checkpoint is None:
            return False

        if is_last_epoch:
            # always save at the last epoch
            return True

        elif isinstance(self._checkpoint.epochs, list):
            # save at specific epochs
            return epoch in self._checkpoint.epochs

        elif isinstance(self._checkpoint.epochs, int):
            # save every N epochs
            return epoch % self._checkpoint.epochs == 0

        return False

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: DataLoader | None = None,
        val_epochs: int = 1,
    ):
        """Train the model for a fixed number of epochs.

        Runs the full training loop including forward/backward passes,
        validation evaluation, checkpointing, and early stopping.
        The exact behavior depends on the validation and early stopping
        configuration:

        * No validation loader, no early stopping: run full ``epochs``.
        * No validation loader, early stopping set: stop on training loss.
        * Validation loader provided, no early stopping: evaluate every
          epoch but continue regardless of metrics.
        * Validation loader and early stopping set: monitor validation
          loss and stop when it plateaus for ``early_stopping_patience``
          epochs.

        Args:
            train_loader: PyTorch ``DataLoader`` yielding ``(data, labels)``
                batches for training.
            epochs: Total number of training epochs. If resuming from a
                checkpoint, only the remaining epochs are run.
            val_loader: Optional ``DataLoader`` for validation evaluation.
            val_epochs: How frequently to evaluate on the validation set
                (e.g. ``val_epochs=2`` means every 2 epochs).

        Returns:
            The trained model.
        """
        pass

    def _replace_state(self, new_state: TrainingState) -> None:
        """Replace the current training state with a previously saved state.

        Loads the model weights and optimizer state from the provided
        state, effectively restoring training to a checkpointed point.

        Args:
            new_state: A :class:`~fenn.nn.utils.TrainingState` containing
                the model and optimizer state dicts to restore.
        """
        self._state = new_state
        if new_state.model_state_dict:
            self._model.load_state_dict(new_state.model_state_dict)
        if new_state.optimizer_state_dict:
            self._optimizer.load_state_dict(new_state.optimizer_state_dict)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load a checkpoint from the given file path and restore training state.

        Restores the model weights, optimizer state, and epoch counter from
        a previously saved checkpoint file.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint file.

        Raises:
            ValueError: If no checkpoint configuration was provided at init.
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if self._checkpoint is None:
            raise ValueError("Cannot load checkpoint: checkpoint_config is missing.")

        new_state = self._checkpoint.load(checkpoint_path)
        self._replace_state(new_state)

    def load_checkpoint_at_epoch(self, epoch: int) -> None:
        """Load the checkpoint saved at a specific epoch.

        Searches the checkpoint directory for the saved state at the
        requested epoch and restores the model and optimizer.

        Args:
            epoch: The epoch whose checkpoint to load (1-indexed).

        Raises:
            ValueError: If no checkpoint configuration was provided at init.
            FileNotFoundError: If no checkpoint exists for the given epoch.
        """
        if self._checkpoint is None:
            raise ValueError("Cannot load checkpoint: checkpoint_config is missing.")

        new_state = self._checkpoint.load_at_epoch(epoch)
        self._replace_state(new_state)

    def load_best_checkpoint(self) -> None:
        """Load the best-performing checkpoint into the trainer's model.

        Restores the model weights from the checkpoint with the lowest
        validation (or training) loss recorded during training.

        Raises:
            ValueError: If no checkpoint configuration was provided at init.
            FileNotFoundError: If no best checkpoint file exists.
        """
        if self._checkpoint is None:
            raise ValueError("Cannot load checkpoint: checkpoint_config is missing.")

        new_state = self._checkpoint.load_best()
        self._replace_state(new_state)

    def save_model(self, model_name: str = "model.pth"):
        torch.save(self._model.state_dict(), (self._exporter.export_dir / model_name))

    @abstractmethod
    def predict(self, dataloader_or_batch: DataLoader | torch.Tensor):
        """Generate predictions from the trained model.

        Runs inference on the provided data without computing gradients,
        returning model predictions in the same format as the training
        labels.

        Args:
            dataloader_or_batch: Either a PyTorch ``DataLoader`` yielding
                data batches, or a single tensor batch.

        Returns:
            A list of predictions (one per sample).
        """
        pass
