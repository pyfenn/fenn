from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

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

from fenn.logging import Logger
from fenn.nn.utils import Checkpoint, ModelPrettyPrinter, TrainingState


class Trainer(ABC):
    """The base Trainer abstract class for classification/Regression tasks."""

    @abstractmethod
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optim: torch.optim.Optimizer,
        device: Union[torch.device, str] = "cpu",
        early_stopping_patience: Optional[int] = None,
        checkpoint_config: Optional[Checkpoint] = None,
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

        self._logger = Logger()

        self._loss_fn = loss_fn
        self._num_classes = 2
        self._device = torch.device(device)

        # training state at epoch 0
        self._model = model.to(device)
        self._optimizer = optim
        self._state = TrainingState(epoch=0)
        self._log_model_summary()

        self._best_state: Optional[TrainingState] = None
        """Best training state based on validation loss."""

        self._best_model: Optional[torch.nn.Module] = None

        # checkpoint setup
        self._checkpoint = checkpoint_config
        if self._checkpoint is not None:
            self._checkpoint._setup()

        # early stopping setup
        self._early_stopping_patience = early_stopping_patience
        if self._early_stopping_patience is not None:
            self._logger.display_info(
                f"Early stopping enabled with patience of {self._early_stopping_patience} epochs."
            )

    def _log_model_summary(self) -> None:
        summary = ModelPrettyPrinter(self._model).render()
        self._logger.display_info(summary, display_on_terminal=False)

    def _move_to_device(self, batch, device: Union[torch.device, str]) -> Any:
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(x, device) for x in batch)
        if isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        return batch

    def _should_save_checkpoint(self, epoch: int, is_last_epoch: bool = False) -> bool:
        """Check if a checkpoint should be saved at the given epoch.

        Args:
            epoch: The current epoch number. (1-indexed)
            is_last_epoch: Whether this is the last epoch.

        Returns:
            True if a checkpoint should be saved, False otherwise.
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
        val_loader: Optional[DataLoader] = None,
        val_epochs: int = 1,
    ):
        """Train the model with optional validation and early stopping.

        The behaviour depends on the combination of `val_loader` and
        ``early_stopping_patience``:

        * No validation loader, no early stopping: run full ``epochs``.
        * No validation loader, early stopping set: stop on training loss.
        * Validation loader provided, no early stopping: evaluate every epoch
          but continue regardless of metrics.
        * Validation loader provided and early stopping set: monitor
          validation loss and stop when it plateaus.

        Args:
            train_loader: DataLoader for training data.
            epochs: Total number of epochs to train for.
            val_loader: DataLoader for validation data (optional).
            val_epochs: How often to evaluate on validation set (in epochs).

        Returns:
            The trained model (returned according to ``return_model``).
        """
        pass

    def _replace_state(self, new_state: TrainingState) -> None:
        """Replace the current training state with a new state.

        Args:
            new_state: The new state to replace the current state with.
        """
        self._state = new_state
        if new_state.model_state_dict:
            self._model.load_state_dict(new_state.model_state_dict)
        if new_state.optimizer_state_dict:
            self._optimizer.load_state_dict(new_state.optimizer_state_dict)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load a checkpoint from a given path to update current training state.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        if self._checkpoint is None:
            raise ValueError("Cannot load checkpoint: checkpoint_config is missing.")

        new_state = self._checkpoint.load(checkpoint_path)
        self._replace_state(new_state)

    def load_checkpoint_at_epoch(self, epoch: int) -> None:
        """Load the checkpoint at the given epoch to update current training state.

        Args:
            epoch: Epoch to load the checkpoint at.
        """
        if self._checkpoint is None:
            raise ValueError("Cannot load checkpoint: checkpoint_config is missing.")

        new_state = self._checkpoint.load_at_epoch(epoch)
        self._replace_state(new_state)

    def load_best_checkpoint(self) -> None:
        """Load the best checkpoint to update current training state."""
        if self._checkpoint is None:
            raise ValueError("Cannot load checkpoint: checkpoint_config is missing.")

        new_state = self._checkpoint.load_best()
        self._replace_state(new_state)

    @abstractmethod
    def predict(self, dataloader_or_batch: Union[DataLoader, torch.Tensor]):
        """Predicts the output of the model for a given dataloader or batch.

        Args:
            dataloader_or_batch: A DataLoader or a torch tensor.

        Returns:
            list: A list of predictions.
        """
        pass
