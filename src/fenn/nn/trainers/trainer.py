from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, cast

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
from fenn.nn.utils import Checkpoint, TrainingState

if TYPE_CHECKING:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )

try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class Trainer:
    """The base Trainer class for classification tasks."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optim: torch.optim.Optimizer,
        num_classes: int,
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
        self._num_classes = num_classes
        self._device = torch.device(device)

        # training state at epoch 0
        self._model = model.to(device)
        self._optimizer = optim
        self._state = TrainingState(epoch=0)

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
            self._logger.system_info(
                f"Early stopping enabled with patience of {self._early_stopping_patience} epochs."
            )

    def _move_to_device(self, batch, device: Union[torch.device, str]):
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(x, device) for x in batch)
        if isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        return batch

    def _should_save_checkpoint(self, epoch: int, is_last_epoch: bool = False):
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

        state = self._state

        if HAS_RICH:
            progress = Progress(
                TextColumn(
                    "[bold blue]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"
                ),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("{task.fields[info]}"),
            )
            progress.start()
            epoch_task = progress.add_task(
                "Training",
                total=epochs - state.epoch,
                epoch=state.epoch,
                total_epochs=epochs,
                info="",
            )

        else:
            progress = None
            epoch_task = None

        for epoch in range(state.epoch + 1, epochs + 1):
            state.epoch = epoch
            state.model_state_dict = None
            state.optimizer_state_dict = None

            # --- TRAIN ---
            self._model.train()
            total_loss = 0.0
            n_batches = 0

            for data, labels in train_loader:
                data = self._move_to_device(data, self._device)
                labels = labels.to(self._device)

                outputs = self._model(data)
                loss = self._loss_fn(outputs, labels)

                self._optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            if n_batches == 0:
                raise ValueError("train_loader produced 0 batches; cannot train.")

            state.train_loss = total_loss / n_batches
            if progress:
                progress.update(
                    epoch_task,  # pyright: ignore[reportArgumentType]
                    advance=1,
                    epoch=epoch,
                    info=f"Train Mean Loss : {state.train_loss:.4f}",
                )
            else:
                print(f"Epoch {epoch}. Train Mean Loss: {state.train_loss:.4f}")

            # --- NO VALIDATION ---
            if val_loader is None:
                state.val_loss = None

                if state.train_loss < state.best_train_loss:
                    state.best_train_loss = state.train_loss
                    state.patience_counter = 0
                else:
                    state.patience_counter += 1

            # --- VALIDATION ---
            elif epoch % val_epochs == 0 or epoch == epochs:
                self._model.eval()
                val_labels = []
                val_predictions = []
                val_total_loss = 0.0
                val_n_batches = 0

                with torch.no_grad():
                    for data, labels in val_loader:
                        data = self._move_to_device(data, self._device)
                        labels = labels.to(self._device)

                        outputs = self._model(data)
                        val_batch_loss = self._loss_fn(outputs, labels)

                        val_total_loss += float(val_batch_loss.item())
                        val_n_batches += 1

                        if self._num_classes == 2:
                            logits = outputs
                            preds = torch.sigmoid(logits).squeeze(-1)
                            preds = (preds > 0.5).long()
                        else:
                            preds = torch.argmax(outputs, dim=1)

                        val_predictions.extend(preds.cpu().tolist())
                        val_labels.extend(labels.cpu().tolist())

                if val_n_batches == 0:
                    raise ValueError("val_loader produced 0 batches; cannot validate.")

                state.val_loss = val_total_loss / val_n_batches
                state.acc = accuracy_score(val_labels, val_predictions)

                if progress:
                    progress.update(
                        epoch_task,  # pyright: ignore[reportArgumentType]
                        info=f"Train Mean Loss: {state.train_loss:.4f} | Val Loss: {state.val_loss:.4f} | Val Acc: {state.acc:.4f}",
                    )
                else:
                    print(f"Epoch {epoch}. Validation Loss: {state.val_loss:.4f}")
                    print(f"Epoch {epoch}. Validation Accuracy: {state.acc:.4f}")

                if state.val_loss < state.best_val_loss:
                    state.best_val_loss = state.val_loss

                    # Update best state for improved val_loss
                    self._best_state = state.clone(
                        model_state_dict=self._model.state_dict(),
                        optimizer_state_dict=self._optimizer.state_dict(),
                    )
                    self._best_model = deepcopy(self._model)
                    self._best_model.load_state_dict(self._best_state.model_state_dict)  # pyright: ignore[reportArgumentType]

                    if self._checkpoint is not None:
                        self._checkpoint.save(self._best_state, is_best=True)

                    state.patience_counter = 0
                else:
                    state.patience_counter += 1

                if state.acc > state.best_acc:
                    state.best_acc = state.acc

            # --- CHECKPOINTING ---
            if self._should_save_checkpoint(epoch, is_last_epoch=(epoch == epochs)):
                state.model_state_dict = self._model.state_dict()
                state.optimizer_state_dict = self._optimizer.state_dict()
                cast(Checkpoint, self._checkpoint).save(state)

            # --- EARLY STOPPING ---
            if (
                self._early_stopping_patience is not None
                and state.patience_counter >= self._early_stopping_patience
            ):
                if val_loader is None:
                    _reason = "training loss"
                else:
                    _reason = "validation loss"
                self._logger.system_info(
                    f"Early stopping triggered. "
                    f"No improvement in {_reason} for {self._early_stopping_patience} epochs."
                )
                break

        if progress:
            progress.stop()

    def _replace_state(self, new_state: TrainingState):
        """Replace the current training state with a new state.

        Args:
            new_state: The new state to replace the current state with.
        """
        self._state = new_state
        if new_state.model_state_dict:
            self._model.load_state_dict(new_state.model_state_dict)
        if new_state.optimizer_state_dict:
            self._optimizer.load_state_dict(new_state.optimizer_state_dict)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load a checkpoint from a given path to update current training state.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        if self._checkpoint is None:
            raise ValueError("Cannot load checkpoint: checkpoint_config is missing.")

        new_state = self._checkpoint.load(checkpoint_path)
        self._replace_state(new_state)

    def load_checkpoint_at_epoch(self, epoch: int):
        """Load the checkpoint at the given epoch to update current training state.

        Args:
            epoch: Epoch to load the checkpoint at.
        """
        if self._checkpoint is None:
            raise ValueError("Cannot load checkpoint: checkpoint_config is missing.")

        new_state = self._checkpoint.load_at_epoch(epoch)
        self._replace_state(new_state)

    def load_best_checkpoint(self):
        """Load the best checkpoint to update current training state."""
        if self._checkpoint is None:
            raise ValueError("Cannot load checkpoint: checkpoint_config is missing.")

        new_state = self._checkpoint.load_best()
        self._replace_state(new_state)

    def predict(self, dataloader_or_batch: Union[DataLoader, torch.Tensor]):
        """Predicts the output of the model for a given dataloader or batch.

        Args:
            dataloader_or_batch: A DataLoader or a torch tensor.

        Returns:
            list: A list of predictions.
        """

        self._model.eval()
        predictions = []

        def predict_batch(batch):
            batch = self._move_to_device(batch, self._device)
            logits = self._model(batch)
            if self._num_classes == 2:
                preds = torch.sigmoid(logits).squeeze(-1)
                preds = (preds > 0.5).long()
            else:
                preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().tolist())

        with torch.no_grad():
            if isinstance(dataloader_or_batch, DataLoader):
                for data, _ in dataloader_or_batch:
                    predict_batch(data)
            else:
                predict_batch(dataloader_or_batch)

        return predictions
