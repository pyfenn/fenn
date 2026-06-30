from copy import deepcopy
from typing import cast

import torch
import torch.nn
import torch.optim
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from sklearn.metrics import (  # noqa: F401
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from fenn.logging import logger
from fenn.nn.utils import Checkpoint

from .trainer import Trainer


class ClassificationTrainer(Trainer):
    """A trainer for classification tasks with PyTorch models.

    Supports binary, multi-class, and multi-label classification by adapting
    the loss computation and prediction logic based on the task type. Handles
    both single-label (``num_classes == 2`` → binary, ``> 2`` → multiclass) and
    multi-label (``multi_label=True``) scenarios.

    The automatic task type detection configures:
    - Binary: sigmoid activation, BCE loss, threshold at 0.5
    - Multiclass: softmax activation, cross-entropy loss
    - Multi-label: sigmoid activation, binary cross-entropy per label

    Args:
        model: The neural network model, expected to output logits for
            the classification task.
        loss_fn: Loss function compatible with the task type (e.g. BCEWithLogitsLoss
            for binary/multi-label, CrossEntropyLoss for multiclass).
        optim: Optimizer for updating trainable parameters.
        num_classes: Number of classes (or labels in multi-label mode). Must be ``>= 1``.
        multi_label: Whether this is a multi-label classification problem.
            Requires ``num_classes >= 2``.
        device: Device to run training on (``'cpu'``, ``'cuda'``, or ``'mps'``).
        early_stopping_patience: Stop training after this many epochs without
            improvement in validation/training loss. ``None`` disables.
        checkpoint_config: Optional :class:`~fenn.nn.utils.Checkpoint` for
            saving training state to disk.

    Note:
        For binary classification (``num_classes == 2``, ``multi_label=False``),
        labels should be ``[0, 1]`` shaped tensors. For multiclass, labels should
        be class indices. For multi-label, labels should be binary vectors of length
        ``num_classes``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optim: torch.optim.Optimizer,
        num_classes: int,
        multi_label: bool = False,
        device: torch.device | str = "cpu",
        early_stopping_patience: int | None = None,
        checkpoint_config: Checkpoint | None = None,
    ):
        """Initialize a ClassificationTrainer instance.

        Args:
            model: The neural network model to train.
            loss_fn: The loss function to use.
            optim: The optimizer to use.
            num_classes: The number of classes to predict.
            multi_label: Whether to use multi-label classification.
            device: The device on which the data will be loaded.
            early_stopping_patience: The number of epochs to wait before early stopping.
            checkpoint_config: The checkpoint configuration. If `None`, checkpointing is disabled.
        """
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optim=optim,
            device=device,
            early_stopping_patience=early_stopping_patience,
            checkpoint_config=checkpoint_config,
        )

        self._num_classes = num_classes
        self.multi_label = multi_label

        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if self.multi_label and num_classes < 2:
            raise ValueError("Multi-label classification requires num_classes >= 2")

        if multi_label:
            self._task_type = "multilabel"
        elif num_classes == 2:
            self._task_type = "binary"
        else:
            self._task_type = "multiclass"

        if multi_label:
            logger.info(f"Multi-label classification ({num_classes} labels) mode")
        elif num_classes == 2:
            logger.info("Binary classification mode detected.")
        else:
            logger.info(f"Multi-class classification ({num_classes} classes)")

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: DataLoader | None = None,
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

        progress = Progress(
            TextColumn(
                "[bold blue]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        progress.start()
        epoch_task = progress.add_task(
            "Training",
            total=epochs - state.epoch,
            epoch=state.epoch,
            total_epochs=epochs,
            info="",
        )

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

                if self._num_classes == 2:
                    labels = labels.view(-1, 1).float()

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

            progress.update(
                epoch_task,  # pyright: ignore[reportArgumentType]
                advance=1,
                epoch=epoch,
                info=f"Train Mean Loss : {state.train_loss:.4f}",
            )

            # --- NO VALIDATION ---
            if val_loader is None:
                state.val_loss = None

                progress.console.print(
                    f"[bold blue]Epoch {epoch}/{epochs}[/bold blue] Train Loss: {state.train_loss:.4f}"
                )
                logger.info(
                    f"Epoch {epoch}/{epochs} - Train Loss: {state.train_loss:.4f}",
                    extra={"skip_console": True},
                )

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

                if val_n_batches > 0:
                    val_mean_loss = val_total_loss / val_n_batches
                    val_acc = accuracy_score(val_labels, val_predictions)

                    progress.console.print(
                        f"[bold blue]Epoch {epoch}/{epochs}[/bold blue] Train Loss: {state.train_loss:.4f} | Val Loss: {val_mean_loss:.4f} | Val Acc: {val_acc:.4f}"
                    )
                    logger.info(
                        f"Epoch {epoch}/{epochs} - Train Loss: {state.train_loss:.4f} | Val Loss: {val_mean_loss:.4f} | Val Acc: {val_acc:.4f}",
                        extra={"skip_console": True},
                    )

                state.val_loss = val_total_loss / val_n_batches
                state.acc = accuracy_score(val_labels, val_predictions)

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
                logger.info(
                    f"Early stopping triggered.  No improvement in {_reason} for {self._early_stopping_patience} epochs.",
                    extra={"skip_console": True},
                )

                break

        progress.stop()

    def predict(
        self,
        dataloader_or_batch: DataLoader | torch.Tensor,
        return_proba: bool = False,
    ):
        """Predicts the output of the model for a given dataloader or batch.

        Args:
            dataloader_or_batch: A DataLoader or a torch tensor.
            return_proba: If true, also returns the predicted probabilities alongside the predicted labels

        Returns:
            list: A list of predictions.
            list[list]:
                If `return_proba=True`, returns a tuple where:
                - first element is the list of predicted labels
                - second element is the list of predicted probabilities
        """

        self._model.eval()
        predictions = []
        proba = []

        def predict_batch(batch):
            batch = self._move_to_device(batch, self._device)
            logits = self._model(batch)
            if self._task_type == "binary":
                preds = torch.sigmoid(logits).squeeze(-1)
                if return_proba:
                    proba.extend(preds.cpu().tolist())
                preds = (preds > 0.5).long()
            elif self._task_type == "multiclass":
                preds = torch.softmax(logits, dim=1)
                if return_proba:
                    proba.extend(preds.cpu().tolist())
                preds = torch.argmax(logits, dim=1)
            else:
                preds = torch.sigmoid(logits)
                if return_proba:
                    proba.extend(preds.cpu().tolist())
                preds = (preds > 0.5).long()
            predictions.extend(preds.cpu().tolist())

        with torch.no_grad():
            if isinstance(dataloader_or_batch, DataLoader):
                for data, _ in dataloader_or_batch:
                    predict_batch(data)
            else:
                predict_batch(dataloader_or_batch)

        if return_proba:
            return predictions, proba

        return predictions
