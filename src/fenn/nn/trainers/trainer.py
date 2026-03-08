from typing import Optional, Union, List
import copy
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fenn.logging import Logger

try: 
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class Trainer:
    """The base Trainer class for classification tasks."""

    def __init__(self,
                 model,
                 loss_fn,
                 optim,
                 epochs,
                 num_classes,
                 device="cpu",
                 return_model: str = "last",
                 checkpoint_dir: Optional[Union[Path, str]] = None,
                 checkpoint_epochs: Optional[Union[int, List[int]]] = None,
                 checkpoint_name: str = "checkpoint",
                 save_best: bool = False,
                 early_stopping_patience: Optional[int] = None
                 ):

        self._logger = Logger()

        self._device = device

        self._model = model.to(device)
        self._loss_fn = loss_fn
        self._optimizer = optim
        self._epochs = epochs
        self._num_classes = num_classes
        self._return_model = return_model.lower()

        if self._return_model not in {"last", "best"}:
            raise ValueError("return_model must be 'last' or 'best'")

        self._metrics = {}

        # chechpoint setup
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._checkpoint_epochs = checkpoint_epochs
        self._checkpoint_name = checkpoint_name
        self._save_best = save_best

        self._best_acc = float("-inf")
        self._best_loss = float("inf")

        # early stopping setup
        self._early_stopping_patience = early_stopping_patience
        self._patience_counter = 0

        # create the checkpoint directory if it doesn't exist and is enabled
        if self._checkpoint_dir and (self._checkpoint_epochs is not None or self._save_best):
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if self._checkpoint_epochs is not None:
                self._logger.system_info(f"Checkpointing enabled. Checkpoints will be saved to {self._checkpoint_dir} every {self._checkpoint_epochs} epochs.")
            if self._save_best:
                self._logger.system_info(f"Best model checkpointing enabled. Best model will be saved to {self._checkpoint_dir}.")

        # log early stopping setup
        if self._early_stopping_patience is not None:
            self._logger.system_info(f"Early stopping enabled with patience of {self._early_stopping_patience} epochs.")


    def _should_save_checkpoint(self, epoch: int):
        """Determine if a checkpoint should be saved at the given epoch.

        Args:
            epoch (int): The current epoch number. (0-indexed)
        Returns:
            bool: True if a checkpoint should be saved, False otherwise.

        """
        if self._checkpoint_dir is None or self._checkpoint_epochs is None:
            return False

        if isinstance(self._checkpoint_epochs, int):
            # save every N epochs
            return epoch % self._checkpoint_epochs == 0 or epoch == self._epochs-1
        elif isinstance(self._checkpoint_epochs, list):
            # save at specific epochs
            return epoch in self._checkpoint_epochs
        return False

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save a checkpoint of the model at the given epoch.

        Args:
            epoch (int): The current epoch number. (0-indexed)
            loss: training loss for this epoch
            is_best: if true save as best model
        """
        if self._checkpoint_dir is None:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'loss': loss,
            'best_acc': self._best_acc,
            'best_loss': self._best_loss
        }

        if not is_best:
            filename = f"{self._checkpoint_name}_epoch_{epoch}.pt"
            filepath = self._checkpoint_dir / filename
            torch.save(checkpoint, filepath)
            self._logger.system_info(f"Checkpoint saved at epoch {epoch} to {filepath}.")

        else:
            best_filepath = self._checkpoint_dir / f"{self._checkpoint_name}_best.pt"
            torch.save(checkpoint, best_filepath)
            self._logger.system_info(f"Best model checkpoint saved to {best_filepath} with loss {loss:.4f}.")

    def _binary_predict(self, data_loader):
        self._model.eval()
        predictions = []
        with torch.no_grad():
            for data, labels in data_loader:
                data = self._move_to_device(data, self._device)
                logits = self._model(data)
                preds = torch.sigmoid(logits).squeeze(-1)
                preds = (preds > 0.5).long()
                predictions.extend(preds.cpu().tolist())
        return predictions

    def _multiclass_predict(self, data_loader):
        self._model.eval()
        predictions = []
        with torch.no_grad():
            for data, labels in data_loader:
                data = self._move_to_device(data, self._device)
                logits = self._model(data)
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().tolist())
        return predictions

    def _move_to_device(self, batch, device):
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(x, device) for x in batch)
        if isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        return batch

    def fit(self, train_loader, val_loader=None, val_epochs: int = 5, start_epoch: int = 0):
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
            val_loader: DataLoader for validation data (optional).
            val_epochs: How often to evaluate on validation set (in epochs).
            start_epoch: Epoch to resume training from.

        Returns:
            The trained model (returned according to ``return_model``).
        """

        best_state_dict = None
        
        if HAS_RICH:
            progress = Progress(
                TextColumn("[bold blue]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("{task.fields[info]}"),
            )
            progress.start()
            epoch_task = progress.add_task(
                "Training",
                total = self._epochs - start_epoch,
                epoch = start_epoch,
                total_epochs = self._epochs,
                info=""
            )
            
        for epoch in range(start_epoch, self._epochs):

            # --- TRAIN ---
            if not HAS_RICH:
                self._logger.system_info(f"Epoch {epoch} started.")
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

            train_mean_loss = total_loss / n_batches
            
            if HAS_RICH:
                progress.update(epoch_task, advance=1, epoch=epoch+1, info=f"Train Mean Loss : {train_mean_loss:.4f}")
            else:
                print(f"Epoch {epoch}. Train Mean Loss: {train_mean_loss:.4f}")

            # --- NO VALIDATION ---
            if val_loader is None:
                if train_mean_loss < self._best_loss:
                    self._best_loss = train_mean_loss
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1

            # --- VALIDATION ---
            elif (epoch - start_epoch) % val_epochs == 0:  # or epoch == self._epochs - 1

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

                self._model.train()

                if val_n_batches == 0:
                    raise ValueError("val_loader produced 0 batches; cannot validate.")
                
                if val_n_batches > 0:
                    val_mean_loss = val_total_loss / val_n_batches
                    val_acc = accuracy_score(val_labels, val_predictions)
                    
                    if HAS_RICH:
                        progress.update(epoch_task, info=f"Train Mean Loss: {train_mean_loss:.4f} | Val Loss: {val_mean_loss:.4f} | Val Acc: {val_acc:.4f}")
                    else:
                        print(f"Epoch {epoch}. Validation Loss: {val_mean_loss:.4f}")
                        print(f"Epoch {epoch}. Validation Accuracy: {val_acc:.4f}")

                if val_acc > self._best_acc:
                    self._best_acc = val_acc
                    best_state_dict = copy.deepcopy(self._model.state_dict())
                    if self._save_best:
                        self._save_checkpoint(epoch, train_mean_loss, is_best=True)

                if val_mean_loss < self._best_loss:
                    self._best_loss = val_mean_loss
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1

            # --- CHECKPOINTING ---
            if self._should_save_checkpoint(epoch):
                self._save_checkpoint(epoch, train_mean_loss, is_best=False)

            # --- EARLY STOPPING ---
            if (
                self._early_stopping_patience is not None
                and self._patience_counter >= self._early_stopping_patience
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

        if self._return_model == "best":
            if val_loader is None:
                self._logger.system_info(
                    "return_model='best' requested but no validation loader provided; returning last model."
                )
            elif best_state_dict is None:
                self._logger.system_info(
                    "return_model='best' requested but best_state_dict has not been updated; returning last model."
                )
            else:
                if HAS_RICH:
                    progress.console.print(f"[green]Loading best model with validation accuracy {self._best_acc:.4f}[/green]")
                else:
                    print(f"Loading best model with validation accuracy {self._best_acc:.4f}")
                self._model.load_state_dict(best_state_dict)

        if HAS_RICH:
            progress.stop()
            
        return self._model

    def predict(self, data_loader):
        if self._num_classes == 2:
            return self._binary_predict(data_loader)
        else:
            return self._multiclass_predict(data_loader)

    def load_checkpoint(self, checkpoint_path: Union[Path, str]):
        """Load a checkpoint from the given path.

        Args:
            checkpoint_path (Path or str): Path to the checkpoint file.

        Returns:
            Epoch number from the loaded checkpoint

        Example:
            > trainer = Trainer(model, loss_fn, optimizer, epoch=100)
            > start_epoch = trainer.load_checkpoint("checkpoints/checkpoint_epoch_50.pt")
            # now you can resume training from epoch 51
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint.get('loss', float('inf'))
        self._best_loss = checkpoint.get('best_loss', float('inf'))
        self._best_acc = checkpoint.get('best_acc', float('-inf'))

        self._logger.system_info(f"Checkpoint loaded from {checkpoint_path}. Resuming from epoch {epoch} with loss {loss:.4f}.")
        return epoch
