import inspect
from copy import deepcopy
from typing import Dict, List, Optional, Union, cast

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
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from fenn.nn.utils import Checkpoint
from fenn.utils.logging import logger

from .trainer import Trainer

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None  # type: ignore
    TaskType = None  # type: ignore

_SUPPORTED_TASK_TYPES = {
    "SEQ_CLS",
    "CAUSAL_LM",
    "SEQ_2_SEQ_LM",
    "TOKEN_CLS",
    "QUESTION_ANS",
}
_GENERATIVE_TASK_TYPES = {"CAUSAL_LM", "SEQ_2_SEQ_LM"}


class LoRATrainer(Trainer):
    """
    LoRATrainer extends the base Trainer to support Parameter-Efficient Fine-Tuning (PEFT)
    using LoRA (Low-Rank Adaptation).

    Designed for HuggingFace transformer models. DataLoaders must yield dicts whose keys
    match the model's forward signature (e.g. ``input_ids``, ``attention_mask``, ``labels``).
    Loss is taken from ``outputs.loss`` when labels are present in the batch; ``loss_fn``
    is used as a fallback if the model does not return a loss.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        task_type: str = "SEQ_CLS",
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        loss_fn: Optional[torch.nn.Module] = None,
        device: Union[torch.device, str] = "cpu",
        early_stopping_patience: Optional[int] = None,
        checkpoint_config: Optional[Checkpoint] = None,
    ):
        """Initialize the LoRATrainer.

        Args:
            model: The base HuggingFace model to fine-tune.
            optim: The optimizer.
            task_type: LoRA task type. One of ``"SEQ_CLS"``, ``"CAUSAL_LM"``,
                ``"SEQ_2_SEQ_LM"``, ``"TOKEN_CLS"``, ``"QUESTION_ANS"``.
                Defaults to ``"SEQ_CLS"``.
            r: LoRA rank — number of low-rank dimensions. Defaults to ``8``.
            lora_alpha: LoRA scaling factor. Defaults to ``16``.
            lora_dropout: Dropout applied to LoRA layers. Defaults to ``0.1``.
            target_modules: Module names to apply LoRA to (e.g. ``["q_proj", "v_proj"]``).
                If ``None``, peft auto-detects based on the architecture.
            bias: Which biases to train. One of ``"none"``, ``"all"``, ``"lora_only"``.
                Defaults to ``"none"``.
            loss_fn: Optional external loss function. Used when the model does not return
                a loss (i.e. labels are absent from the batch). Ignored otherwise.
            device: Device to train on. Defaults to ``"cpu"``.
            early_stopping_patience: Epochs without improvement before early stopping.
                Disabled when ``None``.
            checkpoint_config: Checkpoint configuration. Disabled when ``None``.
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "peft is not installed. Install fenn with the [transformers] extra: "
                "pip install fenn[transformers]"
            )

        task_type_upper = task_type.upper()
        if task_type_upper not in _SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                f"Choose from: {sorted(_SUPPORTED_TASK_TYPES)}"
            )

        lora_config = LoraConfig(
            task_type=getattr(TaskType, task_type_upper),
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # The optimizer was built before get_peft_model injected the LoRA parameters,
        # so those new tensors are not tracked. Rebuild it now with only the trainable
        # (LoRA) parameters, preserving the original optimizer class and hyperparameters.
        # Filter optim.defaults to valid constructor args only — PyTorch adds internal
        # keys (e.g. decoupled_weight_decay) that are not accepted by __init__.
        optimizer_class = type(optim)
        valid_init_params = set(
            inspect.signature(optimizer_class.__init__).parameters
        ) - {"self", "params"}
        optimizer_defaults = {
            k: v for k, v in optim.defaults.items() if k in valid_init_params
        }
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optim = optimizer_class(trainable_params, **optimizer_defaults)

        super().__init__(
            model=model,
            loss_fn=loss_fn,  # type: ignore[arg-type]
            optim=optim,
            device=device,
            early_stopping_patience=early_stopping_patience,
            checkpoint_config=checkpoint_config,
        )

        self._task_type = task_type_upper
        self._is_generative = task_type_upper in _GENERATIVE_TASK_TYPES

        logger.info(
            f"LoRA applied — task: {task_type_upper} | r={r} | alpha={lora_alpha} | dropout={lora_dropout}",
            extra={"skip_console": "True"},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(
        self, batch: Dict
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run one forward pass.

        Returns:
            (loss, logits) — either may be ``None`` if the model / batch
            does not produce them.
        """
        outputs = self._model(**batch)
        loss: Optional[torch.Tensor] = getattr(outputs, "loss", None)
        logits: Optional[torch.Tensor] = getattr(outputs, "logits", None)

        if loss is None and self._loss_fn is not None and logits is not None:
            labels = batch.get("labels")
            if labels is not None:
                loss = self._loss_fn(logits, labels)

        return loss, logits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        val_epochs: int = 1,
    ):
        """Train the model with optional validation and early stopping.

        DataLoaders must yield dicts with at minimum ``input_ids`` and
        ``attention_mask``. Include ``labels`` to have the model (or
        ``loss_fn``) compute the loss.

        Args:
            train_loader: DataLoader for training data.
            epochs: Total number of epochs to train for.
            val_loader: DataLoader for validation data (optional).
            val_epochs: How often to run validation (in epochs).
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

            for batch in train_loader:
                batch = self._move_to_device(batch, self._device)
                loss, _ = self._forward(batch)

                if loss is None:
                    raise ValueError(
                        "No loss was returned from the model. Ensure that `labels` are "
                        "included in the batch or pass a `loss_fn` to LoRATrainer."
                    )

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
                info=f"Train Loss: {state.train_loss:.4f}",
            )

            # --- NO VALIDATION ---
            if val_loader is None:
                state.val_loss = None

                progress.console.print(
                    f"[bold blue]Epoch {epoch}/{epochs}[/bold blue] "
                    f"Train Loss: {state.train_loss:.4f}"
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
                val_total_loss = 0.0
                val_n_batches = 0
                all_preds: list = []
                all_labels: list = []

                with torch.no_grad():
                    for batch in val_loader:
                        batch = self._move_to_device(batch, self._device)
                        val_loss, logits = self._forward(batch)

                        if val_loss is not None:
                            val_total_loss += float(val_loss.item())
                        val_n_batches += 1

                        if logits is not None and not self._is_generative:
                            preds = torch.argmax(logits, dim=-1)
                            all_preds.extend(preds.cpu().tolist())
                            labels = batch.get("labels")
                            if labels is not None:
                                all_labels.extend(labels.cpu().tolist())

                if val_n_batches == 0:
                    raise ValueError("val_loader produced 0 batches; cannot validate.")

                state.val_loss = val_total_loss / val_n_batches

                if all_preds and all_labels:
                    val_acc = accuracy_score(all_labels, all_preds)
                    state.acc = val_acc
                    progress.console.print(
                        f"[bold blue]Epoch {epoch}/{epochs}[/bold blue] "
                        f"Train Loss: {state.train_loss:.4f} | "
                        f"Val Loss: {state.val_loss:.4f} | Val Acc: {val_acc:.4f}"
                    )
                    logger.info(
                        f"Epoch {epoch}/{epochs} - Train Loss: {state.train_loss:.4f} | "
                        f"Val Loss: {state.val_loss:.4f} | Val Acc: {val_acc:.4f}",
                        extra={"skip_console": True},
                    )
                else:
                    progress.console.print(
                        f"[bold blue]Epoch {epoch}/{epochs}[/bold blue] "
                        f"Train Loss: {state.train_loss:.4f} | Val Loss: {state.val_loss:.4f}"
                    )
                    logger.info(
                        f"Epoch {epoch}/{epochs} - Train Loss: {state.train_loss:.4f} | "
                        f"Val Loss: {state.val_loss:.4f}",
                        extra={"skip_console": True},
                    )

                if state.val_loss < state.best_val_loss:
                    state.best_val_loss = state.val_loss

                    self._best_state = state.clone(
                        model_state_dict=self._model.state_dict(),
                        optimizer_state_dict=self._optimizer.state_dict(),
                    )
                    self._best_model = deepcopy(self._model)
                    self._best_model.load_state_dict(self._best_state.model_state_dict)  # pyright: ignore[reportOptionalMemberAccess, reportArgumentType]

                    if self._checkpoint is not None:
                        self._checkpoint.save(self._best_state, is_best=True)  # pyright: ignore[reportArgumentType]

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
                _reason = (
                    "validation loss" if val_loader is not None else "training loss"
                )
                logger.info(
                    f"Early stopping triggered. No improvement in {_reason} "
                    f"for {self._early_stopping_patience} epochs.",
                    extra={"skip_console": True},
                )
                break

        progress.stop()

    def predict(self, dataloader_or_batch: Union[DataLoader, Dict, torch.Tensor]):
        """Generate predictions for a dataloader or a single batch.

        Labels are stripped from dict batches before inference so the model
        does not compute a loss during prediction.

        For classification tasks (``SEQ_CLS``, ``TOKEN_CLS``, ``QUESTION_ANS``),
        returns a flat list of predicted class indices.

        For generative tasks (``CAUSAL_LM``, ``SEQ_2_SEQ_LM``), returns a list
        of logit tensors (one per batch).

        Args:
            dataloader_or_batch: A DataLoader, a dict batch, or a raw tensor.

        Returns:
            list: Predicted class indices (classification) or logit tensors (generative).
        """
        self._model.eval()
        predictions: list = []

        def predict_batch(raw_batch):
            if isinstance(raw_batch, dict):
                raw_batch = self._move_to_device(raw_batch, self._device)
                inference_batch = {k: v for k, v in raw_batch.items() if k != "labels"}
                outputs = self._model(**inference_batch)
                logits = outputs.logits
            else:
                raw_batch = self._move_to_device(raw_batch, self._device)
                logits = self._model(raw_batch)

            if self._is_generative:
                predictions.append(logits.cpu())
            else:
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().tolist())

        with torch.no_grad():
            if isinstance(dataloader_or_batch, DataLoader):
                for raw in dataloader_or_batch:
                    predict_batch(raw)
            else:
                predict_batch(dataloader_or_batch)

        return predictions
