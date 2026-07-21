"""LSTM models for sequence classification and next-token generation.

Training is delegated to fenn's existing trainers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as torch_optim
from torch.utils.data import DataLoader, TensorDataset

from fenn.nn.trainers import ClassificationTrainer

_SOLVERS = {
    "adam": torch_optim.Adam,
    "sgd": torch_optim.SGD,
}


def _flatten_targets(batch):
    """Stack sequence inputs and flatten per-position targets."""
    X, y = zip(*batch)
    return torch.stack(X), torch.stack(y).reshape(-1)


def _make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    flatten_targets: bool = False,
) -> DataLoader:
    dataset = TensorDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        collate_fn=_flatten_targets if flatten_targets else None,
    )


class _LSTMClassificationModel(nn.Module):
    """Internal many-to-one LSTM model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.output = nn.Linear(hidden_size * directions, output_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(X)
        return self.output(sequence_output[:, -1, :])


class _LSTMGenerationModel(nn.Module):
    """Internal many-to-many LSTM model for next-token prediction."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(self.embedding(X.long()))
        logits = self.output(sequence_output)
        return logits.reshape(-1, logits.shape[-1])


class BaseLSTM:
    """Shared configuration and helpers for high-level LSTM models."""

    def __init__(
        self,
        hidden_size: int = 100,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        solver: str = "adam",
        learning_rate_init: float = 0.001,
        batch_size: int = 32,
        max_iter: int = 200,
        early_stopping: bool = False,
        n_iter_no_change: int = 10,
        validation_fraction: float = 0.1,
        device: str = "cpu",
    ):
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than 0.")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than 0.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1.")
        if solver not in _SOLVERS:
            raise ValueError(
                f"Unknown solver '{solver}'. Must be one of {list(_SOLVERS)}."
            )
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.device = device

        self._model: nn.Module | None = None
        self._trainer: ClassificationTrainer | None = None

    def _make_trainer(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        num_classes: int,
    ) -> ClassificationTrainer:
        optimizer = _SOLVERS[self.solver](
            model.parameters(),
            lr=self.learning_rate_init,
        )
        return ClassificationTrainer(
            model=model,
            loss_fn=loss_fn,
            optim=optimizer,
            num_classes=num_classes,
            device=self.device,
            early_stopping_patience=(
                self.n_iter_no_change if self.early_stopping else None
            ),
        )

    def _make_loaders(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        flatten_targets: bool = False,
    ) -> tuple[DataLoader, DataLoader | None]:
        if not self.early_stopping:
            return (
                _make_loader(
                    X,
                    y,
                    self.batch_size,
                    True,
                    flatten_targets,
                ),
                None,
            )

        n_val = max(1, int(len(X) * self.validation_fraction))
        train_loader = _make_loader(
            X[:-n_val],
            y[:-n_val],
            self.batch_size,
            True,
            flatten_targets,
        )
        val_loader = _make_loader(
            X[-n_val:],
            y[-n_val:],
            self.batch_size,
            False,
            flatten_targets,
        )
        return train_loader, val_loader

    def _fit(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        num_classes: int,
        X: torch.Tensor,
        y: torch.Tensor,
        flatten_targets: bool = False,
    ) -> None:
        self._model = model
        self._trainer = self._make_trainer(model, loss_fn, num_classes)
        train_loader, val_loader = self._make_loaders(
            X,
            y,
            flatten_targets,
        )
        self._trainer.fit(
            train_loader,
            epochs=self.max_iter,
            val_loader=val_loader,
        )

    def _check_is_fitted(self) -> None:
        if self._trainer is None:
            raise RuntimeError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )


class LSTMClassifier(BaseLSTM):
    """LSTM classifier for 3D sequence data."""

    classes_: np.ndarray

    def fit(self, X, y) -> "LSTMClassifier":
        """Fit the classifier on sequence data."""
        X_t = self._prepare_input(X)
        y_arr = np.asarray(y)

        if len(X_t) != len(y_arr):
            raise ValueError("X and y must contain the same number of samples.")

        self.classes_ = np.unique(y_arr)
        num_classes = len(self.classes_)

        if num_classes < 2:
            raise ValueError(
                "LSTMClassifier requires at least 2 distinct classes in y."
            )

        label_to_index = {label: index for index, label in enumerate(self.classes_)}
        y_t = torch.as_tensor(
            [label_to_index[label] for label in y_arr],
            dtype=torch.long,
        )

        self.n_features_in_ = X_t.shape[2]
        model = _LSTMClassificationModel(
            input_size=self.n_features_in_,
            hidden_size=self.hidden_size,
            output_size=1 if num_classes == 2 else num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        loss_fn = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()

        self._fit(model, loss_fn, num_classes, X_t, y_t)
        return self

    def _prepare_input(self, X) -> torch.Tensor:
        if torch.is_tensor(X):
            X_t = X.float()
        else:
            X_t = torch.as_tensor(
                np.asarray(X),
                dtype=torch.float32,
            )

        if X_t.ndim != 3:
            raise ValueError(
                "X must be a 3D array with shape "
                "(n_samples, sequence_length, n_features)."
            )
        return X_t

    def predict(self, X) -> np.ndarray:
        """Predict a class for each input sequence."""
        self._check_is_fitted()
        assert self._trainer is not None

        predictions = self._trainer.predict(self._prepare_input(X))
        return self.classes_[np.asarray(predictions)]

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for each input sequence."""
        self._check_is_fitted()
        assert self._trainer is not None

        _, probabilities = self._trainer.predict(
            self._prepare_input(X),
            return_proba=True,
        )
        probabilities = np.asarray(probabilities)

        if probabilities.ndim == 1:
            probabilities = np.stack(
                [1 - probabilities, probabilities],
                axis=1,
            )

        return probabilities

    def score(self, X, y) -> float:
        """Return mean classification accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


class LSTMGenerator(BaseLSTM):
    """LSTM model for next-token sequence generation."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 100,
        num_layers: int = 1,
        dropout: float = 0.0,
        solver: str = "adam",
        learning_rate_init: float = 0.001,
        batch_size: int = 32,
        max_iter: int = 200,
        early_stopping: bool = False,
        n_iter_no_change: int = 10,
        validation_fraction: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            solver=solver,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=validation_fraction,
            device=device,
        )

        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def fit(self, X, y) -> "LSTMGenerator":
        """Fit the generator for next-token prediction."""
        X_t = torch.as_tensor(
            np.asarray(X),
            dtype=torch.long,
        )
        y_t = torch.as_tensor(
            np.asarray(y),
            dtype=torch.long,
        )
        self._validate_data(X_t, y_t)

        model = _LSTMGenerationModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        self._fit(
            model,
            nn.CrossEntropyLoss(),
            self.vocab_size,
            X_t,
            y_t,
            flatten_targets=True,
        )
        return self

    def _validate_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("X and y must be 2D arrays.")

        if X.shape != y.shape:
            raise ValueError("X and y must have the same shape.")

        if X.numel() == 0:
            raise ValueError("X and y cannot be empty.")

        for array, name in ((X, "X"), (y, "y")):
            if array.min() < 0 or array.max() >= self.vocab_size:
                raise ValueError(
                    f"{name} contains token IDs outside the configured vocabulary."
                )

    def generate(self, seed, length: int) -> np.ndarray:
        """Generate tokens autoregressively from a seed sequence."""
        self._check_is_fitted()

        if length < 0:
            raise ValueError("length must be greater than or equal to 0.")

        seed_t = torch.as_tensor(
            np.asarray(seed),
            dtype=torch.long,
        )

        if seed_t.ndim != 1 or seed_t.numel() == 0:
            raise ValueError("seed must be a non-empty 1D sequence of token IDs.")

        if seed_t.min() < 0 or seed_t.max() >= self.vocab_size:
            raise ValueError(
                "seed contains token IDs outside the configured vocabulary."
            )

        assert self._model is not None

        generated = seed_t.tolist()
        model = self._model.to(self.device)
        model.eval()

        with torch.no_grad():
            for _ in range(length):
                sequence = torch.as_tensor(
                    [generated],
                    dtype=torch.long,
                    device=self.device,
                )
                logits = model(sequence)
                next_token = int(torch.argmax(logits[-1]).item())
                generated.append(next_token)

        return np.asarray(generated)
