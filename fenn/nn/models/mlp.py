"""Scikit-learn-inspired Multi-Layer Perceptron models.

This module provides :class:`MLPClassifier` and :class:`MLPRegressor`, two
high-level estimators for users who want to train a simple feed-forward
neural network without writing any PyTorch training code themselves.

Both classes build a plain ``torch.nn.Sequential`` MLP internally and
delegate the entire training loop to fenn's existing trainers
(:class:`~fenn.nn.trainers.ClassificationTrainer` and
:class:`~fenn.nn.trainers.RegressionTrainer`). No training logic lives in
this module.

The public API intentionally mirrors scikit-learn's
``sklearn.neural_network.MLPClassifier`` / ``MLPRegressor``. See
https://scikit-learn.org/stable/modules/neural_networks_supervised.html
for the API this module takes inspiration from.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as torch_optim
from torch.utils.data import DataLoader, TensorDataset

from fenn.nn.trainers import ClassificationTrainer, RegressionTrainer

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "logistic": nn.Sigmoid,
    "identity": nn.Identity,
}

_SOLVERS = {
    "adam": torch_optim.Adam,
    "sgd": torch_optim.SGD,
}


def _build_mlp(
    input_size: int,
    hidden_layer_sizes: Sequence[int],
    output_size: int,
    activation: str,
) -> nn.Sequential:
    """Build a feed-forward MLP as a ``torch.nn.Sequential`` stack.

    Args:
        input_size: Number of input features.
        hidden_layer_sizes: Number of units in each hidden layer, e.g. ``(100,)``.
        output_size: Number of output units (number of classes for
            multi-class classification, ``1`` for binary classification or
            regression).
        activation: Activation applied after each hidden layer. One of
            ``'relu'``, ``'tanh'``, ``'logistic'``, ``'identity'``.

    Returns:
        A ``torch.nn.Sequential`` module. No activation is applied to the
        final layer; outputs are raw logits (classification) or raw
        predictions (regression).

    Raises:
        ValueError: If ``activation`` is not a recognized name.
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{activation}'. Must be one of {list(_ACTIVATIONS)}."
        )

    activation_cls = _ACTIVATIONS[activation]

    layers: list[nn.Module] = []
    in_features = input_size
    for units in hidden_layer_sizes:
        layers.append(nn.Linear(in_features, units))
        layers.append(activation_cls())
        in_features = units

    layers.append(nn.Linear(in_features, output_size))
    return nn.Sequential(*layers)


def _to_tensor(data) -> torch.Tensor:
    """Convert array-like input (list, numpy array, or tensor) to a float tensor."""
    if torch.is_tensor(data):
        return data.float()
    return torch.as_tensor(np.asarray(data), dtype=torch.float32)


def _make_loader(
    X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(X, y)
    return DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle
    )


class BaseMLP:
    """Shared setup logic for :class:`MLPClassifier` and :class:`MLPRegressor`.

    This class is not meant to be instantiated directly; use one of the two
    subclasses instead.

    Args:
        hidden_layer_sizes: Number of neurons in each hidden layer, e.g.
            ``(100,)`` for a single hidden layer of 100 units, or
            ``(64, 32)`` for two hidden layers.
        activation: Activation function for the hidden layers. One of
            ``'relu'``, ``'tanh'``, ``'logistic'``, ``'identity'``.
        solver: Optimizer used to train the weights. One of ``'adam'``, ``'sgd'``.
        learning_rate_init: Initial learning rate used by the optimizer.
        batch_size: Size of minibatches used during training.
        max_iter: Maximum number of training epochs.
        early_stopping: Whether to hold out ``validation_fraction`` of the
            training data and stop training when validation loss stops
            improving for ``n_iter_no_change`` epochs.
        n_iter_no_change: Number of epochs with no improvement to wait
            before stopping, when ``early_stopping=True``.
        validation_fraction: Proportion of training data to set aside for
            early stopping validation, when ``early_stopping=True``.
        device: Device to train on, e.g. ``'cpu'``, ``'cuda'``, ``'mps'``.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (100,),
        activation: str = "relu",
        solver: str = "adam",
        learning_rate_init: float = 0.001,
        batch_size: int = 32,
        max_iter: int = 200,
        early_stopping: bool = False,
        n_iter_no_change: int = 10,
        validation_fraction: float = 0.1,
        device: str = "cpu",
    ):
        if solver not in _SOLVERS:
            raise ValueError(
                f"Unknown solver '{solver}'. Must be one of {list(_SOLVERS)}."
            )
        if not (0.0 < validation_fraction < 1.0):
            raise ValueError("validation_fraction must be between 0 and 1.")
        if not isinstance(hidden_layer_sizes, (list, tuple)):
            raise TypeError("hidden_layer_sizes must be a sequence.")

        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.device = device

        self._model: nn.Module | None = None
        self._trainer: ClassificationTrainer | RegressionTrainer | None = None
        self.n_features_in_: int | None = None

    def _split_validation(self, X: torch.Tensor, y: torch.Tensor):
        """Hold out a deterministic validation split for early stopping."""
        n_val = max(1, int(len(X) * self.validation_fraction))
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        return X_train, y_train, X_val, y_val

    def _make_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return _SOLVERS[self.solver](model.parameters(), lr=self.learning_rate_init)

    def _check_is_fitted(self) -> None:
        if self._trainer is None:
            raise RuntimeError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )


class MLPClassifier(BaseMLP):
    """Multi-layer Perceptron classifier.

    A scikit-learn-style estimator for feed-forward neural network
    classification. Supports both binary and multi-class classification;
    the number of classes is inferred automatically from the labels passed
    to :meth:`fit`. Internally builds a ``torch.nn.Sequential`` MLP and
    delegates all training to
    :class:`~fenn.nn.trainers.ClassificationTrainer`.

    Example:
        >>> clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50)
        >>> clf.fit(X_train, y_train)
        >>> clf.predict(X_test)

    Note:
        Multi-label classification is not yet supported by this estimator,
        even though the underlying :class:`ClassificationTrainer` supports it.
    """

    _trainer: ClassificationTrainer | None
    classes_: np.ndarray

    def fit(self, X, y) -> "MLPClassifier":
        """Fit the MLP classifier on the given training data.

        Args:
            X: Array-like of shape ``(n_samples, n_features)``.
            y: Array-like of shape ``(n_samples,)`` with class labels.
                Labels do not need to be pre-encoded as integers.

        Returns:
            self
        """
        X_t = _to_tensor(X)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        num_classes = len(self.classes_)

        if num_classes < 2:
            raise ValueError("MLPClassifier requires at least 2 distinct classes in y.")

        # Encode labels as class indices (0..num_classes-1) based on sorted classes_.
        label_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        y_encoded = np.array([label_to_index[label] for label in y_arr])
        y_t = torch.as_tensor(y_encoded, dtype=torch.long)

        self.n_features_in_ = X_t.shape[1]
        out_features = 1 if num_classes == 2 else num_classes
        model = _build_mlp(
            self.n_features_in_, self.hidden_layer_sizes, out_features, self.activation
        )
        self._model = model

        loss_fn = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        optimizer = self._make_optimizer(model)

        trainer = ClassificationTrainer(
            model=model,
            loss_fn=loss_fn,
            optim=optimizer,
            num_classes=num_classes,
            device=self.device,
            early_stopping_patience=self.n_iter_no_change
            if self.early_stopping
            else None,
        )
        self._trainer = trainer

        val_loader = None
        if self.early_stopping:
            X_train, y_train, X_val, y_val = self._split_validation(X_t, y_t)
            train_loader = _make_loader(X_train, y_train, self.batch_size, shuffle=True)
            val_loader = _make_loader(X_val, y_val, self.batch_size, shuffle=False)
        else:
            train_loader = _make_loader(X_t, y_t, self.batch_size, shuffle=True)

        trainer.fit(train_loader, epochs=self.max_iter, val_loader=val_loader)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels for samples in ``X``.

        Args:
            X: Array-like of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples,)`` with predicted labels, using the
            original label values seen during :meth:`fit`.
        """
        self._check_is_fitted()
        assert self._trainer is not None
        X_t = _to_tensor(X)
        preds = self._trainer.predict(X_t)
        return self.classes_[np.asarray(preds)]

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for samples in ``X``.

        Args:
            X: Array-like of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples, n_classes)`` with predicted
            probabilities for each class, ordered as in ``self.classes_``.
        """
        self._check_is_fitted()
        assert self._trainer is not None
        X_t = _to_tensor(X)
        _, proba = self._trainer.predict(X_t, return_proba=True)
        proba_arr = np.asarray(proba)
        if proba_arr.ndim == 1:
            # Binary case: trainer returns P(class 1); expand to two columns.
            proba_arr = np.stack([1 - proba_arr, proba_arr], axis=1)
        return proba_arr

    def score(self, X, y) -> float:
        """Return the mean accuracy on the given test data and labels."""
        preds = self.predict(X)
        return float(np.mean(np.asarray(preds) == np.asarray(y)))


class MLPRegressor(BaseMLP):
    """Multi-layer Perceptron regressor.

    A scikit-learn-style estimator for feed-forward neural network
    regression on a single continuous target. Internally builds a
    ``torch.nn.Sequential`` MLP with a single output unit and delegates all
    training to :class:`~fenn.nn.trainers.RegressionTrainer`.

    Example:
        >>> reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=50)
        >>> reg.fit(X_train, y_train)
        >>> reg.predict(X_test)
    """

    _trainer: RegressionTrainer | None

    def fit(self, X, y) -> "MLPRegressor":
        """Fit the MLP regressor on the given training data.

        Args:
            X: Array-like of shape ``(n_samples, n_features)``.
            y: Array-like of shape ``(n_samples,)`` with continuous targets.

        Returns:
            self
        """
        X_t = _to_tensor(X)
        y_t = _to_tensor(y).view(-1, 1)

        self.n_features_in_ = X_t.shape[1]
        model = _build_mlp(
            self.n_features_in_, self.hidden_layer_sizes, 1, self.activation
        )
        self._model = model

        loss_fn = nn.MSELoss()
        optimizer = self._make_optimizer(model)

        trainer = RegressionTrainer(
            model=model,
            loss_fn=loss_fn,
            optim=optimizer,
            device=self.device,
            early_stopping_patience=self.n_iter_no_change
            if self.early_stopping
            else None,
        )
        self._trainer = trainer

        val_loader = None
        if self.early_stopping:
            X_train, y_train, X_val, y_val = self._split_validation(X_t, y_t)
            train_loader = _make_loader(X_train, y_train, self.batch_size, shuffle=True)
            val_loader = _make_loader(X_val, y_val, self.batch_size, shuffle=False)
        else:
            train_loader = _make_loader(X_t, y_t, self.batch_size, shuffle=True)

        trainer.fit(train_loader, epochs=self.max_iter, val_loader=val_loader)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict continuous targets for samples in ``X``."""
        self._check_is_fitted()
        assert self._trainer is not None
        X_t = _to_tensor(X)
        preds = self._trainer.predict(X_t)
        return np.asarray(preds)

    def score(self, X, y) -> float:
        """Return the coefficient of determination (R^2) on the given test data."""
        preds = self.predict(X)
        y_arr = np.asarray(y, dtype=float)
        ss_res = np.sum((y_arr - preds) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)
