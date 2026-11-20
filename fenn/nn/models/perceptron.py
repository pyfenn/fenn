"""Single-Layer Perceptron models.

This module provides :class:`SingleLayerPerceptronClassifier` and
:class:`SingleLayerPerceptronRegressor`, which are essentially Multi-Layer
Perceptrons with zero hidden layers, connecting inputs directly to outputs.
"""

from __future__ import annotations

from fenn.nn.models.mlp import MLPClassifier, MLPRegressor


class SingleLayerPerceptronClassifier(MLPClassifier):
    """Single-layer Perceptron classifier.

    A scikit-learn-style estimator for a single-layer neural network
    classification (equivalent to logistic regression for binary classification,
    or multinomial logistic regression for multi-class).

    Args:
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
            hidden_layer_sizes=(),
            activation="identity",
            solver=solver,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=validation_fraction,
            device=device,
        )


class SingleLayerPerceptronRegressor(MLPRegressor):
    """Single-layer Perceptron regressor.

    A scikit-learn-style estimator for a single-layer neural network
    regression (equivalent to linear regression).

    Args:
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
            hidden_layer_sizes=(),
            activation="identity",
            solver=solver,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=validation_fraction,
            device=device,
        )

# Alias for backwards compatibility with users who might just expect SingleLayerPerceptron
SingleLayerPerceptron = SingleLayerPerceptronClassifier
