from dataclasses import asdict, dataclass, replace
from typing import Any, Optional, TypeAlias

StateDict: TypeAlias = dict[str, Any]


@dataclass
class TrainingState:
    """Training state for a neural network model."""

    # May use pydantic.BaseModel instead of dataclass

    epoch: int

    acc: Optional[float] = None
    """Accuracy on the validation set (if provided)"""

    train_loss: Optional[float] = None
    """Train mean loss over all batches"""

    val_loss: Optional[float] = None
    """Validation mean loss over all batches"""

    model_state_dict: Optional[StateDict] = None
    optimizer_state_dict: Optional[StateDict] = None

    patience_counter: int = 0
    """Patience counter up to this epoch for early stopping."""

    best_acc: float = float("-inf")
    """Best validation accuracy achieved up to this epoch"""

    best_train_loss: float = float("inf")
    """Best train loss achieved up to this epoch"""

    best_val_loss: float = float("inf")
    """Best validation loss achieved up to this epoch"""

    def to_dict(self):
        """Serialize the training state to a dictionary."""

        assert self.model_state_dict is not None
        assert self.optimizer_state_dict is not None

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        """Deserialize a dictionary to a TrainingState instance.

        Args:
            data: The serialized training state.

        Returns:
            A new TrainingState instance.
        """

        return cls(**data)

    def clone(self, **kwargs):
        """Clone the training state with optional updated fields.

        Returns:
            A new TrainingState instance.
        """

        return replace(self, **kwargs)
