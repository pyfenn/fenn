from pathlib import Path
from typing import List, Optional, Union

import torch

from fenn.logging import Logger
from fenn.nn.utils.state import TrainingState


class Checkpoint:
    """Checkpoint training state at the given epochs."""

    def __init__(
        self,
        *,
        name: str = "checkpoint",
        dir: Union[Path, str],
        epochs: Optional[Union[int, List[int]]] = None,
        save_best: bool = True,
    ):
        """Initialize the checkpoint configuration.

        Args:
            name: The name of the checkpoint file.
            dir: The directory to save checkpoints to.
            epochs: The epochs at which to save checkpoints.
            save_best: Whether to checkpoint the best model (based on validation or training loss).
        """

        self._logger = Logger()

        self.name = name
        self.dir = Path(dir)
        self.epochs = epochs
        self.save_best = save_best

    def _setup(self):
        """Set up the checkpoint directory and checks."""
        self.dir.mkdir(parents=True, exist_ok=True)

        if self.epochs is None and not self.save_best:
            self._logger.system_warning(
                "Checkpoint configuration is passed, but both `epochs` and `save_best` are unset.\n"
                "Models will not be checkpointed."
            )
            return

        if self.epochs is not None:
            self._logger.system_info(
                f"Checkpointing enabled. Checkpoints will be saved to {self.dir} every {self.epochs} epochs."
            )

        if self.save_best:
            self._logger.system_info(
                f"Best model checkpointing enabled. Best model will be saved to {self.dir}."
            )

        return self

    def save(self, state: TrainingState, is_best: bool = False):
        """Save a checkpoint of the training state at the current epoch.

        Args:
            state: The training state to checkpoint.
            is_best: If true save as best model
        """

        epoch = state.epoch

        if not is_best:
            filename = f"{self.name}_epoch_{epoch}.pt"
            filepath = self.dir / filename
            torch.save(state.to_dict(), filepath)
            self._logger.system_info(
                f"Checkpoint saved at epoch {epoch} to {filepath}."
            )

        elif is_best and self.save_best:
            filename = f"{self.name}_best.pt"
            filepath = self.dir / filename
            torch.save(state.to_dict(), filepath)
            self._logger.system_info(
                f"Best model checkpoint saved to {filepath} with acc {state.acc:.4f}."
            )

    def load(
        self, checkpoint_path: Union[str, Path], device: Optional[torch.device] = None
    ):
        """Load a checkpoint from the given path.

        Args:
            path: Path to the checkpoint file.
            device: The device to load the checkpoint onto.

        Returns:
            The training state of the checkpoint.
        """

        filepath = Path(checkpoint_path)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {filepath}")
        if not filepath.suffix == ".pt":
            raise ValueError(
                f"Invalid checkpoint path: {filepath}. Checkpoint must be a .pt file."
            )

        checkpoint = torch.load(filepath, map_location=device)
        state = TrainingState.from_dict(checkpoint)

        self._logger.system_info(
            f"Checkpoint loaded from {checkpoint_path}. Resuming from "
            f"epoch {state.epoch} with training loss {state.train_loss:.4f}."
        )
        return state

    def load_at_epoch(self, epoch: int, device: Optional[torch.device] = None):
        """Load the checkpoint at the given epoch.

        Args:
            epoch: Epoch to load the checkpoint at.
            device: The device to load the checkpoint onto.

        Returns:
            The training state of the checkpoint.
        """

        filepath = self.dir / f"{self.name}_epoch_{epoch}.pt"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Filepath does not exist: {filepath}.\n"
                f"Training state at epoch {epoch} has not been checkpointed."
            )
        return self.load(filepath, device)

    def load_best(self, device: Optional[torch.device] = None):
        """Load the best checkpoint.

        Args:
            device: The device to load the checkpoint onto.

        Returns:
            The training state of the checkpoint.
        """

        filepath = self.dir / f"{self.name}_best.pt"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Filepath does not exist: {filepath}.\n"
                f"Best training state has not been checkpointed."
            )
        return self.load(filepath, device)
