from pathlib import Path
from typing import Any, Dict, Optional

from fenn.args import Parser
from fenn.secrets.keystore import KeyStore

from fenn.logging.backends.logging import LoggingBackend
from fenn.logging.backends.wandb import WandbBackend
from fenn.logging.backends.tensorboard import TensorboardBackend

class Logger:
    """Singleton logging system for FENN (facade over multiple backends)."""

    _instance: Optional["Logger"] = None

    def __new__(cls) -> "Logger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def get_instance() -> "Logger":
        return Logger()

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._parser = Parser()
        self._keystore = KeyStore()

        self._logging_backend = LoggingBackend()
        self._wandb_backend = WandbBackend(
            keystore=self._keystore,
            system_info=self._logging_backend.system_info,
            system_warning=self._logging_backend.system_warning,
            system_exception=self._logging_backend.system_exception,
        )
        self._tensorboard_backend = TensorboardBackend(
            system_info=self._logging_backend.system_info,
            system_warning=self._logging_backend.system_warning,
            system_exception=self._logging_backend.system_exception,
        )

        self._args: Optional[Dict[str, Any]] = None
        self._initialized = True

    # --------------------------
    # same public API as before
    # --------------------------
    def system_info(self, message: str) -> None:
        self._logging_backend.system_info(message)

    def system_warning(self, message: str) -> None:
        self._logging_backend.system_warning(message)

    def system_exception(self, message: str) -> None:
        self._logging_backend.system_exception(message)

    def user_info(self, message: str) -> None:
        self._logging_backend.user_info(message)

    def user_warning(self, message: str) -> None:
        self._logging_backend.user_warning(message)

    def user_exception(self, message: str) -> None:
        self._logging_backend.user_exception(message)

    # --------------------------
    # lifecycle
    # --------------------------
    def start(self) -> None:
        self._args = self._parser.args
        self._logging_backend.start(self._args)

        if self._args.get("wandb"):
            self._wandb_backend.start(self._args)

        if self._args.get("tensorboard"):
            self._tensorboard_backend.start(self._args)

    def stop(self) -> None:
        # stop external backends first, then restore print
        self._wandb_backend.stop()
        self._tensorboard_backend.stop()
        self._logging_backend.stop()

    # --------------------------
    # accessors (optional)
    # --------------------------
    @property
    def wandb_run(self) -> Optional[Any]:
        return self._wandb_backend.run

    @property
    def tensorboard(self) -> Optional[Any]:
        return self._tensorboard_backend.writer

    @property
    def log_file(self) -> Optional[Path]:
        return self._logging_backend.log_file
