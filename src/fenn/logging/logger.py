from pathlib import Path
from typing import Any, Dict, Optional

from colorama import Fore, Style

from fenn.args import Parser
from fenn.logging.backends.fnxml import FnXmlBackend
from fenn.logging.backends.logging import LoggingBackend
from fenn.logging.backends.tensorboard import TensorboardBackend
from fenn.logging.backends.wandb import WandbBackend
from fenn.secrets.keystore import KeyStore

try:
    from fenn.logging.backends.loguru import LoguruBackend
    _LOGURU_AVAILABLE = True
except ImportError:
    _LOGURU_AVAILABLE = False

class Logger:
    """Singleton logging system for Fenn (facade over multiple backends)."""

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

        self._logging_backend = LoguruBackend() if _LOGURU_AVAILABLE else LoggingBackend()
        self._fnxml_backend = FnXmlBackend()

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

        if hasattr(self._logging_backend, "set_print_sink"):
            self._logging_backend.set_print_sink(self._fnxml_backend.log_print)

    # --------------------------
    # same public API as before
    # --------------------------
    def system_info(self, message: str) -> None:
        self._logging_backend.system_info(message)
        
        # system_info is called before arguments are loaded, so we need to check if self._args is not None before accessing it
        if self._args and self._args.get("logger", {}).get("fnxml", False):
            self._fnxml_backend.system_info(message)

    def system_exception(self, message: str) -> None:
        self._logging_backend.system_exception(message)
        if self._args.get("logger", {}).get("fnxml", False):
            self._fnxml_backend.system_exception(message)

    def user_info(self, message: str) -> None:
        self._logging_backend.user_info(message)
        if self._args.get("logger", {}).get("fnxml", False):
            self._fnxml_backend.user_info(message)

    def user_warning(self, message: str) -> None:
        self._logging_backend.user_warning(message)
        if self._args.get("logger", {}).get("fnxml", False):
            self._fnxml_backend.user_warning(message)

    def user_exception(self, message: str) -> None:
        self._logging_backend.user_exception(message)
        if self._args.get("logger", {}).get("fnxml", False):
            self._fnxml_backend.user_exception(message)

    def write_config(self, message: str) -> None:

        colors = [
            Fore.LIGHTCYAN_EX,
            Fore.LIGHTBLUE_EX,
            Fore.LIGHTMAGENTA_EX,
            Fore.LIGHTGREEN_EX,
        ]

        flat_config = self._flatten_dict(self._args)

        for k, v in flat_config.items():
            parts = k.split("/")
            colored_parts = []

            for i, part in enumerate(parts):
                color = colors[i % len(colors)]
                colored_parts.append(f"{color}{part}{Style.RESET_ALL}")
            

            if hasattr(self._logging_backend, "write_config"):
                self._logging_backend.write_config(f"{'/'.join(colored_parts)}: {v}")
            else:
                self._logging_backend.user_info(f"{'/'.join(colored_parts)}: {v}")

        if hasattr(self._logging_backend, "flush_config_table"):
            self._logging_backend.flush_config_table()
        
        if self._args.get("logger", {}).get("fnxml", False):
            self._fnxml_backend.write_config(message)

    # --------------------------
    # lifecycle
    # --------------------------
    def start(self) -> None:
        self._args = self._parser.args
        self._logging_backend.start(self._args)

        if self._args.get("logger", {}).get("fnxml", False):
            self._fnxml_backend.start(self._args)

        if self._args.get("wandb"):
            self._wandb_backend.start(self._args)

        if self._args.get("tensorboard"):
            self._tensorboard_backend.start(self._args)

    def stop(self) -> None:
        # stop external backends first, then restore print
        self._logging_backend.stop()

        if self._args.get("wandb"):
            self._wandb_backend.stop()
        if self._args.get("tensorboard"):
            self._tensorboard_backend.stop()    
        if self._args.get("logger", {}).get("fnxml", False):
            self._fnxml_backend.stop()

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
        """Recursively flattens a nested dictionary."""

        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    Logger._flatten_dict(v, new_key, sep=sep).items()
                )
            else:
                items.append((new_key, v))

        return dict(items)
    
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

    @property
    def fn_log_file(self) -> Optional[Path]:
        return self._fnxml_backend.log_file
