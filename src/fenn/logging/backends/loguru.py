import builtins
import os
import sys
from pathlib import Path
from loguru import logger
from typing import Any, Dict, Optional
from fenn.logging.backends.baseLogger import baseLogger


class LoguruBackend(baseLogger):
    def __init__(self) -> None:
        self._original_print = builtins.print
        self._log_file: Optional[Path] = None
        self._file_handler_id: Optional[int] = None
        self._enabled = False

    # ---- public (system tags) ----
    def system_info(self, message: str) -> None:
        logger.info(message)

    def system_warning(self, message: str) -> None:
        logger.warning(message)

    def system_exception(self, message: str) -> None:
        logger.exception(message)

    # ---- public (user logs: no tags) ----
    def user_info(self, message: str) -> None:
        logger.info(message)

    def user_warning(self, message: str) -> None:
        logger.warning(message)

    def user_exception(self, message: str) -> None:
        logger.exception(message)

    # ---- lifecycle ----
    def start(self, args: Dict[str, Any]) -> None:
        log_root = Path(args["logger"]["dir"]).expanduser()
        log_dir = log_root / Path(args["project"])
        log_filename = f'{args["session_id"]}.log'
        self._log_file = log_dir / log_filename

        os.makedirs(log_root, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        logger.remove()  # evita duplicados
        logger.add(sys.stdout, level="INFO")
        self._file_handler_id = logger.add(
            self._log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            mode="w",
        )

        self._enabled = True
        self.system_info(f"Logging file {log_filename} created in {log_dir}")

    def stop(self) -> None:
        if self._enabled:
            logger.remove()
            self._file_handler_id = None
            self._enabled = False

    @property
    def log_file(self) -> Optional[Path]:
        return self._log_file