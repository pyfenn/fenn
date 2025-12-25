import builtins
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from colorama import Fore, Style

# ==========================================================
# CUSTOM LOGGING BACKEND (file + print override)
# ==========================================================
class LoggingBackend:
    def __init__(self) -> None:
        self._original_print = builtins.print
        self._log_file: Optional[Path] = None
        self._ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        self._enabled = False

    # ---- public (system tags) ----
    def system_info(self, message: str) -> None:
        tag = f"{Fore.GREEN}[INFO]{Style.RESET_ALL}"
        self._system_print(f"{tag} {message}")

    def system_warning(self, message: str) -> None:
        tag = f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL}"
        self._system_print(f"{tag} {message}")

    def system_exception(self, message: str) -> None:
        tag = f"{Fore.RED}[EXCEPTION]{Style.RESET_ALL}"
        self._system_print(f"{tag} {message}")

    # ---- public (user logs: no tags) ----
    def user_info(self, message: str) -> None:
        self._log_print(message)

    def user_warning(self, message: str) -> None:
        self._log_print(message)

    def user_exception(self, message: str) -> None:
        self._log_print(message)

    # ---- lifecycle ----
    def start(self, args: Dict[str, Any]) -> None:
        log_root = Path(args["logger"]["dir"])
        log_dir = log_root / Path(args["project"])
        log_filename = f'{args["session_id"]}.log'
        self._log_file = log_dir / log_filename

        os.makedirs(log_root, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # truncate/create
        with open(self._log_file, "w", encoding="utf-8") as f:
            f.write("")

        self.system_info(f"Logging file {log_filename} created in {log_dir}")

        # override global print
        builtins.print = self._log_print
        self._enabled = True

    def stop(self) -> None:
        if self._enabled:
            builtins.print = self._original_print
            self._enabled = False

    # ---- internal ----
    def _system_print(
        self,
        *objects: Any,
        sep: str = " ",
        end: str = "\n",
        file: Optional[Any] = None,
        flush: bool = False,
    ) -> None:
        self._original_print(*objects, sep=sep, end=end, file=file, flush=flush)

    def _log_print(
        self,
        *objects: Any,
        sep: str = " ",
        end: str = "\n",
        file: Optional[Any] = None,
        flush: bool = False,
    ) -> None:
        message = sep.join(map(str, objects))

        if self._log_file:
            clean_message = self._ansi_escape.sub("", message)
            timestamp = datetime.now().replace(microsecond=0).isoformat(" ")
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {clean_message}\n")

        self._original_print(*objects, sep=sep, end=end, file=file, flush=flush)

    @property
    def log_file(self) -> Optional[Path]:
        return self._log_file