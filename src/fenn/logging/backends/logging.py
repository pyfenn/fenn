import builtins
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from fenn.logging.backends.baseLogger import baseLogger

from colorama import Fore, Style

try:
    from rich.console import Console
    _rich_console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ==========================================================
# CUSTOM LOGGING BACKEND (file + print override)
# ==========================================================
class LoggingBackend(baseLogger):
    def __init__(self) -> None:
        self._original_print = builtins.print
        self._log_file: Optional[Path] = None
        self._ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        self._enabled = False
        self._print_sink: Optional[Callable[[str, datetime], None]] = None

    # ---- public (system tags) ----
    def system_info(self, message: str) -> None:
        if HAS_RICH:
            _rich_console.print(f"[bold green][INFO][/bold green] {message}")
        else:
            tag = f"{Fore.GREEN}[INFO]{Style.RESET_ALL}"
            self._system_print(f"{tag} {message}")

    def system_warning(self, message: str) -> None:
        if HAS_RICH:
            _rich_console.print(f"[bold yellow][WARNING][/bold yellow] {message}")
        else:
            tag = f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL}"
            self._system_print(f"{tag} {message}")

    def system_exception(self, message: str) -> None:
        if HAS_RICH:
            _rich_console.print(f"[bold red][EXCEPTION][/bold red] {message}")
        else:
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
            timestamp_dt = datetime.now().replace(microsecond=0)
            timestamp = timestamp_dt.isoformat(" ")
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {clean_message}\n")
            if self._print_sink is not None:
                self._print_sink(clean_message, timestamp_dt)

        self._original_print(*objects, sep=sep, end=end, file=file, flush=flush)

    @property
    def log_file(self) -> Optional[Path]:
        return self._log_file

    def set_print_sink(self, sink: Optional[Callable[[str, datetime], None]]) -> None:
        self._print_sink = sink
