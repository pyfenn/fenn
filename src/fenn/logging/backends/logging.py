import builtins
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from resend import templates
from rich.console import Console
from rich.table import Table
from rich.text import Text
from fenn.args import Parser
from colorama import Fore, Style


# ==========================================================
# CUSTOM LOGGING BACKEND (file + print override)
# ==========================================================

class LoggingBackend():
    def __init__(self) -> None:
        self._console = Console()
        self._original_print = builtins.print
        self._log_file: Optional[Path] = None
        self._ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        self._enabled = False
        self._print_sink: Optional[Callable[[str, datetime], None]] = None
        self._config_table: Optional[Table] = None
        self._file_handler_id = None

    # ---- public methods -----

    def info (self, message: str, display_on_terminal : bool = True, write_on_file : bool = True) -> None:
        if write_on_file:
            self._log_print(message, display=False)
        if display_on_terminal:
            self._console.print(f"[bold green][INFO][/bold green] {message}")

    def warning (self, message: str, display_on_terminal : bool = True, write_on_file : bool = True) -> None:
        if write_on_file:
            self._log_print(message, display=False)
        if display_on_terminal:
            self._console.print(f"[bold yellow][WARNING][/bold yellow] {message}")
        
    def exception (self, message: str, display_on_terminal : bool = True, write_on_file : bool = True) -> None:
        if write_on_file:
            self._log_print(message, display=False)
        if display_on_terminal:
            self._console.print(f"[bold red][EXCEPTION][/bold red] {message}")
    
    def debug (self, message: str, display_on_terminal : bool = True, write_on_file : bool = True) -> None:
        if write_on_file:
            self._log_print(message, display=False)
        if display_on_terminal:
            self._console.print(f"[dim][DEBUG][/dim] {message}")
    

    # ---- config table methods ----
    def write_config(self, message: str) -> None:
        if self._config_table is None:
            table = Table(title="")
            table.add_column(f"Configuration file {Parser().config_file} loaded", style="", width=80)
            self._config_table = table
        self._config_table.add_row(Text.from_ansi(f"- {message}"))
        self.info(f"{message}", display_on_terminal=False)

    def flush_config_table(self) -> None:
        if self._config_table is None:
            return
        self._console.print(self._config_table)
        self._config_table = None

    # ---- lifecycle ----
    def start(self, args: Dict[str, Any]) -> None:
        log_root = Path(args["logger"]["dir"])
        log_dir = log_root / Path(args["project"])
        log_filename = f'{args["session_id"]}.log'
        self._log_file = log_dir / log_filename

        os.makedirs(log_root, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        with open(self._log_file, "w", encoding="utf-8") as f:
            f.write("")

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
        display: bool = True,
        level: str = "INFO",
    ) -> None:
        message = sep.join(map(str, objects))

        if self._log_file:
            clean_message = self._ansi_escape.sub("", message)
            timestamp_dt = datetime.now().replace(microsecond=0)
            timestamp = timestamp_dt.isoformat(" ")
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {level} | {clean_message}\n")
            if self._print_sink is not None:
                self._print_sink(clean_message, timestamp_dt)

        if display:
            self._original_print(*objects, sep=sep, end=end, file=file, flush=flush)

    @property
    def log_file(self) -> Optional[Path]:
        return self._log_file

    def set_print_sink(self, sink: Optional[Callable[[str, datetime], None]]) -> None:
        self._print_sink = sink
