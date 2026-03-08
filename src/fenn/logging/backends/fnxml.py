import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class FnXmlBackend:
    """Writes log entries to a simple XML-like .fn file for dashboards."""

    def __init__(self) -> None:
        self._ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        self._log_file: Optional[Path] = None
        self._enabled = False
        self._session_id: Optional[str] = None
        self._project: Optional[str] = None

    # ---- public (system tags) ----
    def system_info(self, message: str) -> None:
        self._write_entry(kind="system", level="info", message=message)

    def system_warning(self, message: str) -> None:
        self._write_entry(kind="system", level="warning", message=message)

    def system_exception(self, message: str) -> None:
        self._write_entry(kind="system", level="exception", message=message)

    # ---- public (user logs: no tags) ----
    def user_info(self, message: str) -> None:
        self._write_entry(kind="user", level="info", message=message)

    def user_warning(self, message: str) -> None:
        self._write_entry(kind="user", level="warning", message=message)

    def user_exception(self, message: str) -> None:
        self._write_entry(kind="user", level="exception", message=message)

    # ---- lifecycle ----
    def start(self, args: Dict[str, Any]) -> None:
        log_root = Path(args["logger"]["dir"]).expanduser()
        log_dir = log_root / Path(args["project"])
        log_filename = f'{args["session_id"]}.fn'
        self._log_file = log_dir / log_filename

        self._session_id = str(args["session_id"])
        self._project = str(args["project"])

        os.makedirs(log_root, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        started = datetime.now().replace(microsecond=0).isoformat(" ")
        with open(self._log_file, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write(
                f'<fenn-log project="{self._escape(self._project)}" '
                f'session_id="{self._escape(self._session_id)}" '
                f'started="{self._escape(started)}">\n'
            )

        self._enabled = True

    def stop(self) -> None:
        if not self._enabled or not self._log_file:
            return

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write("</fenn-log>\n")

        self._enabled = False

    # ---- print capture (optional) ----
    def log_print(self, message: str, timestamp: Optional[datetime] = None) -> None:
        self._write_entry(kind="print", level="info", message=message, timestamp=timestamp)

    @property
    def log_file(self) -> Optional[Path]:
        return self._log_file

    # ---- internal ----
    def _write_entry(
        self,
        *,
        kind: str,
        level: str,
        message: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        if not self._enabled or not self._log_file:
            return

        ts = (timestamp or datetime.now()).replace(microsecond=0).isoformat(" ")
        clean_message = self._escape(self._ansi_escape.sub("", message))

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(
                f'  <entry ts="{self._escape(ts)}" kind="{self._escape(kind)}" '
                f'level="{self._escape(level)}">{clean_message}</entry>\n'
            )

    @staticmethod
    def _escape(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
