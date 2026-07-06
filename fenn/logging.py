import builtins
import logging
import re
from collections.abc import MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from colorama import Fore, Style
from rich.console import Console
from rich.table import Table

console: Console = Console()

_ansi_escape: re.Pattern[str] = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class XmlMixin:
    _started_at: datetime
    fn_file: Path

    def _write_config_xml(
        self, args: dict[str, Any], flat_config: dict[str, Any], log_path: Path
    ) -> None:
        self._init_fnxml(
            args=args,
            fn_xml=log_path,
        )
        self._write_config_params(flat_config, log_path)

    def _write_config_params(self, flat_config: dict[str, Any], log_file: Path) -> None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("  <config>\n")
            for key, value in flat_config.items():
                f.write(
                    f'    <item key="{self._escape(str(key))}" '
                    f'value="{self._escape(str(value))}" />\n'
                )
            f.write("  </config>\n")

    def _escape(self, value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    def _init_fnxml(
        self,
        args: dict[str, Any],
        fn_xml: Path,
    ) -> None:
        with open(fn_xml, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write(
                f'<fenn-log project="{self._escape(args["project"])}" '
                f'session_id="{self._escape(args["session_id"])}" '
                f'started="{self._started_at}">\n'
            )

    def _write_entry(self, record: logging.LogRecord, file_path: Path) -> None:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
        clean_message = self._escape(_ansi_escape.sub("", record.getMessage()))
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(
                f'  <entry ts="{ts.strftime("%Y-%m-%d %H:%M:%S")}" kind="user" '
                f'level="{self._escape(record.levelname)}">{clean_message}</entry>\n'
            )

    def _write_stop_info(self, started_at: datetime) -> None:
        ended_at = datetime.now().replace(microsecond=0)
        ended = ended_at.isoformat(" ")
        duration_s = int((ended_at - started_at).total_seconds()) if started_at else 0

        with open(self.fn_file, "a", encoding="utf-8") as f:
            f.write(
                f'  <meta ended="{self._escape(ended)}" '
                f'duration_s="{duration_s}" '
                f'status="completed" />\n'
            )
            f.write("</fenn-log>\n")


class FennHandler(XmlMixin, logging.Handler):
    _log_file: Path | None
    _fn_xml: Path | None

    def __init__(self, level: int = 0) -> None:
        self._started_at = datetime.now().replace(microsecond=0)
        self._log_file = None
        self._fn_xml = None
        super().__init__(level)

    def configure(self, args: dict[str, Any], started_at: datetime) -> None:
        log_root = Path(args["logger"]["dir"])
        log_dir = log_root / Path(args["project"])
        log_filename = f"{args['session_id']}.log"
        fnxml_filename = f"{args['session_id']}.fn"
        self._log_file = log_dir / log_filename
        self._fn_xml = log_dir / fnxml_filename
        self._started_at = started_at

    def _write_to_log(
        self,
        record: logging.LogRecord,
    ) -> None:
        if self._log_file is None:
            return
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
        clean_message = _ansi_escape.sub("", record.getMessage())
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(
                f"[{ts.strftime('%Y-%m-%d %H:%M:%S')}] {record.levelname} | {clean_message}\n"
            )

    def _write_to_console(
        self,
        record: logging.LogRecord,
    ) -> None:
        clean_message = _ansi_escape.sub("", record.getMessage())
        match record.levelno:
            case logging.DEBUG:
                color = "[dim][DEBUG][/dim]"
            case logging.INFO:
                color = "[bold green][INFO][/bold green]"
            case logging.WARNING:
                color = "[bold yellow][WARNING][/bold yellow]"
            case logging.ERROR:
                color = "[bold red][EXCEPTION][/bold red]"
        console.print(f"{color} | {clean_message}\n")

    def emit(
        self,
        record: logging.LogRecord,
    ) -> None:

        if self._fn_xml and self._log_file:
            self._write_entry(record=record, file_path=self._fn_xml)
            self._write_to_log(record=record)
        if not getattr(record, "skip_console", False):
            self._write_to_console(record=record)


class FennLogger(XmlMixin, logging.LoggerAdapter):
    _started_at: datetime
    handler: "FennHandler"
    fn_file: Path
    txt_file: Path

    def __init__(self, logger: logging.Logger, handler: "FennHandler") -> None:
        super().__init__(logger, {})
        self._started_at = datetime.now().replace(microsecond=0)
        self.handler = handler

    def write_config(self, args: dict[str, Any], config_file: str) -> None:
        self.handler.configure(args, self._started_at)
        self._create_config(args=args, config_file=config_file)

    def close(self) -> None:
        self._write_stop_info(started_at=self._started_at)
        self.handler._log_file = None
        self.handler._fn_xml = None

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        kwargs["extra"] = {**self.extra, **kwargs.get("extra", {})}
        return msg, kwargs

    def _form_log_paths(self, args: dict[str, Any]) -> None:
        log_root = Path(args["logger"]["dir"]).expanduser()
        log_dir = log_root / Path(args["project"])
        log_dir.mkdir(parents=True, exist_ok=True)
        fn_filename = f"{args['session_id']}.fn"
        txt_filename = f"{args['session_id']}.log"
        self.fn_file = log_dir / fn_filename
        self.txt_file = log_dir / txt_filename

    def _create_config(
        self,
        args: dict[str, Any],
        config_file: str,
    ) -> None:
        if not args:
            return
        flat_config = self._flatten_dict(args)
        self._form_log_paths(args=args)
        self._display_config(
            flat_config=flat_config, config_file=config_file, file_path=self.txt_file
        )
        self._write_config_xml(
            args=args, flat_config=flat_config, log_path=self.fn_file
        )

    def _flatten_dict(
        self, d: dict[str, Any], parent_key: str = "", sep: str = "/"
    ) -> dict[str, Any]:
        """Recursively flattens a nested dictionary."""

        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))

        return dict(items)

    def _get_colored_parts(self, key: str) -> list[str]:
        colors = [
            Fore.LIGHTCYAN_EX,
            Fore.LIGHTBLUE_EX,
            Fore.LIGHTMAGENTA_EX,
            Fore.LIGHTGREEN_EX,
        ]
        parts = key.split("/")
        colored_parts = []

        for i, part in enumerate(parts):
            color = colors[i % len(colors)]
            colored_parts.append(f"{color}{part}{Style.RESET_ALL}")
        return colored_parts

    def _display_config(
        self, flat_config: dict[str, Any], config_file: str, file_path: Path
    ) -> None:
        table = Table(title="")
        table.add_column(f"Configuration file {config_file} loaded", style="", width=80)
        timestamp_dt = datetime.now().replace(microsecond=0)
        timestamp = timestamp_dt.isoformat(" ")
        for k, v in flat_config.items():
            colored_parts = self._get_colored_parts(key=k)
            table.add_row(f"{'/'.join(colored_parts)}: {v}")
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] INFO | {k}: {v}\n")
        console.print(table)


fenn_handler: FennHandler = FennHandler()
base: logging.Logger = logging.getLogger("__name__")
base.addHandler(fenn_handler)

logger: FennLogger = FennLogger(base, fenn_handler)

logger.setLevel(logging.DEBUG)


def _custom_print(*args: Any, **kwargs: Any) -> None:
    logger.info(" ".join(str(a) for a in args))


original_print: Any = builtins.print


def redirect_prints() -> None:
    builtins.print = _custom_print


def restore_prints() -> None:
    builtins.print = original_print
