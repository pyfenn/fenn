"""Launch downloaded templates as background processes and track them."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from fenn.dashboard.scanner import FennScanner

# Grace period to catch immediate startup failures (missing deps, syntax
# errors, bad imports) before handing control back to the caller. These
# crashes happen before any .fn log file exists, so the scanner alone can't
# surface them — we have to watch the process directly for a short window.
_DEFAULT_STARTUP_GRACE_S = 0.75

_DEFAULT_ENTRYPOINT = "main.py"
_FENN_YAML_NAME = "fenn.yaml"
_DEFAULT_LOGGER_DIR = "logger"
_STDERR_TRUNCATE = 500


class TemplateLaunchError(Exception):
    """Raised when a template cannot be launched or fails during startup."""


@dataclass
class RunningTemplate:
    """A tracked, launched template process."""

    run_id: str
    template_path: Path
    log_dir: Path
    process: subprocess.Popen
    started_at: float

    def poll(self) -> Optional[int]:
        """Return the exit code if the process has finished, else None."""
        return self.process.poll()


class TemplateRunner:
    """Launches templates via subprocess.Popen and tracks active processes."""

    def __init__(self, startup_grace_s: float = _DEFAULT_STARTUP_GRACE_S) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, RunningTemplate] = {}
        self._startup_grace_s = startup_grace_s

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @staticmethod
    def _read_logger_dir(template_path: Path) -> Path:
        """Read logger.dir from fenn.yaml, resolved relative to the template.

        Falls back to "logger" if the key is absent, matching the shipped
        template default.
        """
        yaml_path = template_path / _FENN_YAML_NAME
        if not yaml_path.exists():
            raise TemplateLaunchError(f"{_FENN_YAML_NAME} not found in {template_path}")

        try:
            content = yaml_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TemplateLaunchError(
                f"Could not read {_FENN_YAML_NAME}: {exc}"
            ) from exc

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise TemplateLaunchError(f"Invalid {_FENN_YAML_NAME}: {exc}") from exc

        if not isinstance(data, dict):
            raise TemplateLaunchError(f"{_FENN_YAML_NAME} must contain a mapping")

        logger_cfg = data.get("logger")
        raw_dir = _DEFAULT_LOGGER_DIR
        if isinstance(logger_cfg, dict):
            configured = logger_cfg.get("dir")
            if isinstance(configured, str) and configured.strip():
                raw_dir = configured.strip()

        return (template_path / raw_dir).resolve()

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(
        self,
        template_path: Path,
        scanner: FennScanner,
        entrypoint: str = _DEFAULT_ENTRYPOINT,
    ) -> RunningTemplate:
        """Launch a template's entrypoint and register its log directory.

        Raises TemplateLaunchError if the template path is invalid, the
        entrypoint is missing, or the process exits within the startup
        grace period (treated as a startup failure).
        """
        template_path = template_path.resolve()
        if not template_path.is_dir():
            raise TemplateLaunchError(
                f"Template path is not a directory: {template_path}"
            )

        entry_path = template_path / entrypoint
        if not entry_path.exists():
            raise TemplateLaunchError(
                f"Entrypoint {entrypoint!r} not found in {template_path}"
            )

        log_dir = self._read_logger_dir(template_path)

        try:
            process = subprocess.Popen(
                [sys.executable, str(entry_path)],
                cwd=str(template_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            raise TemplateLaunchError(f"Failed to start process: {exc}") from exc

        # Watch for a fast crash before any .fn log would exist.
        try:
            process.wait(timeout=self._startup_grace_s)
        except subprocess.TimeoutExpired:
            pass
        exit_code = process.poll()
        if exit_code is not None and exit_code != 0:
            _, stderr = process.communicate()
            raise TemplateLaunchError(
                f"Template exited immediately with code {exit_code}: "
                f"{(stderr or '').strip()[:_STDERR_TRUNCATE]}"
            )

        # Register the log directory only after confirming the process is
        # actually alive, so the scanner never picks up a directory for a
        # template that never started.
        scanner.add_dirs([str(log_dir)])

        run_id = uuid.uuid4().hex
        running = RunningTemplate(
            run_id=run_id,
            template_path=template_path,
            log_dir=log_dir,
            process=process,
            started_at=time.time(),
        )
        with self._lock:
            self._active[run_id] = running
        return running

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def get(self, run_id: str) -> Optional[RunningTemplate]:
        with self._lock:
            return self._active.get(run_id)

    def list_active(self) -> list[RunningTemplate]:
        """Return all tracked processes, pruning any that have since exited."""
        with self._lock:
            finished = [
                rid for rid, rt in self._active.items() if rt.poll() is not None
            ]
            for rid in finished:
                del self._active[rid]
            return list(self._active.values())
