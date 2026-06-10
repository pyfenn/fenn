"""FnXML file discovery and parsing for the Fenn dashboard."""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

# Default directories to scan (resolved at runtime)
_DEFAULT_DIRS = [
    "./logger",
    "~/logger",
    "~/.fenn/logs",
]

# A session whose .fn file is missing the closing tag and has not been touched
# for this many seconds is treated as crashed rather than still running.
_DEFAULT_RUNNING_TIMEOUT_S = 300

# Short TTL for the directory listing — absorbs request bursts (e.g. session
# pages auto-refreshing) without making the dashboard feel stale.
_FILES_CACHE_TTL_S = 1.0

_VALID_STATUSES = ("running", "crashed", "completed", "failed")
_VALID_SORT_FIELDS = (
    "started",
    "ended",
    "duration_s",
    "warning_count",
    "exception_count",
)


def _running_timeout_s() -> float:
    """Read the running-session timeout from FENN_RUNNING_TIMEOUT_S, with fallback."""
    raw = os.environ.get("FENN_RUNNING_TIMEOUT_S")
    if not raw:
        return float(_DEFAULT_RUNNING_TIMEOUT_S)
    try:
        v = float(raw)
        return v if v > 0 else float(_DEFAULT_RUNNING_TIMEOUT_S)
    except ValueError:
        return float(_DEFAULT_RUNNING_TIMEOUT_S)


class FennScanner:
    """Discovers and parses .fn log files from configured directories."""

    def __init__(self, extra_dirs: Optional[List[str]] = None) -> None:
        self._dirs: List[Path] = []

        # Parse-result cache keyed by file path. Stores (mtime, parsed_dict).
        # Entries are reused only when mtime matches, so any file write
        # (including in-progress logging) invalidates naturally.
        self._parse_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

        # Directory-listing cache: (timestamp, files). Reused for up to
        # _FILES_CACHE_TTL_S to absorb bursty requests cheaply.
        self._files_cache: Optional[Tuple[float, List[Path]]] = None

        # Load from environment variable
        env_dirs = os.environ.get("FENN_LOG_DIRS", "")
        if env_dirs:
            for d in env_dirs.split(":"):
                self._add_dir(d)

        # Add defaults
        for d in _DEFAULT_DIRS:
            self._add_dir(d)

        # Add any extra dirs passed explicitly
        if extra_dirs:
            for d in extra_dirs:
                self._add_dir(d)

    def add_dirs(self, dirs: List[str]) -> None:
        for d in dirs:
            self._add_dir(d)

    def _add_dir(self, path: str) -> None:
        p = Path(path).expanduser().resolve()
        if p not in self._dirs:
            self._dirs.append(p)
        # Invalidate the listing cache so the new directory is picked up.
        self._files_cache = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_fn_files(self) -> List[Path]:
        """Return all .fn files sorted by modification time (newest first)."""
        now = time.time()
        if self._files_cache is not None:
            ts, cached = self._files_cache
            if now - ts < _FILES_CACHE_TTL_S:
                return cached

        files: List[Path] = []
        for d in self._dirs:
            if d.exists() and d.is_dir():
                files.extend(d.rglob("*.fn"))
        seen = set()
        unique = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        ordered = sorted(unique, key=lambda path: path.stat().st_mtime, reverse=True)
        self._files_cache = (now, ordered)
        return ordered

    def parse_fn_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Parse a single .fn file. Handles incomplete (running) sessions.

        Results are cached by ``(path, mtime)``; identical files are not
        re-parsed across requests. Running sessions are re-evaluated against
        the configured timeout on every call, since their "crashed" flip
        depends on wall-clock time rather than file content.
        """
        key = str(path)
        try:
            stat = path.stat()
        except OSError:
            self._parse_cache.pop(key, None)
            return None
        mtime = stat.st_mtime

        cached = self._parse_cache.get(key)
        if cached is not None and cached[0] == mtime:
            return self._refresh_running_status(cached[1], mtime)

        parsed = self._parse_uncached(path, stat)
        if parsed is None:
            self._parse_cache.pop(key, None)
            return None
        self._parse_cache[key] = (mtime, parsed)
        return self._refresh_running_status(parsed, mtime)

    @staticmethod
    def _parse_uncached(path: Path, stat: os.stat_result) -> Optional[Dict[str, Any]]:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, PermissionError):
            return None

        status = "completed"

        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError:
            # Session may still be running — try appending the closing tag
            try:
                root = ElementTree.fromstring(content + "\n</fenn-log>")
                status = "running"
            except ElementTree.ParseError:
                return None

        # Override status from <meta> if present
        meta = root.find("meta")
        if meta is not None:
            status = meta.get("status", "completed")

        # Config
        config: Dict[str, str] = {}
        config_el = root.find("config")
        if config_el is not None:
            for item in config_el.findall("item"):
                k = item.get("key", "")
                v = item.get("value", "")
                if k:
                    config[k] = v

        # Log entries
        entries = []
        for entry in root.findall("entry"):
            entries.append(
                {
                    "ts": entry.get("ts", ""),
                    "kind": entry.get("kind", ""),
                    "level": entry.get("level", ""),
                    "message": entry.text or "",
                }
            )

        # Timing from <meta>
        ended: Optional[str] = None
        duration_s: Optional[int] = None
        if meta is not None:
            ended = meta.get("ended")
            try:
                duration_s = int(meta.get("duration_s", 0))
            except (ValueError, TypeError):
                pass

        warnings = sum(1 for e in entries if e["level"] == "warning")
        exceptions = sum(1 for e in entries if e["level"] == "exception")

        return {
            "session_id": root.get("session_id", path.stem),
            "project": root.get("project", path.parent.name),
            "started": root.get("started", ""),
            "ended": ended,
            "duration_s": duration_s,
            "status": status,
            "config": config,
            "entries": entries,
            "entry_count": len(entries),
            "warning_count": warnings,
            "exception_count": exceptions,
            "file_path": str(path),
            "file_size": stat.st_size,
            "file_mtime": stat.st_mtime,
        }

    @staticmethod
    def _refresh_running_status(parsed: Dict[str, Any], mtime: float) -> Dict[str, Any]:
        """Re-evaluate stale "running" sessions against the current timeout.

        Cached results are reused across requests, but the running→crashed
        flip is time-dependent. Caller-side staleness is cheap to check and
        keeps the cache safe to share across calls.
        """
        if parsed.get("status") == "running" and mtime > 0:
            if time.time() - mtime > _running_timeout_s():
                return {**parsed, "status": "crashed"}
        return parsed

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Return all parsed sessions, newest first."""
        sessions = []
        for path in self.find_fn_files():
            parsed = self.parse_fn_file(path)
            if parsed:
                sessions.append(parsed)
        return sessions

    @staticmethod
    def _build_projects_list(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate per-project stats from an already-loaded sessions list."""
        projects: Dict[str, Dict[str, Any]] = {}
        for s in sessions:
            name = s["project"]
            if name not in projects:
                projects[name] = {
                    "name": name,
                    "session_count": 0,
                    "running_count": 0,
                    "crashed_count": 0,
                    "warning_count": 0,
                    "exception_count": 0,
                    "last_active": s["started"],
                }
            p = projects[name]
            p["session_count"] += 1
            p["warning_count"] += s["warning_count"]
            p["exception_count"] += s["exception_count"]
            if s["status"] == "running":
                p["running_count"] += 1
            elif s["status"] == "crashed":
                p["crashed_count"] += 1
        return sorted(
            projects.values(), key=lambda project: project["last_active"], reverse=True
        )

    def get_overview(self) -> Dict[str, Any]:
        """Aggregate stats for the dashboard home page."""
        sessions = self.get_all_sessions()
        project_list = self._build_projects_list(sessions)

        return {
            "projects": project_list,
            "recent_sessions": sessions[:20],
            "total_sessions": len(sessions),
            "total_projects": len(project_list),
            "total_warnings": sum(s["warning_count"] for s in sessions),
            "total_exceptions": sum(s["exception_count"] for s in sessions),
            "running_sessions": sum(1 for s in sessions if s["status"] == "running"),
            "crashed_sessions": sum(1 for s in sessions if s["status"] == "crashed"),
            "active_page": "home",
        }

    def get_project(self, project_name: str) -> Dict[str, Any]:
        """Return all sessions for a specific project."""
        all_sessions = self.get_all_sessions()
        sessions = [s for s in all_sessions if s["project"] == project_name]

        return {
            "projects": self._build_projects_list(all_sessions),
            "project_name": project_name,
            "sessions": sessions,
            "total_sessions": len(sessions),
            "running_sessions": sum(1 for s in sessions if s["status"] == "running"),
            "crashed_sessions": sum(1 for s in sessions if s["status"] == "crashed"),
            "total_warnings": sum(s["warning_count"] for s in sessions),
            "total_exceptions": sum(s["exception_count"] for s in sessions),
            "active_page": "project",
            "active_project": project_name,
        }

    def get_session(
        self, project_name: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return full data for a single session.

        Fenn writes ``<session_id>.fn`` so the filename stem matches the id;
        we prefer those candidates and fall back to a full scan only if the
        convention does not hold (e.g. user-supplied legacy files).
        """
        all_files = self.find_fn_files()
        candidates = [p for p in all_files if p.stem == session_id] or all_files
        for path in candidates:
            parsed = self.parse_fn_file(path)
            if (
                parsed
                and parsed["project"] == project_name
                and parsed["session_id"] == session_id
            ):
                overview = self.get_overview()
                return {
                    **parsed,
                    "projects": overview["projects"],
                    "active_page": "session",
                    "active_project": project_name,
                }
        return None

    # ------------------------------------------------------------------
    # Filtered / paginated listing
    # ------------------------------------------------------------------

    def list_sessions(
        self,
        project: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort: str = "-started",
    ) -> Dict[str, Any]:
        """Return a filtered, sorted, paginated slice of sessions.

        ``sort`` is a field name optionally prefixed with ``-`` for descending
        order. Valid fields: started, ended, duration_s, warning_count,
        exception_count. Status must be one of running/crashed/completed/failed.

        Raises ``ValueError`` on invalid status or sort field; callers are
        responsible for converting these into HTTP-shaped errors.
        """
        if status is not None and status not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {list(_VALID_STATUSES)}")

        descending = sort.startswith("-")
        field = sort[1:] if descending else sort
        if field not in _VALID_SORT_FIELDS:
            raise ValueError(f"sort field must be one of {list(_VALID_SORT_FIELDS)}")

        sessions = self.get_all_sessions()
        if project:
            sessions = [s for s in sessions if s["project"] == project]
        if status:
            sessions = [s for s in sessions if s["status"] == status]

        # Sorts None values last (ascending) / first (descending) without comparing None to
        # non-None values, but may raise TypeError if non-None values are of different types
        # (e.g., int vs str).
        def sort_key(s: Dict[str, Any]):
            v = s.get(field)
            return v is None, v

        sessions.sort(key=sort_key, reverse=descending)

        total = len(sessions)
        items = sessions[offset : offset + limit]
        # Strip the heavy 'entries' list from the listing payload — clients
        # that need per-entry data fetch the single-session endpoint.
        slim = [{k: v for k, v in s.items() if k != "entries"} for s in items]
        return {
            "items": slim,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @staticmethod
    def format_duration(seconds: Optional[int]) -> str:
        if seconds is None:
            return "—"
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}h {m}m"

    @staticmethod
    def format_size(size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.1f} MB"
