"""TypedDict contracts for Fenn dashboard session and scanner data."""

from typing import Dict, List, Literal, TypedDict

SessionStatus = Literal["running", "crashed", "completed", "failed"]


class LogEntry(TypedDict):
    """Single XML log entry from a Fenn session."""

    ts: str
    kind: str
    level: str
    message: str


class ProjectStats(TypedDict):
    """Aggregated statistics for a project."""

    name: str
    session_count: int
    running_count: int
    crashed_count: int
    warning_count: int
    exception_count: int
    last_active: str


class SessionData(TypedDict):
    """Complete parsed session data from _parse_uncached()."""

    session_id: str
    display_name: str | None
    project: str
    started: str
    ended: str | None
    duration_s: int | None
    status: SessionStatus

    config: Dict[str, str]
    entries: List[LogEntry]

    entry_count: int
    warning_count: int
    exception_count: int

    file_path: str
    file_size: int
    file_mtime: float


class SessionListItem(TypedDict):
    """Lightweight session for paginated listings (entries & config stripped)."""

    session_id: str
    display_name: str | None
    project: str
    started: str
    ended: str | None
    duration_s: int | None
    status: SessionStatus

    entry_count: int
    warning_count: int
    exception_count: int

    file_path: str
    file_size: int
    file_mtime: float


class OverviewPayload(TypedDict):
    """GET /api/overview — dashboard home page response."""

    projects: List[ProjectStats]
    recent_sessions: List[SessionData]

    total_sessions: int
    total_projects: int
    total_warnings: int
    total_exceptions: int

    running_sessions: int
    crashed_sessions: int

    active_page: Literal["home"]


class ProjectPayload(TypedDict):
    """GET /api/project/<name> — project page response."""

    projects: List[ProjectStats]

    project_name: str
    sessions: List[SessionData]

    total_sessions: int
    running_sessions: int
    crashed_sessions: int

    total_warnings: int
    total_exceptions: int

    active_page: Literal["project"]
    active_project: str


class SessionPagePayload(TypedDict):
    """GET /api/project/<name>/session/<id> — session detail response."""

    session_id: str
    display_name: str | None
    project: str
    started: str
    ended: str | None
    duration_s: int | None
    status: SessionStatus

    config: Dict[str, str]
    entries: List[LogEntry]

    entry_count: int
    warning_count: int
    exception_count: int

    file_path: str
    file_size: int
    file_mtime: float

    projects: List[ProjectStats]
    active_page: Literal["session"]
    active_project: str


class SessionListResponse(TypedDict):
    """GET /api/sessions (with filters) — paginated session listing response."""

    items: List[SessionListItem]
    total: int
    limit: int
    offset: int
