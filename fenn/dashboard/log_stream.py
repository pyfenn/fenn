"""Live-tail a growing plain-text log file, one complete line at a time.

Used to stream a session's raw ``.log`` file (see ``fenn.logging.FennHandler``,
which appends to it line-by-line as a run progresses) into the dashboard's
"Live Terminal" view over a WebSocket.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path

_DEFAULT_POLL_INTERVAL_S = 0.4
_DEFAULT_BACKLOG_LINES = 200


def iter_new_lines(
    path: Path,
    *,
    from_start: bool = False,
    poll_interval: float = _DEFAULT_POLL_INTERVAL_S,
    backlog_lines: int = _DEFAULT_BACKLOG_LINES,
    sleep_fn: Callable[[float], None] = time.sleep,
    should_continue: Callable[[], bool] = lambda: True,
) -> Iterator[str]:
    """Yield complete lines from ``path`` as they are appended.

    If ``from_start`` is True, first yields up to ``backlog_lines`` of
    existing content (tail-style), then polls for new lines by re-reading
    from the last known byte offset. Between polls, ``sleep_fn`` is called
    with ``poll_interval`` — callers can substitute their own pacing
    (e.g. a WebSocket's own receive-with-timeout, which also detects
    client disconnects).

    Stops once ``should_continue()`` returns False, after draining and
    yielding whatever was written since the last poll. Silently stops if
    the file disappears mid-stream.
    """
    try:
        with open(path, "rb") as f:
            initial = f.read()
            offset = f.tell()
    except OSError:
        return

    if from_start:
        lines = initial.decode("utf-8", errors="replace").splitlines()
        for line in lines[-backlog_lines:]:
            yield line

    pending = ""
    while True:
        try:
            with open(path, "rb") as f:
                f.seek(offset)
                raw = f.read()
                offset = f.tell()
        except OSError:
            break

        chunk = pending + raw.decode("utf-8", errors="replace")
        *complete, pending = chunk.split("\n")
        yield from complete

        if not should_continue():
            if pending:
                yield pending
            break

        sleep_fn(poll_interval)
