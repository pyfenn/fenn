"""HTTP client for the Fenn remote execution service (``/v1/*`` API)."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import requests

from fenn.exceptions import (
    AuthError,
    InsufficientCreditsError,
    NetworkError,
    RemoteError,
)

DEFAULT_REMOTE_HOST = os.environ.get("FENN_REMOTE_HOST", "https://pyfenn.com")

_CONNECT_TIMEOUT = 10
_REQUEST_TIMEOUT = 60
# SSE read timeout must survive quiet stretches of a running job; the server
# sends keep-alive comments, so a couple of minutes is plenty.
_STREAM_TIMEOUT = 180


def _parse_sse(response) -> Iterator[dict]:
    """Yield ``{"event": ..., "data": ...}`` dicts from an SSE response.

    ``data`` is JSON-decoded when possible, otherwise the raw string.
    Comment/keep-alive lines (starting with ``:``) are skipped.
    """
    event: Optional[str] = None
    data_lines: list[str] = []

    for raw in response.iter_lines(decode_unicode=True):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if raw is None:
            continue
        line = raw

        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    data = payload
                yield {"event": event or "message", "data": data}
            event = None
            data_lines = []
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())

    if data_lines:
        payload = "\n".join(data_lines)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = payload
        yield {"event": event or "message", "data": data}


class RemoteClient:
    """Thin ``requests`` wrapper with bearer auth and typed errors."""

    def __init__(
        self,
        host: str,
        api_key: str,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._session = session or requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers.setdefault("User-Agent", "fenn-cli")

    # -- lifecycle ----------------------------------------------------------

    def __enter__(self) -> "RemoteClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    # -- plumbing -----------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self._host}{path}"

    def _raise_for_status(self, response) -> None:
        status = getattr(response, "status_code", 0)
        if status < 400:
            return
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = None
        if not detail:
            detail = getattr(response, "text", "") or getattr(response, "reason", "")

        if status in (401, 403):
            raise AuthError(f"HTTP {status}: {detail or 'invalid or revoked API key'}")
        if status == 402:
            raise InsufficientCreditsError(str(detail or "not enough credits"))
        raise RemoteError(f"HTTP {status}: {detail}")

    def _request(self, method: str, path: str, **kwargs):
        kwargs.setdefault("timeout", (_CONNECT_TIMEOUT, _REQUEST_TIMEOUT))
        try:
            response = self._session.request(method, self._url(path), **kwargs)
        except requests.exceptions.SSLError:
            raise
        except requests.exceptions.ConnectionError:
            raise
        except requests.exceptions.Timeout as exc:
            raise NetworkError(f"Request to {self._host} timed out: {exc}") from exc
        self._raise_for_status(response)
        return response

    # -- API ----------------------------------------------------------------

    def me(self) -> dict:
        """Return the authenticated account: credits, plan, rate card."""
        return self._request("GET", "/v1/me").json()

    def submit_job(
        self,
        workspace_tar: Path,
        *,
        script: str,
        max_runtime: int,
        project: Optional[str] = None,
        venv: Optional[dict] = None,
        machine_class: Optional[str] = None,
    ) -> dict:
        """Upload a workspace and enqueue a job.

        Matches the server contract (``POST /v1/jobs``): a ``tarball`` file
        part plus a ``meta`` JSON form field. Returns the server response,
        e.g. ``{"job_id": ..., "credit_hold": N, "credits_remaining": N}``.
        """
        meta = {
            "script": script,
            "max_runtime": int(max_runtime),
        }
        if project:
            meta["project"] = project
        if venv:
            meta["venv"] = venv
        if machine_class:
            meta["machine_class"] = machine_class

        with open(workspace_tar, "rb") as fh:
            response = self._request(
                "POST",
                "/v1/jobs",
                data={"meta": json.dumps(meta)},
                files={"tarball": ("workspace.tar.gz", fh, "application/gzip")},
                timeout=(_CONNECT_TIMEOUT, _STREAM_TIMEOUT),
            )
        return response.json()

    def get_job(self, job_id: str) -> dict:
        return self._request("GET", f"/v1/jobs/{job_id}").json()

    def cancel(self, job_id: str) -> dict:
        return self._request("DELETE", f"/v1/jobs/{job_id}").json()

    @contextmanager
    def stream_events(self, job_id: str):
        """Context manager yielding an iterator of parsed SSE events."""
        response = self._request(
            "GET",
            f"/v1/jobs/{job_id}/events",
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=(_CONNECT_TIMEOUT, _STREAM_TIMEOUT),
        )
        try:
            yield _parse_sse(response)
        finally:
            response.close()

    def download_artifacts(self, job_id: str, dest: Path) -> Path:
        """Stream the job's artifact tarball to ``dest`` and return it."""
        response = self._request(
            "GET",
            f"/v1/jobs/{job_id}/artifacts",
            stream=True,
            timeout=(_CONNECT_TIMEOUT, _STREAM_TIMEOUT),
        )
        try:
            with open(dest, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        fh.write(chunk)
        finally:
            response.close()
        return dest
