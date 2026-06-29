"""On-disk cache for the pyfenn.com dashboard token + cached identity.

Stored at ``~/.fenn/dashboard_session.json``. On POSIX the file is
chmod'd to ``0600`` (user-only). On Windows the user's home directory is
already user-private under the default ACL, so we accept the default.

The cache holds the same data the server already trusts the user with:
their own dashboard token plus the ``user_id`` / ``email`` that pyfenn.com
returns. No encryption layer — the token is high-entropy and the threat
model matches existing tools that drop credentials in ``$HOME`` (aws,
gcloud, gh, etc.).
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import TypedDict

from fenn.logging import logger

_PATH = Path.home() / ".fenn" / "dashboard_session.json"


class StoredSession(TypedDict):
    token: str
    user: dict  # {"user_id": str, "email": str}


def _is_valid(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    token = payload.get("token")
    user = payload.get("user")
    if not isinstance(token, str) or not token:
        return False
    if not isinstance(user, dict):
        return False
    return isinstance(user.get("user_id"), str) and isinstance(user.get("email"), str)


def load() -> StoredSession | None:
    """Return the stored session, or ``None`` if absent / unreadable."""
    try:
        raw = _PATH.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    except UnicodeDecodeError:
        # Corrupted file — drop it so we don't keep retrying.
        clear()
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        clear()
        return None
    if not _is_valid(data):
        clear()
        return None
    return {
        "token": data["token"],
        "user": {
            "user_id": data["user"]["user_id"],
            "email": data["user"]["email"],
        },
    }


def save(token: str, user: dict) -> None:
    """Persist ``token`` + minimal user identity. Best-effort."""
    payload = {
        "token": token,
        "user": {
            "user_id": user["user_id"],
            "email": user["email"],
        },
    }
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        logger.warning("dashboard-token-store: could not write %s: %s", _PATH, exc)
        return
    # Restrict to owner read/write on POSIX. Windows uses ACLs; the default
    # placement under ~/.fenn is already user-private.
    try:
        os.chmod(_PATH, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def clear() -> None:
    """Remove the stored session. No-op if it doesn't exist."""
    try:
        _PATH.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("dashboard-token-store: could not delete %s: %s", _PATH, exc)


def path() -> Path:
    """Expose the storage path (useful for tests + diagnostics)."""
    return _PATH
