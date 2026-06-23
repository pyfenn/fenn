"""pyfenn.com auth gate for the local Fenn log dashboard.

The dashboard is a localhost-only Flask app. To authenticate, the user
generates a "dashboard token" on pyfenn.com, pastes it into the
``/connect`` page, and the server validates it once against
``https://pyfenn.com/api/dashboard/me``. On success, we store
``{user_id, email}`` in a signed Flask session cookie and discard the
token — it is never written to disk or kept in memory.
"""

from __future__ import annotations

import re
from functools import wraps
from typing import Optional

import requests
from flask import g, redirect, session, url_for

from fenn.utils.logging import logger

AUTH_URL = "https://pyfenn.com"
ME_PATH = "/api/dashboard/me"

# Mirror of the regex in simple-server's app/server/dashboard_auth.py.
# Update both sides together if the token format changes.
_TOKEN_RE = re.compile(r"^fdt_[A-Za-z0-9_-]{43}$")
_MAX_TOKEN_LEN = 64
_MAX_RESPONSE_BYTES = 4096
_TIMEOUT = (5, 10)  # (connect, read) seconds


class InvalidTokenError(Exception):
    """Token was rejected by pyfenn.com (401 or malformed)."""


class AuthUnreachableError(Exception):
    """pyfenn.com could not be reached or returned an unexpected response."""


def current_user() -> Optional[dict]:
    """Return the logged-in user dict, or ``None``. Cached on ``g``."""
    if "current_user" in g:
        return g.current_user
    g.current_user = session.get("user")
    return g.current_user


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if current_user() is None:
            return redirect(url_for("connect"))
        return view(*args, **kwargs)

    return wrapper


def validate_token(token: str) -> dict:
    """Validate a dashboard token against pyfenn.com.

    Returns ``{"user_id": ..., "email": ...}`` on success.
    Raises :class:`InvalidTokenError` if pyfenn.com says the token is bad,
    or :class:`AuthUnreachableError` if the network call fails or the
    response is not shaped as expected.
    """
    if not token:
        raise InvalidTokenError("empty token")

    # Trim incidental whitespace from paste, then enforce length + format
    # *before* sending anything over the wire — this prevents accidentally
    # exfiltrating arbitrary pasted content as a Bearer header.
    candidate = token.strip()
    if len(candidate) > _MAX_TOKEN_LEN or not _TOKEN_RE.match(candidate):
        raise InvalidTokenError("malformed token")

    try:
        response = requests.get(
            AUTH_URL + ME_PATH,
            headers={"Authorization": f"Bearer {candidate}"},
            timeout=_TIMEOUT,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        # Never include the candidate in the log — even on failure.
        logger.warning("dashboard-auth: pyfenn.com unreachable: %s", type(exc).__name__)
        raise AuthUnreachableError(str(exc)) from exc

    if response.status_code == 401:
        raise InvalidTokenError("token rejected")
    if response.status_code != 200:
        logger.warning("dashboard-auth: unexpected status %d", response.status_code)
        raise AuthUnreachableError(f"unexpected status {response.status_code}")

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("application/json"):
        raise AuthUnreachableError("unexpected response content-type")
    if len(response.content) > _MAX_RESPONSE_BYTES:
        raise AuthUnreachableError("response too large")

    try:
        body = response.json()
    except ValueError as exc:
        raise AuthUnreachableError("malformed JSON response") from exc

    user_id = body.get("user_id")
    email = body.get("email")
    if not isinstance(user_id, str) or not isinstance(email, str):
        raise AuthUnreachableError("missing user_id or email in response")

    return {"user_id": user_id, "email": email}
