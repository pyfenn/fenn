"""Fenn Dashboard — Flask application for browsing fnxml log files."""

from __future__ import annotations

import argparse
import logging
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import werkzeug
from flask import (
    Flask,
    Response,
    abort,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_wtf.csrf import CSRFError, CSRFProtect
from werkzeug.exceptions import HTTPException

from fenn.logging import logger

try:
    from fenn.dashboard import auth as dashboard_auth
    from fenn.dashboard import token_store
    from fenn.dashboard.scanner import FennScanner
except ImportError:  # standalone: python app.py
    import auth as dashboard_auth  # ty: ignore[unresolved-import]
    import token_store  # ty: ignore[unresolved-import]
    from scanner import FennScanner  # ty: ignore[unresolved-import]

_HERE = Path(__file__).parent

app = Flask(
    __name__,
    template_folder=str(_HERE / "templates"),
    static_folder=str(_HERE / "static"),
)

# Fresh per-launch signing key — sessions don't survive a restart, by design.
# We don't need cross-launch persistence and rotating the key auto-invalidates
# every cookie that survived the previous process.
app.secret_key = secrets.token_bytes(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,  # localhost-only — HTTPS not in scope
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    WTF_CSRF_TIME_LIMIT=None,
)

# CSRF on /connect/start and /logout. Even though we listen on 127.0.0.1, any
# tab in the user's browser could POST cross-origin without it.
csrf = CSRFProtect(app)

scanner = FennScanner()


@app.context_processor
def _inject_current_user() -> dict[str, Any]:
    return {"current_user": dashboard_auth.current_user()}


# Endpoints that must remain reachable without a session.
_PUBLIC_ENDPOINTS = frozenset(
    {"connect", "connect_start", "connect_callback", "logout", "static"}
)

_STORED_TOKEN_EXPIRED_MESSAGE = (
    "Your saved dashboard session expired or was revoked. Sign in again to continue."
)
_STORED_TOKEN_OFFLINE_MESSAGE = (
    "Could not reach https://pyfenn.com to verify your saved token. "
    "Using cached identity — the dashboard will revalidate next launch."
)


def _try_stored_session() -> werkzeug.wrappers.response.Response | None:
    """Re-establish a Flask session from ``~/.fenn/dashboard_session.json``.

    Returns ``None`` if the caller should proceed normally, or a Flask
    response if the caller should short-circuit (e.g. redirect to connect
    when the saved token has been revoked server-side).
    """
    stored = token_store.load()
    if stored is None:
        return None

    try:
        user = dashboard_auth.validate_token(stored["token"])
    except dashboard_auth.InvalidTokenError:
        # Server explicitly rejected the saved token — drop it and force a
        # fresh sign-in. Flash a message so the user understands why.
        token_store.clear()
        session.clear()
        session["pending_info"] = _STORED_TOKEN_EXPIRED_MESSAGE
        return redirect(url_for("connect"))
    except dashboard_auth.AuthUnreachableError:
        # Offline or pyfenn.com is down. Fall back to the cached identity so
        # the user can still browse their local logs.
        user = stored["user"]
    else:
        # Server may have updated email; refresh the cached copy.
        token_store.save(stored["token"], user)

    session.clear()
    session["user"] = user
    session.permanent = True
    # current_user() caches on g; invalidate so the rest of this request
    # sees the freshly-loaded identity instead of the prior None.
    g.pop("current_user", None)
    return None


@app.before_request
def _require_login() -> werkzeug.wrappers.response.Response | None:
    endpoint = request.endpoint
    if endpoint in _PUBLIC_ENDPOINTS:
        return None
    if dashboard_auth.current_user() is not None:
        return None
    response = _try_stored_session()
    if response is not None:
        return response
    if dashboard_auth.current_user() is not None:
        return None
    return redirect(url_for("connect"))


@app.errorhandler(CSRFError)
def _csrf_failed(_e: CSRFError) -> tuple[str, int]:
    return (
        render_template(
            "connect.html",
            error_message="Form expired. Please try again.",
            info_message=None,
        ),
        400,
    )


# --------------------------------------------------------------------------- #
# Template filters
# --------------------------------------------------------------------------- #


@app.template_filter("duration")
def duration_filter(seconds: int | None) -> str:
    return scanner.format_duration(seconds)


@app.template_filter("filesize")
def filesize_filter(size: int) -> str:
    return scanner.format_size(size)


@app.template_filter("short_id")
def short_id_filter(session_id: str) -> str:
    return session_id[:8] if len(session_id) > 8 else session_id


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.route("/")
def index() -> str:
    return render_template("index.html", **scanner.get_overview())


@app.route("/project/<project_name>")
def project(project_name: str) -> str:
    return render_template("project.html", **scanner.get_project(project_name))


@app.route("/session/<project_name>/<session_id>", endpoint="session")
def session_view(project_name: str, session_id: str) -> str:
    data = scanner.get_session(project_name, session_id)
    if data is None:
        abort(404)
    return render_template("session.html", **data)


@app.route("/api/overview")
def api_overview() -> Response:
    return jsonify(scanner.get_overview())


@app.route("/api/session/<project_name>/<session_id>")
def api_session(project_name: str, session_id: str) -> Response:
    data = scanner.get_session(project_name, session_id)
    if data is None:
        abort(404)
    data.pop("projects", None)  # ty: ignore[call-non-callable]
    return jsonify(data)


@app.route("/api/session/<project_name>/<session_id>/rename", methods=["POST"])
def api_session_rename(
    project_name: str, session_id: str
) -> tuple[Response, int] | Response:
    payload = request.get_json(silent=True)
    display_name: str | None = None
    if isinstance(payload, dict):
        raw_name = payload.get("display_name")
        if raw_name is not None and not isinstance(raw_name, str):
            return _api_error(
                "invalid_param", "display_name must be a string", "display_name"
            )
        display_name = raw_name
    if display_name is None:
        display_name = request.form.get("display_name")
    if display_name is None:
        return _api_error("invalid_param", "display_name is required", "display_name")

    try:
        ok = scanner.rename_session(project_name, session_id, display_name)
    except ValueError as e:
        return _api_error("invalid_param", str(e), "display_name")

    if not ok:
        abort(404)
    return jsonify({"display_name": display_name.strip()})


@app.route("/api/session/<project_name>/<session_id>/delete", methods=["POST"])
def api_session_delete(
    project_name: str, session_id: str
) -> tuple[Response, int] | Response:
    try:
        ok = scanner.delete_session(project_name, session_id)
    except ValueError as e:
        return _api_error("invalid_param", str(e), "session_id")

    if not ok:
        abort(404)
    return jsonify({"deleted": True})


# Pagination / filtering limits. 200 is large enough for any plausible UI
# without letting a client ask for "everything" by accident.
_MAX_LIMIT = 200
_DEFAULT_LIMIT = 20


def _api_error(
    code: str, message: str, param: str | None = None
) -> tuple[Response, int]:
    """Standard 400 envelope so clients can branch on `error.code`."""
    body = {"error": {"code": code, "message": message}}
    if param is not None:
        body["error"]["param"] = param
    return jsonify(body), 400


class _ApiBadRequest(Exception):
    def __init__(self, message: str, param: str | None = None) -> None:
        self.message = message
        self.param = param


def _parse_int_arg(
    name: str, raw: str | None, default: int, min_v: int, max_v: int
) -> int:
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        raise _ApiBadRequest(f"{name} must be an integer", name)
    if v < min_v or v > max_v:
        raise _ApiBadRequest(f"{name} must be between {min_v} and {max_v}", name)
    return v


@app.route("/api/sessions")
def api_sessions() -> tuple[Response, int] | Response:
    """Filtered, sorted, paginated session listing.

    Query params:
        project,
        status,
        limit (1..200, default 20),
        offset (>=0, default 0),
        sort (field, optionally ``-`` prefixed for descending),
        started_after, started_before (timestamps formatted as ``YYYY-MM-DD HH:MM:SS``).
    """
    try:
        project_name = request.args.get("project") or None
        status = request.args.get("status") or None
        sort = request.args.get("sort") or "-started"
        limit = _parse_int_arg(
            "limit", request.args.get("limit"), _DEFAULT_LIMIT, 1, _MAX_LIMIT
        )
        offset = _parse_int_arg("offset", request.args.get("offset"), 0, 0, 1_000_000)

        started_after = None
        started_after_raw = request.args.get("started_after") or None
        if started_after_raw is not None:
            try:
                started_after = datetime.strptime(
                    started_after_raw, "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                raise _ApiBadRequest(
                    "started_after must be formatted as YYYY-MM-DD HH:MM:SS",
                    "started_after",
                )

        started_before = None
        started_before_raw = request.args.get("started_before") or None
        if started_before_raw is not None:
            try:
                started_before = datetime.strptime(
                    started_before_raw, "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                raise _ApiBadRequest(
                    "started_before must be formatted as YYYY-MM-DD HH:MM:SS",
                    "started_before",
                )

        try:
            result = scanner.list_sessions(
                project=project_name,
                status=status,
                started_after=started_after,
                started_before=started_before,
                limit=limit,
                offset=offset,
                sort=sort,
            )
        except ValueError as e:
            # Pick the parameter name from the message so the envelope is
            # consistent with the int-parsing errors above.
            msg = str(e)
            param = "status" if msg.startswith("status") else "sort"
            return _api_error("invalid_param", msg, param)

        return jsonify(result)
    except _ApiBadRequest as e:
        return _api_error("invalid_param", e.message, e.param)


@app.errorhandler(404)
def not_found(_e: HTTPException) -> tuple[str, int]:
    return render_template("404.html", **scanner.get_overview()), 404


# --------------------------------------------------------------------------- #
# Auth routes
# --------------------------------------------------------------------------- #

_NETWORK_ERROR_MESSAGE = (
    "Could not reach https://pyfenn.com. Verify your internet connection "
    "or open an issue at https://github.com/pyfenn/fenn/issues."
)
_INVALID_TOKEN_MESSAGE = "Sign-in failed or was declined. Try connecting again."
_STATE_MISMATCH_MESSAGE = (
    "The sign-in response didn't match this session. Start again from this page."
)


@app.route("/connect")
def connect():
    """Landing page: a button that starts the pyfenn.com browser sign-in."""
    # Already signed in → bounce to home.
    if dashboard_auth.current_user() is not None:
        return redirect(url_for("index"))

    # One-shot info message set by the auth gate (e.g. "your saved token
    # expired"). Pop so it doesn't persist past this render.
    info_message = session.pop("pending_info", None)
    return render_template(
        "connect.html", error_message=None, info_message=info_message
    )


@app.route("/connect/start", methods=["POST"])
def connect_start():
    """Kick off the browser-OAuth pairing: redirect to pyfenn.com/dashboard/link."""
    # A per-attempt random value round-tripped through pyfenn.com so the
    # callback can prove the response belongs to a sign-in we initiated.
    state = secrets.token_urlsafe(32)
    session["oauth_state"] = state

    redirect_uri = url_for("connect_callback", _external=True)
    query = urlencode({"redirect_uri": redirect_uri, "state": state})
    return redirect(f"{dashboard_auth.AUTH_URL}{dashboard_auth.LINK_PATH}?{query}")


@app.route("/connect/callback")
def connect_callback():
    """Receive the one-time code from pyfenn.com and exchange it for a session."""
    expected_state = session.pop("oauth_state", None)
    got_state = request.args.get("state", "")
    if not expected_state or not secrets.compare_digest(expected_state, got_state):
        return render_template(
            "connect.html",
            error_message=_STATE_MISMATCH_MESSAGE,
            info_message=None,
        ), 400

    code = request.args.get("code", "")
    try:
        result = dashboard_auth.exchange_code(code)
    except dashboard_auth.InvalidTokenError:
        return render_template(  # type: ignore[return-value]
            "connect.html",
            error_message=_INVALID_TOKEN_MESSAGE,
            info_message=None,
        ), 401
    except dashboard_auth.AuthUnreachableError:
        return render_template(  # type: ignore[return-value]
            "connect.html",
            error_message=_NETWORK_ERROR_MESSAGE,
            info_message=None,
        ), 503

    user = {"user_id": result["user_id"], "email": result["email"]}
    # Session fixation defence: rotate the session ID before binding the
    # user, matching the pattern simple-server uses on OAuth callback.
    session.clear()
    session["user"] = user
    session.permanent = True
    # Persist the auto-provisioned token so the next launch skips sign-in and
    # re-validates silently against /api/dashboard/me.
    token_store.save(result["token"], user)
    return redirect(url_for("index"))


@app.route("/logout", methods=["POST"])
def logout() -> werkzeug.wrappers.response.Response:
    session.clear()
    # Also drop the disk cache — otherwise the next request would
    # silently re-establish the same session via _try_stored_session().
    token_store.clear()
    return redirect(url_for("connect"))


# --------------------------------------------------------------------------- #
# Public API (used by CLI)
# --------------------------------------------------------------------------- #


def run(
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
    log_dirs: list[str] | None = None,
) -> None:
    """Configure and start the dashboard server."""
    if log_dirs:
        scanner.add_dirs(log_dirs)
    log_level = logging.DEBUG if debug else logging.INFO
    app.logger.setLevel(log_level)
    logger.setLevel(log_level)
    logger.info(f"Fenn dashboard started at http://{host}:{port}")
    from werkzeug.serving import make_server

    make_server(host, port, app).serve_forever()


# --------------------------------------------------------------------------- #
# Standalone entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fenn-dashboard",
        description="Fenn Dashboard — browse fnxml log files in your browser",
    )
    parser.add_argument(
        "--log-dir",
        nargs="+",
        metavar="DIR",
        help="Extra directories to scan for .fn files",
    )
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(host=args.host, port=args.port, debug=args.debug, log_dirs=args.log_dir)


if __name__ == "__main__":
    main()
