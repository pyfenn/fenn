"""Fenn Dashboard — Flask application for browsing fnxml log files."""

import argparse
import logging
import secrets
from datetime import timedelta
from pathlib import Path
from typing import Optional

from flask import (
    Flask,
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

from fenn.utils.logging import logger

try:
    from fenn.dashboard import auth as dashboard_auth
    from fenn.dashboard import token_store
    from fenn.dashboard.scanner import FennScanner
except ImportError:  # standalone: python app.py
    import auth as dashboard_auth  # type: ignore[no-redef]
    import token_store  # type: ignore[no-redef]
    from scanner import FennScanner  # type: ignore[no-redef]

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

# CSRF on /connect and /logout. Even though we listen on 127.0.0.1, any tab in
# the user's browser could POST cross-origin without it.
csrf = CSRFProtect(app)

scanner = FennScanner()


@app.context_processor
def _inject_current_user():
    return {"current_user": dashboard_auth.current_user()}


# Endpoints that must remain reachable without a session.
_PUBLIC_ENDPOINTS = frozenset({"connect", "logout", "static"})

_STORED_TOKEN_EXPIRED_MESSAGE = (
    "Your saved dashboard token expired or was revoked. Paste a fresh one to continue."
)
_STORED_TOKEN_OFFLINE_MESSAGE = (
    "Could not reach https://pyfenn.com to verify your saved token. "
    "Using cached identity — the dashboard will revalidate next launch."
)


def _try_stored_session():
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
        # fresh paste. Flash a message so the user understands why.
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
def _require_login():
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
def _csrf_failed(_e):
    return render_template(
        "connect.html",
        error_message="Form expired. Please try again.",
        info_message=None,
    ), 400


# --------------------------------------------------------------------------- #
# Template filters
# --------------------------------------------------------------------------- #


@app.template_filter("duration")
def duration_filter(seconds):
    return scanner.format_duration(seconds)


@app.template_filter("filesize")
def filesize_filter(size):
    return scanner.format_size(size)


@app.template_filter("short_id")
def short_id_filter(session_id: str) -> str:
    return session_id[:8] if len(session_id) > 8 else session_id


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.route("/")
def index():
    return render_template("index.html", **scanner.get_overview())


@app.route("/project/<project_name>")
def project(project_name: str):
    return render_template("project.html", **scanner.get_project(project_name))


@app.route("/session/<project_name>/<session_id>", endpoint="session")
def session_view(project_name: str, session_id: str):
    data = scanner.get_session(project_name, session_id)
    if data is None:
        abort(404)
    return render_template("session.html", **data)


@app.route("/api/overview")
def api_overview():
    return jsonify(scanner.get_overview())


@app.route("/api/session/<project_name>/<session_id>")
def api_session(project_name: str, session_id: str):
    data = scanner.get_session(project_name, session_id)
    if data is None:
        abort(404)
    data.pop("projects", None)
    return jsonify(data)


# Pagination / filtering limits. 200 is large enough for any plausible UI
# without letting a client ask for "everything" by accident.
_MAX_LIMIT = 200
_DEFAULT_LIMIT = 20


def _api_error(code: str, message: str, param: Optional[str] = None):
    """Standard 400 envelope so clients can branch on `error.code`."""
    body = {"error": {"code": code, "message": message}}
    if param is not None:
        body["error"]["param"] = param
    return jsonify(body), 400


def _parse_int_arg(
    name: str, raw: Optional[str], default: int, min_v: int, max_v: int
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


class _ApiBadRequest(Exception):
    def __init__(self, message: str, param: Optional[str] = None):
        self.message = message
        self.param = param


@app.route("/api/sessions")
def api_sessions():
    """Filtered, sorted, paginated session listing.

    Query params: project, status, limit (1..200, default 20), offset (>=0,
    default 0), sort (field, optionally ``-`` prefixed for descending).
    """
    try:
        project_name = request.args.get("project") or None
        status = request.args.get("status") or None
        sort = request.args.get("sort") or "-started"
        limit = _parse_int_arg(
            "limit", request.args.get("limit"), _DEFAULT_LIMIT, 1, _MAX_LIMIT
        )
        offset = _parse_int_arg("offset", request.args.get("offset"), 0, 0, 1_000_000)

        try:
            result = scanner.list_sessions(
                project=project_name,
                status=status,
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
def not_found(_e):
    return render_template("404.html", **scanner.get_overview()), 404


# --------------------------------------------------------------------------- #
# Auth routes
# --------------------------------------------------------------------------- #

_NETWORK_ERROR_MESSAGE = (
    "Could not reach https://pyfenn.com. Verify your internet connection "
    "or open an issue at https://github.com/pyfenn/fenn/issues."
)
_INVALID_TOKEN_MESSAGE = "Invalid or expired token."


@app.route("/connect", methods=["GET", "POST"])
def connect():
    # Already signed in → bounce to home.
    if dashboard_auth.current_user() is not None:
        return redirect(url_for("index"))

    # One-shot info message set by the auth gate (e.g. "your saved token
    # expired"). Pop so it doesn't persist past this render.
    info_message = session.pop("pending_info", None)

    if request.method == "GET":
        return render_template(
            "connect.html", error_message=None, info_message=info_message
        )

    token = request.form.get("token", "")
    try:
        user = dashboard_auth.validate_token(token)
    except dashboard_auth.InvalidTokenError:
        return render_template(
            "connect.html",
            error_message=_INVALID_TOKEN_MESSAGE,
            info_message=None,
        ), 401
    except dashboard_auth.AuthUnreachableError:
        return render_template(
            "connect.html",
            error_message=_NETWORK_ERROR_MESSAGE,
            info_message=None,
        ), 503

    # Session fixation defence: rotate the session ID before binding the
    # user, matching the pattern simple-server uses on OAuth callback.
    session.clear()
    session["user"] = user
    session.permanent = True
    # Persist so the next dashboard launch skips the paste step.
    token_store.save(token.strip(), user)
    return redirect(url_for("index"))


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    # Also drop the disk cache — otherwise the next request would
    # silently re-establish the same session via _try_stored_session().
    token_store.clear()
    return redirect(url_for("connect"))


# --------------------------------------------------------------------------- #
# Public API (used by CLI)
# --------------------------------------------------------------------------- #


def run(
    host: str = "127.0.0.1", port: int = 5000, debug: bool = False, log_dirs=None
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


def main():
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
