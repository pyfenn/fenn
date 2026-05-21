"""Fenn Dashboard — Flask application for browsing fnxml log files."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from flask import Flask, abort, jsonify, render_template, request

try:
    from fenn.dashboard.scanner import FennScanner
except ImportError:  # standalone: python app.py
    from scanner import FennScanner  # type: ignore[no-redef]

_HERE = Path(__file__).parent

app = Flask(
    __name__,
    template_folder=str(_HERE / "templates"),
    static_folder=str(_HERE / "static"),
)

scanner = FennScanner()


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


@app.route("/session/<project_name>/<session_id>")
def session(project_name: str, session_id: str):
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
        raise _ApiBadRequest(
            f"{name} must be between {min_v} and {max_v}", name
        )
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
        project = request.args.get("project") or None
        status = request.args.get("status") or None
        sort = request.args.get("sort") or "-started"
        limit = _parse_int_arg(
            "limit", request.args.get("limit"), _DEFAULT_LIMIT, 1, _MAX_LIMIT
        )
        offset = _parse_int_arg(
            "offset", request.args.get("offset"), 0, 0, 1_000_000
        )

        try:
            result = scanner.list_sessions(
                project=project,
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
def not_found(e):
    return render_template("404.html", **scanner.get_overview()), 404


# --------------------------------------------------------------------------- #
# Public API (used by CLI)
# --------------------------------------------------------------------------- #


def run(
    host: str = "127.0.0.1", port: int = 5000, debug: bool = False, log_dirs=None
) -> None:
    """Configure and start the dashboard server."""
    if log_dirs:
        scanner.add_dirs(log_dirs)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.logger.setLevel(logging.ERROR)
    print(f"Fenn dashboard started at http://{host}:{port}")
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
