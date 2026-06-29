import argparse

import fenn.cli.auth as auth
import fenn.cli.dashboard as dashboard
import fenn.cli.grid as grid
import fenn.cli.list as list
import fenn.cli.pull as pull
import fenn.cli.run as run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fenn")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ========= PULL =========
    p_pull = subparsers.add_parser(
        "pull", help="Download a template from the fenn templates repository"
    )

    # --- Level 2 ---
    p_pull.add_argument(
        "template",
        nargs="?",
        help="Name of the template to download (e.g., 'base')",
    )

    p_pull.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Target directory (default: current directory)",
    )

    p_pull.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if needed",
    )

    p_pull.set_defaults(func=pull.execute)

    # ========= LIST =========
    p_list = subparsers.add_parser(
        "list", help="List available templates in the fenn templates repository"
    )
    p_list.set_defaults(func=list.execute)

    # ========= DASHBOARD =========
    p_dash = subparsers.add_parser(
        "dashboard", help="Launch the Fenn log-browser dashboard"
    )
    p_dash.add_argument(
        "--log-dir",
        nargs="+",
        metavar="DIR",
        help="Extra directories to scan for .fn files",
    )
    p_dash.add_argument(
        "--port", type=int, default=5000, help="Port to bind (default: 5000)"
    )
    p_dash.add_argument("--debug", action="store_true", help="Run in debug mode")
    p_dash.set_defaults(func=dashboard.execute)

    # ========= RUN =========
    p_run = subparsers.add_parser(
        "run",
        help="Run a Fenn project on the Fenn remote service",
    )
    p_run.add_argument(
        "script",
        nargs="?",
        default=None,
        help="Path to the entrypoint script (default: main.py)",
    )
    p_run.add_argument(
        "--api-key",
        default=None,
        help="API key (overrides env, credentials file, and .env)",
    )
    p_run.add_argument(
        "--profile",
        default=None,
        help="Credentials profile name (default: 'default' or $FENN_PROFILE)",
    )
    p_run.add_argument(
        "--max-runtime",
        type=int,
        default=10,
        help="Maximum allowed wall-time in minutes (server enforces; default: 10)",
    )
    p_run.add_argument(
        "--detach",
        action="store_true",
        help="Submit the job and exit without streaming logs",
    )
    p_run.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download artifacts on completion",
    )
    p_run.add_argument(
        "--include",
        action="append",
        metavar="PATH",
        help="Extra path (relative to CWD) to include in the upload tarball",
    )
    p_run.add_argument(
        "--exclude",
        action="append",
        metavar="PATTERN",
        help="Extra shell-glob pattern to exclude from the upload tarball",
    )
    p_run.set_defaults(func=run.execute)

    # ========= AUTH =========
    p_auth = subparsers.add_parser(
        "auth", help="Manage credentials for the Fenn remote service"
    )
    auth_subparsers = p_auth.add_subparsers(dest="auth_command", required=True)

    p_login = auth_subparsers.add_parser("login", help="Save an API key for a profile")
    p_login.add_argument(
        "--profile", default=None, help="Profile name (default: 'default')"
    )
    p_login.add_argument(
        "--api-key",
        default=None,
        help="API key (if omitted, will prompt or read from stdin)",
    )
    p_login.set_defaults(func=auth.execute)

    p_status = auth_subparsers.add_parser(
        "status", help="Show the currently configured profile and credit balance"
    )
    p_status.add_argument(
        "--profile", default=None, help="Profile name (default: 'default')"
    )
    p_status.set_defaults(func=auth.execute)

    p_logout = auth_subparsers.add_parser(
        "logout", help="Remove a profile from the credentials file"
    )
    p_logout.add_argument(
        "--profile", default=None, help="Profile name (default: 'default')"
    )
    p_logout.set_defaults(func=auth.execute)

    # ========= GRID =========

    p_grid = subparsers.add_parser(
        "grid",
        help="Run a Fenn project several times, with all possible grid hyperparams",
    )

    p_grid.add_argument(
        "path",
        nargs="?",
        help="Target directory (default: current directory)",
    )

    p_grid.set_defaults(func=grid.execute)

    return parser


def main(argv=None):
    parser = build_parser()
    # parse_args will exit with error if commands are missing due to required=True
    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
