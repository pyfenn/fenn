import argparse

import fenn.cli.dashboard as dashboard
import fenn.cli.grid as grid
import fenn.cli.list as list
import fenn.cli.pull as pull


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
