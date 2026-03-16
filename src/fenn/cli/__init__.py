import argparse

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
