"""Profile CLI support for fenn templates.

This module provides the implementation for the `fenn profile` subcommand.
It runs the selected template's `main.py` under `cProfile` and writes a
summary report to `profiling_results/<template>/cprofile.txt`.
"""

import argparse
import builtins
import pstats
import subprocess
import sys
from pathlib import Path

from fenn.logging import logger, original_print


def execute(args: argparse.Namespace) -> None:
    """Run the fenn template under cProfile and write profiling output.

    Args:
        args: Parsed CLI arguments with attributes:
            - template: name of the template folder to profile
            - limit: number of lines to include in the report
    """
    template = (Path.cwd() / args.template).resolve()
    entrypoint = template / "main.py"

    if not entrypoint.is_file():
        logger.error(f"Unknown template: {args.template}")
        sys.exit(1)

    output_dir = (Path.cwd() / "profiling_results" / args.template).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_path = output_dir / "cprofile.prof"
    report_path = output_dir / "cprofile.txt"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "cProfile",
            "-o",
            str(profile_path),
            entrypoint.name,
        ],
        cwd=template,
        check=True,
    )

    saved_print = builtins.print
    builtins.print = original_print
    try:
        with report_path.open("w", encoding="utf-8") as report:
            stats = pstats.Stats(str(profile_path), stream=report)
            stats.strip_dirs().sort_stats("cumulative")
            stats.print_stats(args.limit)
    finally:
        builtins.print = saved_print

    logger.info(f"Profile: {profile_path}")
    logger.info(f"Report:  {report_path}")
