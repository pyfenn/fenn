import argparse
import pstats
import subprocess
import sys
from pathlib import Path

from fenn.utils.logging import logger

ROOT = Path(__file__).resolve().parents[1]


def execute(args: argparse.Namespace) -> None:
    template = ROOT / args.template
    entrypoint = template / "main.py"

    if not entrypoint.is_file() or template.parent != ROOT:
        logger.info(f"Unknown template: {args.template}")
        sys.exit(1)

    output_dir = ROOT / "profiling" / "results" / args.template
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

    with report_path.open("w", encoding="utf-8") as report:
        (
            pstats.Stats(str(profile_path), stream=report)
            .strip_dirs()
            .sort_stats("cumulative")
            .print_stats(args.limit)
        )

    logger.info(f"Profile: {profile_path}")
    logger.info(f"Report:  {report_path}")