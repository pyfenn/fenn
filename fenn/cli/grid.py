import argparse
import os
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path

import yaml
from colorama import Fore, Style

from fenn.exceptions import TemplateError
from fenn.logging import logger
from fenn.parser import Parser


def execute(args: argparse.Namespace) -> None:
    """
    Execute the fenn grid command to train a model with different seeds, epoch counts, or learning rates.

    Args:
        args: Parsed command-line arguments containing:
            - path: Target directory (default: current directory)
    """

    main_path: Path = Path(args.path).resolve() if args.path else Path.cwd() / "main.py"
    yaml_path: Path = main_path.parent / "fenn.yaml"
    yaml_copy: Path = main_path.parent / "fenn_copy.yaml"
    try:
        parsed_grid: list[dict] = _parse_grid(yaml_path=yaml_path)
    except TemplateError as e:
        logger.error(
            f"{Fore.RED}Template error: missing grid section{e}{Style.RESET_ALL}"
        )
        sys.exit(1)
    shutil.copy(yaml_path, yaml_copy)
    try:
        for hyperparameter in parsed_grid:
            _execute_fenn(
                hyperparameter=hyperparameter, main_path=main_path, yaml_path=yaml_path
            )
    finally:
        shutil.copy(yaml_copy, yaml_path)
        os.remove(yaml_copy)


def _build_variants(raw_grid: dict[str, list | int]) -> list[dict[str, int]]:
    keys = raw_grid.keys()
    values: list[list | list[int]] = [
        v if isinstance(v, list) else [v] for v in raw_grid.values()
    ]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _parse_grid(yaml_path: Path) -> list[dict[str, int]]:
    parsed_yaml = Parser(config_file=yaml_path).load_configuration()
    if parsed_yaml.get("grid") is None:
        raise TemplateError
    return _build_variants(raw_grid=parsed_yaml.get("grid").get("train"))


def _execute_fenn(
    hyperparameter: dict[str, int], main_path: Path, yaml_path: Path
) -> None:
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train_data = config["train"]
    for key, value in hyperparameter.items():
        train_data.update({key: value})
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    subprocess.run(["python3", main_path], cwd=main_path.parent)
