"Test for `fenn grid` command."

from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytest

from fenn.args.parser import Parser
from fenn.cli import build_parser
from fenn.cli.grid import TemplateError, _build_variants, _execute_fenn, _parse_grid


def test_grid_parser_defaults() -> None:
    parser: ArgumentParser = build_parser()
    args: Namespace = parser.parse_args(["grid"])
    assert args.command == "grid"
    assert args.path is None


def test_run_parser_collects_includes_and_excludes() -> None:
    parser: ArgumentParser = build_parser()
    args: Namespace = parser.parse_args(
        [
            "grid",
            "main.py",
        ]
    )
    assert args.path == "main.py"


def test_cartesian_product() -> None:
    expected_result: list[dict[str, int]] = [
        {"seed": 42, "batch": 16, "epochs": 30, "lr": 1e-3},
        {"seed": 41, "batch": 16, "epochs": 30, "lr": 1e-3},
        {"seed": 42, "batch": 15, "epochs": 30, "lr": 1e-3},
        {"seed": 41, "batch": 15, "epochs": 30, "lr": 1e-3},
    ]
    test_dict: dict[str : list[int] | int] = {
        "seed": [42, 41],
        "batch": [16, 15],
        "epochs": 30,
        "lr": 1e-3,
    }
    test_result: list[dict[str, int]] = _build_variants(test_dict)
    assert len(test_result) == 4
    for variant in expected_result:
        assert variant in test_result


def test_template_error(monkeypatch) -> None:
    monkeypatch.setattr(Parser, "_instance", None)
    yaml: Path = Path("src/tests/templates/default/fenn.yaml")
    with pytest.raises(TemplateError):
        _parse_grid(yaml)


def test_execution(monkeypatch) -> None:
    monkeypatch.setattr(Parser, "_instance", None)
    yaml: Path = Path("src/tests/templates/with_grid/fenn.yaml")
    main: Path = Path("src/tests/templates/with_grid/main.py")
    hyperparams: list[dict[str, int]] = _parse_grid(yaml)
    for hyperparameter in hyperparams:
        _execute_fenn(hyperparameter=hyperparameter, main_path=main, yaml_path=yaml)
