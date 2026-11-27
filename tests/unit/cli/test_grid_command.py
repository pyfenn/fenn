"Test for `fenn grid` command."

from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytest

from fenn.cli import build_parser
from fenn.cli.grid import TemplateError, _build_variants, _parse_grid
from fenn.parser import Parser


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
    yaml: Path = Path("tests/unit/mock/templates/default/fenn.yaml")
    with pytest.raises(TemplateError):
        _parse_grid(yaml)


def test_execute(monkeypatch) -> None:
    monkeypatch.setattr(Parser, "_instance", None)
    from fenn.cli.grid import execute

    parser: ArgumentParser = build_parser()
    args: Namespace = parser.parse_args(
        ["grid", "tests/unit/mock/templates/with_grid/main.py"]
    )
    execute(args=args)
    assert not Path("tests/unit/mock/templates/with_grid/fenn_copy.yaml").exists()


def test_execute_fenn_uses_sys_executable(monkeypatch, tmp_path: Path) -> None:
    import sys
    from unittest.mock import MagicMock

    from fenn.cli import grid

    yaml_path: Path = tmp_path / "fenn.yaml"
    yaml_path.write_text("train:\n  lr: 0.1\n", encoding="utf-8")
    main_path: Path = tmp_path / "main.py"
    run_mock = MagicMock()
    monkeypatch.setattr(grid.subprocess, "run", run_mock)

    grid._execute_fenn({"lr": 0.2}, main_path, yaml_path)

    run_mock.assert_called_once()
    assert run_mock.call_args.args[0][0] == sys.executable
    assert run_mock.call_args.args[0][1] == main_path
