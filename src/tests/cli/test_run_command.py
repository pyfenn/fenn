"""Tests for the `fenn run` CLI wiring."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fenn.cli import build_parser


def test_run_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["run"])
    assert args.command == "run"
    assert args.script is None
    assert args.host is None
    assert args.api_key is None
    assert args.profile is None
    assert args.detach is False
    assert args.no_download is False
    assert args.max_runtime == 3600


def test_run_parser_collects_includes_and_excludes():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "main.py",
            "--host",
            "http://localhost:8000",
            "--include",
            "data",
            "--include",
            "models",
            "--exclude",
            "*.tmp",
        ]
    )
    assert args.script == "main.py"
    assert args.host == "http://localhost:8000"
    assert args.include == ["data", "models"]
    assert args.exclude == ["*.tmp"]


def test_run_local_invokes_runpy(tmp_path):
    script = tmp_path / "main.py"
    script.write_text("RAN = True\n")

    from fenn.cli import run as run_module

    parser = build_parser()
    args = parser.parse_args(["run", str(script)])

    with patch("fenn.remote.local_runner.runpy.run_path") as run_path:
        run_module.execute(args)
        run_path.assert_called_once()
        (called_path,) = run_path.call_args.args
        assert called_path == str(script.resolve())
        assert run_path.call_args.kwargs.get("run_name") == "__main__"


def test_run_local_supports_relative_imports(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    script = project / "main.py"
    script.write_text(
        "from pathlib import Path\n"
        "from .modules import VALUE\n"
        "Path('result.txt').write_text(VALUE, encoding='utf-8')\n",
        encoding="utf-8",
    )
    (project / "modules.py").write_text("VALUE = 'ok'\n", encoding="utf-8")

    from fenn.remote.local_runner import run_local

    run_local(script)

    assert (project / "result.txt").read_text(encoding="utf-8") == "ok"


def test_run_missing_script_exits(tmp_path, capsys):
    from fenn.cli import run as run_module

    parser = build_parser()
    args = parser.parse_args(["run", str(tmp_path / "nope.py")])

    with pytest.raises(SystemExit) as exc:
        run_module.execute(args)
    assert exc.value.code == 1


def test_auth_parser_subcommands():
    parser = build_parser()
    args = parser.parse_args(["auth", "login", "--profile", "work"])
    assert args.command == "auth"
    assert args.auth_command == "login"
    assert args.profile == "work"
