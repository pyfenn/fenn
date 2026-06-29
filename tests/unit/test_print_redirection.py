import builtins

import pytest

from fenn import Fenn
from fenn.utils.logging import original_print


def _write_config(path):
    path.write_text(
        "session_id: test-session\nlogger:\n  dir: logs\nexport:\n  dir: exports\n",
        encoding="utf-8",
    )


def test_print_is_restored_after_entrypoint_exception(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path / "fenn.yaml")

    app = Fenn()
    app.disable_disclaimer()

    @app.entrypoint
    def main(args):
        assert builtins.print is not original_print
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        app.run()

    assert builtins.print is original_print


def test_print_is_restored_after_successful_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path / "fenn.yaml")

    app = Fenn()
    app.disable_disclaimer()

    @app.entrypoint
    def main(args):
        assert builtins.print is not original_print
        print("captured by fenn logger")
        return "ok"

    assert app.run() == "ok"

    assert builtins.print is original_print
