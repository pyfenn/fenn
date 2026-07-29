"""Unit tests for TemplateRunner."""

import signal
import sys
import textwrap
import time

import pytest

from fenn.dashboard.runner import (
    PauseUnsupportedError,
    TemplateLaunchError,
    TemplateRunner,
)
from fenn.dashboard.scanner import FennScanner


def _write_template(tmp_path, entrypoint_body, fenn_yaml=None, entrypoint="main.py"):
    (tmp_path / entrypoint).write_text(
        textwrap.dedent(entrypoint_body), encoding="utf-8"
    )
    if fenn_yaml is not None:
        (tmp_path / "fenn.yaml").write_text(
            textwrap.dedent(fenn_yaml), encoding="utf-8"
        )
    return tmp_path


@pytest.fixture()
def scanner(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "FENN_DASHBOARD_OVERRIDES_PATH", str(tmp_path / "overrides.json")
    )
    return FennScanner(extra_dirs=[])


@pytest.fixture()
def runner():
    # Windows process spawn is slower than fork()-based platforms; keep
    # enough headroom to reliably catch immediate crashes there.
    return TemplateRunner(startup_grace_s=0.4)


class TestReadLoggerDir:
    def test_reads_configured_dir(self, tmp_path):
        _write_template(
            tmp_path,
            "import time\ntime.sleep(5)\n",
            fenn_yaml="logger:\n  dir: custom_logs\n",
        )
        runner_ = TemplateRunner()
        log_dir = runner_._read_logger_dir(tmp_path)
        assert log_dir == (tmp_path / "custom_logs").resolve()

    def test_defaults_when_key_missing(self, tmp_path):
        _write_template(tmp_path, "pass\n", fenn_yaml="project: test\n")
        runner_ = TemplateRunner()
        log_dir = runner_._read_logger_dir(tmp_path)
        assert log_dir == (tmp_path / "logger").resolve()

    def test_missing_fenn_yaml_raises(self, tmp_path):
        _write_template(tmp_path, "pass\n", fenn_yaml=None)
        runner_ = TemplateRunner()
        with pytest.raises(TemplateLaunchError, match="fenn.yaml"):
            runner_._read_logger_dir(tmp_path)

    def test_invalid_yaml_raises(self, tmp_path):
        _write_template(tmp_path, "pass\n", fenn_yaml="logger: [unterminated\n")
        runner_ = TemplateRunner()
        with pytest.raises(TemplateLaunchError, match="Invalid fenn.yaml"):
            runner_._read_logger_dir(tmp_path)


class TestLaunch:
    def test_launch_success_registers_log_dir(self, tmp_path, scanner, runner):
        _write_template(
            tmp_path,
            """
            import time
            time.sleep(2)
            """,
            fenn_yaml="logger:\n  dir: logs\n",
        )
        running = runner.launch(tmp_path, scanner=scanner)
        try:
            assert running.log_dir == (tmp_path / "logs").resolve()
            assert running.log_dir in scanner._dirs
            assert running.process.poll() is None
            assert runner.get(running.run_id) is running
        finally:
            running.process.kill()
            running.process.wait()

    def test_missing_directory_raises(self, tmp_path, scanner, runner):
        with pytest.raises(TemplateLaunchError, match="not a directory"):
            runner.launch(tmp_path / "nope", scanner=scanner)

    def test_missing_entrypoint_raises(self, tmp_path, scanner, runner):
        (tmp_path / "fenn.yaml").write_text("logger:\n  dir: logs\n", encoding="utf-8")
        with pytest.raises(TemplateLaunchError, match="Entrypoint"):
            runner.launch(tmp_path, scanner=scanner)

    def test_immediate_crash_raises_with_stderr(self, tmp_path, scanner, runner):
        _write_template(
            tmp_path,
            """
            import sys
            sys.stderr.write("boom: bad config\\n")
            sys.exit(1)
            """,
            fenn_yaml="logger:\n  dir: logs\n",
        )
        with pytest.raises(TemplateLaunchError, match="boom: bad config"):
            runner.launch(tmp_path, scanner=scanner)

    def test_crash_does_not_register_log_dir(self, tmp_path, scanner, runner):
        _write_template(
            tmp_path,
            "import sys\nsys.exit(1)\n",
            fenn_yaml="logger:\n  dir: logs\n",
        )
        with pytest.raises(TemplateLaunchError):
            runner.launch(tmp_path, scanner=scanner)
        assert (tmp_path / "logs").resolve() not in scanner._dirs

    def test_successful_exit_zero_within_grace_still_registers(
        self, tmp_path, scanner, runner
    ):
        _write_template(
            tmp_path,
            "pass\n",
            fenn_yaml="logger:\n  dir: logs\n",
        )
        running = runner.launch(tmp_path, scanner=scanner)
        assert running.log_dir in scanner._dirs


class TestListActive:
    def test_list_active_prunes_finished(self, tmp_path, scanner, runner):
        _write_template(
            tmp_path,
            "import time\ntime.sleep(2)\n",
            fenn_yaml="logger:\n  dir: logs\n",
        )
        running = runner.launch(tmp_path, scanner=scanner)
        assert running in runner.list_active()
        running.process.kill()
        running.process.wait()
        assert running.run_id not in {r.run_id for r in runner.list_active()}


class TestFindBySession:
    def test_finds_by_session_id(self, tmp_path, scanner, runner):
        _write_template(
            tmp_path,
            "import time\ntime.sleep(2)\n",
            fenn_yaml="logger:\n  dir: logs\n",
        )
        running = runner.launch(tmp_path, scanner=scanner)
        try:
            running.session_id = "abc123"
            assert runner.find_by_session("abc123") is running
            assert runner.find_by_session("unknown") is None
        finally:
            running.process.kill()
            running.process.wait()


class TestPauseResumeSignalDelivery:
    """Signal delivery, mocked — portable and doesn't depend on process timing."""

    def test_pause_sends_sigstop_and_sets_flag(
        self, tmp_path, scanner, runner, monkeypatch
    ):
        _write_template(
            tmp_path,
            "import time\ntime.sleep(2)\n",
            fenn_yaml="logger:\n  dir: logs\n",
        )
        running = runner.launch(tmp_path, scanner=scanner)
        try:
            calls = []
            monkeypatch.setattr(
                "fenn.dashboard.runner.os.kill",
                lambda pid, sig: calls.append((pid, sig)),
            )
            runner.pause(running.run_id)
            assert calls == [(running.process.pid, signal.SIGSTOP)]
            assert running.paused is True
        finally:
            running.process.kill()
            running.process.wait()

    def test_resume_sends_sigcont_and_clears_flag(
        self, tmp_path, scanner, runner, monkeypatch
    ):
        _write_template(
            tmp_path,
            "import time\ntime.sleep(2)\n",
            fenn_yaml="logger:\n  dir: logs\n",
        )
        running = runner.launch(tmp_path, scanner=scanner)
        try:
            calls = []
            monkeypatch.setattr(
                "fenn.dashboard.runner.os.kill",
                lambda pid, sig: calls.append((pid, sig)),
            )
            running.paused = True
            runner.resume(running.run_id)
            assert calls == [(running.process.pid, signal.SIGCONT)]
            assert running.paused is False
        finally:
            running.process.kill()
            running.process.wait()

    def test_pause_unknown_run_id_is_noop(self, runner, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "fenn.dashboard.runner.os.kill", lambda pid, sig: calls.append((pid, sig))
        )
        runner.pause("unknown-run-id")  # should not raise
        assert calls == []

    def test_signal_none_raises_unsupported(self, runner):
        with pytest.raises(PauseUnsupportedError):
            runner._signal("whatever", None)


@pytest.mark.skipif(
    sys.platform != "linux", reason="Verifies real stopped-process state via /proc"
)
class TestPauseResumeRealProcess:
    def test_pause_actually_stops_the_process(self, tmp_path, scanner, runner):
        _write_template(
            tmp_path,
            "import time\ntime.sleep(5)\n",
            fenn_yaml="logger:\n  dir: logs\n",
        )
        running = runner.launch(tmp_path, scanner=scanner)
        try:
            runner.pause(running.run_id)
            assert _wait_for_state(running.process.pid, "T") == "T"

            runner.resume(running.run_id)
            assert _wait_for_state(running.process.pid, "T", negate=True) != "T"
        finally:
            running.process.kill()
            running.process.wait()


def _proc_state(pid: int) -> str:
    with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return fields[0]


def _wait_for_state(pid: int, target: str, *, negate: bool = False) -> str:
    """Poll /proc/<pid>/stat until it reaches (or leaves) ``target`` state."""
    state = _proc_state(pid)
    for _ in range(20):
        state = _proc_state(pid)
        if (state != target) if negate else (state == target):
            return state
        time.sleep(0.05)
    return state
