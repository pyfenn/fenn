"""Unit tests for fenn/dashboard/log_stream.py"""

from fenn.dashboard.log_stream import iter_new_lines


def _noop_sleep(_seconds: float) -> None:
    pass


class TestBacklog:
    def test_yields_existing_lines_when_from_start(self, tmp_path):
        path = tmp_path / "run.log"
        path.write_text("line1\nline2\nline3\n", encoding="utf-8")

        lines = list(
            iter_new_lines(
                path,
                from_start=True,
                sleep_fn=_noop_sleep,
                should_continue=lambda: False,
            )
        )
        assert lines == ["line1", "line2", "line3"]

    def test_backlog_is_truncated(self, tmp_path):
        path = tmp_path / "run.log"
        path.write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")

        lines = list(
            iter_new_lines(
                path,
                from_start=True,
                backlog_lines=2,
                sleep_fn=_noop_sleep,
                should_continue=lambda: False,
            )
        )
        assert lines == ["l4", "l5"]

    def test_no_backlog_when_from_start_false(self, tmp_path):
        path = tmp_path / "run.log"
        path.write_text("line1\nline2\n", encoding="utf-8")

        lines = list(
            iter_new_lines(
                path,
                from_start=False,
                sleep_fn=_noop_sleep,
                should_continue=lambda: False,
            )
        )
        assert lines == []

    def test_missing_file_yields_nothing(self, tmp_path):
        lines = list(
            iter_new_lines(
                tmp_path / "does-not-exist.log",
                from_start=True,
                sleep_fn=_noop_sleep,
                should_continue=lambda: False,
            )
        )
        assert lines == []


class TestPolling:
    def test_yields_lines_appended_between_polls(self, tmp_path):
        path = tmp_path / "run.log"
        path.write_text("line1\n", encoding="utf-8")

        state = {"step": 0}

        def should_continue() -> bool:
            state["step"] += 1
            if state["step"] == 1:
                with open(path, "a", encoding="utf-8") as f:
                    f.write("line2\n")
                return True
            return False

        lines = list(
            iter_new_lines(
                path,
                from_start=True,
                sleep_fn=_noop_sleep,
                should_continue=should_continue,
            )
        )
        assert lines == ["line1", "line2"]

    def test_yields_trailing_unterminated_line_on_stop(self, tmp_path):
        path = tmp_path / "run.log"
        path.write_text("line1\n", encoding="utf-8")

        state = {"step": 0}

        def should_continue() -> bool:
            state["step"] += 1
            if state["step"] == 1:
                with open(path, "a", encoding="utf-8") as f:
                    f.write("partial")  # no trailing newline
                return True
            return False

        lines = list(
            iter_new_lines(
                path,
                from_start=True,
                sleep_fn=_noop_sleep,
                should_continue=should_continue,
            )
        )
        assert lines == ["line1", "partial"]

    def test_sleep_fn_invoked_between_polls(self, tmp_path):
        path = tmp_path / "run.log"
        path.write_text("line1\n", encoding="utf-8")

        sleeps = []
        state = {"step": 0}

        def should_continue() -> bool:
            state["step"] += 1
            return state["step"] <= 2

        list(
            iter_new_lines(
                path,
                from_start=True,
                poll_interval=0.25,
                sleep_fn=sleeps.append,
                should_continue=should_continue,
            )
        )
        assert sleeps == [0.25, 0.25]
