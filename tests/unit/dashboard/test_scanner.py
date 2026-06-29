"""Unit tests for FennScanner.get_all_sessions() and get_session()."""

import time
from pathlib import Path

from fenn.dashboard.scanner import FennScanner

_SAMPLE_FN = """\
<?xml version="1.0" encoding="utf-8"?>
<fenn-log project="{project}" session_id="{sid}" started="2026-05-21 08:00:00">
  <meta ended="2026-05-21 08:00:02" duration_s="2" status="completed" />
</fenn-log>
"""


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# get_all_sessions
# ---------------------------------------------------------------------------


class TestGetAllSessions:
    def test_empty_directory_returns_empty_list(self, tmp_path):
        """No .fn files should result in an empty list."""
        scanner = FennScanner(extra_dirs=[str(tmp_path)])
        assert scanner.get_all_sessions() == []

    def test_returns_all_valid_sessions(self, tmp_path):
        """Multiple valid .fn files should all appear in the result."""
        _write(tmp_path / "s1.fn", _SAMPLE_FN.format(project="proj", sid="s1"))
        _write(tmp_path / "s2.fn", _SAMPLE_FN.format(project="proj", sid="s2"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        result = scanner.get_all_sessions()
        session_ids = {s["session_id"] for s in result}

        assert session_ids == {"s1", "s2"}
        assert len(result) == 2

    def test_newest_file_appears_first(self, tmp_path):
        """Sessions should be ordered by file mtime, newest first."""
        old_fn = _write(tmp_path / "old.fn", _SAMPLE_FN.format(project="proj", sid="old"))
        time.sleep(0.05)
        new_fn = _write(tmp_path / "new.fn", _SAMPLE_FN.format(project="proj", sid="new"))

        scanner = FennScanner(extra_dirs=[str(tmp_path)])
        result = scanner.get_all_sessions()

        assert result[0]["session_id"] == "new"
        assert result[1]["session_id"] == "old"

    def test_unparsable_file_is_skipped(self, tmp_path):
        """A file that fails to parse should not appear in the results."""
        _write(tmp_path / "broken.fn", "not even close to xml <<<")
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        result = scanner.get_all_sessions()

        assert result == []


# ---------------------------------------------------------------------------
# get_session — additional edge cases beyond test_scanner_cache.py
# ---------------------------------------------------------------------------


class TestGetSessionEdgeCases:
    def test_wrong_project_name_returns_none(self, tmp_path):
        """A session_id match with a different project must return None."""
        _write(tmp_path / "sess_x.fn", _SAMPLE_FN.format(project="proj_a", sid="sess_x"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        result = scanner.get_session("proj_b", "sess_x")

        assert result is None

    def test_falls_back_to_full_scan_when_stem_mismatches(self, tmp_path):
        """If no file stem matches session_id, all files are scanned as fallback."""
        # filename stem ("legacy") does not match the session_id inside it
        _write(tmp_path / "legacy.fn", _SAMPLE_FN.format(project="proj", sid="sess_y"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        result = scanner.get_session("proj", "sess_y")

        assert result is not None
        assert result["session_id"] == "sess_y"