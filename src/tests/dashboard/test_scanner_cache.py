"""Unit tests for FennScanner caching (Pick 2) and running-session detection."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from fenn.dashboard.scanner import FennScanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMPLETED_FN = """\
<?xml version="1.0" encoding="utf-8"?>
<fenn-log project="proj" session_id="{sid}" started="2026-05-21 08:00:00">
  <config><item key="lr" value="0.01" /></config>
  <entry ts="2026-05-21 08:00:00" kind="print" level="info">hello</entry>
  <entry ts="2026-05-21 08:00:01" kind="print" level="warning">warn</entry>
  <meta ended="2026-05-21 08:00:02" duration_s="2" status="completed" />
</fenn-log>
"""

_RUNNING_FN = """\
<?xml version="1.0" encoding="utf-8"?>
<fenn-log project="proj" session_id="{sid}" started="2026-05-21 09:00:00">
  <config><item key="lr" value="0.001" /></config>
  <entry ts="2026-05-21 09:00:00" kind="print" level="info">started</entry>
"""


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Parse cache
# ---------------------------------------------------------------------------


class TestParseCache:
    """FennScanner._parse_cache — hits, misses, and mtime invalidation."""

    def test_first_parse_populates_cache(self, tmp_path):
        """Parsing a file should store the result in _parse_cache."""
        fn = _write(tmp_path / "s1.fn", _COMPLETED_FN.format(sid="s1"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        scanner.parse_fn_file(fn)

        assert str(fn) in scanner._parse_cache

    def test_cache_hit_reuses_result(self, tmp_path):
        """Second call with same mtime must not re-read the file."""
        fn = _write(tmp_path / "s1.fn", _COMPLETED_FN.format(sid="s1"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        first = scanner.parse_fn_file(fn)
        # Overwrite with different content but keep mtime by faking it
        orig_mtime = fn.stat().st_mtime

        with patch.object(Path, "read_text", side_effect=AssertionError("should not read")):
            # Inject a matching mtime so the cache is considered valid
            scanner._parse_cache[str(fn)] = (orig_mtime, first)
            second = scanner.parse_fn_file(fn)

        assert second is not None
        assert second["session_id"] == first["session_id"]

    def test_mtime_change_invalidates_cache(self, tmp_path):
        """Changing mtime should trigger a fresh parse."""
        fn = _write(tmp_path / "s1.fn", _COMPLETED_FN.format(sid="s1"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])
        scanner.parse_fn_file(fn)

        # Rewrite with different content (changes mtime on real FS)
        updated = _COMPLETED_FN.format(sid="s1").replace("hello", "world")
        fn.write_text(updated, encoding="utf-8")

        result = scanner.parse_fn_file(fn)

        assert result is not None
        entry_texts = [e["message"] for e in result["entries"]]
        assert "world" in entry_texts

    def test_missing_file_evicts_cache(self, tmp_path):
        """Deleting a file should return None and evict the cache entry."""
        fn = _write(tmp_path / "s1.fn", _COMPLETED_FN.format(sid="s1"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])
        scanner.parse_fn_file(fn)
        assert str(fn) in scanner._parse_cache

        fn.unlink()
        result = scanner.parse_fn_file(fn)

        assert result is None
        assert str(fn) not in scanner._parse_cache

    def test_parse_extracts_counts(self, tmp_path):
        """Parsed result should report correct warning and exception counts."""
        content = """\
<?xml version="1.0" encoding="utf-8"?>
<fenn-log project="proj" session_id="cx" started="2026-05-21 08:00:00">
  <entry ts="2026-05-21 08:00:00" kind="print" level="warning">w1</entry>
  <entry ts="2026-05-21 08:00:01" kind="print" level="warning">w2</entry>
  <entry ts="2026-05-21 08:00:02" kind="print" level="exception">ex1</entry>
  <meta ended="2026-05-21 08:00:03" duration_s="3" status="completed" />
</fenn-log>
"""
        fn = _write(tmp_path / "cx.fn", content)
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        result = scanner.parse_fn_file(fn)

        assert result["warning_count"] == 2
        assert result["exception_count"] == 1


# ---------------------------------------------------------------------------
# Running → crashed re-evaluation
# ---------------------------------------------------------------------------


class TestRunningStatus:
    """Running sessions flip to 'crashed' when mtime is stale."""

    def test_fresh_running_stays_running(self, tmp_path):
        """A recently-modified open file should stay 'running'."""
        fn = _write(tmp_path / "r1.fn", _RUNNING_FN.format(sid="r1"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        with patch("fenn.dashboard.scanner._running_timeout_s", return_value=300.0):
            with patch("fenn.dashboard.scanner.time") as mock_time:
                mock_time.time.return_value = fn.stat().st_mtime + 10
                result = scanner.parse_fn_file(fn)

        assert result is not None
        assert result["status"] == "running"

    def test_stale_running_becomes_crashed(self, tmp_path):
        """An open file untouched beyond the timeout should become 'crashed'."""
        fn = _write(tmp_path / "r2.fn", _RUNNING_FN.format(sid="r2"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        with patch("fenn.dashboard.scanner._running_timeout_s", return_value=300.0):
            with patch("fenn.dashboard.scanner.time") as mock_time:
                mock_time.time.return_value = fn.stat().st_mtime + 400
                result = scanner.parse_fn_file(fn)

        assert result is not None
        assert result["status"] == "crashed"

    def test_cache_hit_still_checks_timeout(self, tmp_path):
        """Even when the cache is valid, running status must be re-evaluated."""
        fn = _write(tmp_path / "r3.fn", _RUNNING_FN.format(sid="r3"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])
        mtime = fn.stat().st_mtime

        # Prime cache with a "running" entry
        with patch("fenn.dashboard.scanner._running_timeout_s", return_value=300.0):
            with patch("fenn.dashboard.scanner.time") as mock_time:
                mock_time.time.return_value = mtime + 10
                scanner.parse_fn_file(fn)

        assert scanner._parse_cache[str(fn)][1]["status"] == "running"

        # Now time has advanced past the threshold — cache mtime unchanged
        with patch("fenn.dashboard.scanner._running_timeout_s", return_value=300.0):
            with patch("fenn.dashboard.scanner.time") as mock_time:
                mock_time.time.return_value = mtime + 400
                result = scanner.parse_fn_file(fn)

        assert result["status"] == "crashed"
        # The cached record itself must still say "running" (only the returned copy flips)
        assert scanner._parse_cache[str(fn)][1]["status"] == "running"

    def test_env_timeout_override(self, tmp_path, monkeypatch):
        """FENN_RUNNING_TIMEOUT_S env var must change the crash threshold."""
        monkeypatch.setenv("FENN_RUNNING_TIMEOUT_S", "60")
        fn = _write(tmp_path / "r4.fn", _RUNNING_FN.format(sid="r4"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        with patch("fenn.dashboard.scanner.time") as mock_time:
            mock_time.time.return_value = fn.stat().st_mtime + 90
            result = scanner.parse_fn_file(fn)

        assert result["status"] == "crashed"


# ---------------------------------------------------------------------------
# Directory listing cache (TTL)
# ---------------------------------------------------------------------------


class TestFileListingCache:
    """find_fn_files() should cache directory scans for _FILES_CACHE_TTL_S."""

    def test_listing_is_cached_within_ttl(self, tmp_path):
        """Two rapid calls should return the same list object (cache hit)."""
        _write(tmp_path / "s1.fn", _COMPLETED_FN.format(sid="s1"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        first = scanner.find_fn_files()
        second = scanner.find_fn_files()

        assert first is second  # same object == cache was reused

    def test_listing_refreshes_after_ttl(self, tmp_path):
        """After TTL expires a new scan should pick up added files."""
        _write(tmp_path / "s1.fn", _COMPLETED_FN.format(sid="s1"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])
        scanner.find_fn_files()

        # Expire the cache manually
        ts, files = scanner._files_cache
        scanner._files_cache = (ts - 10.0, files)

        _write(tmp_path / "s2.fn", _COMPLETED_FN.format(sid="s2"))
        refreshed = scanner.find_fn_files()

        assert any(p.stem == "s2" for p in refreshed)

    def test_add_dir_invalidates_listing_cache(self, tmp_path):
        """_add_dir() must invalidate _files_cache so the new dir is scanned."""
        _write(tmp_path / "s1.fn", _COMPLETED_FN.format(sid="s1"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])
        scanner.find_fn_files()
        assert scanner._files_cache is not None

        extra = tmp_path / "extra"
        extra.mkdir()
        scanner._add_dir(str(extra))

        assert scanner._files_cache is None


# ---------------------------------------------------------------------------
# get_session short-circuit by filename stem
# ---------------------------------------------------------------------------


class TestGetSessionShortCircuit:
    """get_session() should prefer the file whose stem matches session_id."""

    def test_finds_session_by_stem(self, tmp_path):
        """Session is resolved via stem match, not a full scan."""
        _write(tmp_path / "sess_abc.fn", _COMPLETED_FN.format(sid="sess_abc"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        result = scanner.get_session("proj", "sess_abc")

        assert result is not None
        assert result["session_id"] == "sess_abc"

    def test_returns_none_for_unknown_session(self, tmp_path):
        """Non-existent session should return None."""
        _write(tmp_path / "sess_abc.fn", _COMPLETED_FN.format(sid="sess_abc"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        result = scanner.get_session("proj", "no_such_session")

        assert result is None

    def test_session_includes_projects_list(self, tmp_path):
        """Returned session dict must include the top-level 'projects' key."""
        _write(tmp_path / "sess_abc.fn", _COMPLETED_FN.format(sid="sess_abc"))
        scanner = FennScanner(extra_dirs=[str(tmp_path)])

        result = scanner.get_session("proj", "sess_abc")

        assert "projects" in result
