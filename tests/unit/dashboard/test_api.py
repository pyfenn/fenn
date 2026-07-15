"""Unit tests for the /api/sessions endpoint (Pick 3)."""

import pytest

from fenn.dashboard.app import app
from fenn.dashboard.scanner import FennScanner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FN_TMPL = """\
<?xml version="1.0" encoding="utf-8"?>
<fenn-log project="{project}" session_id="{sid}" started="{started}">
  <meta ended="{ended}" duration_s="{dur}" status="{status}" />
</fenn-log>
"""

_RUNNING_TMPL = """\
<?xml version="1.0" encoding="utf-8"?>
<fenn-log project="{project}" session_id="{sid}" started="{started}">
  <entry ts="{started}" kind="print" level="info">running</entry>
"""


def _write(path, content):
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def scanner_with_sessions(tmp_path):
    """Return a FennScanner loaded with a small set of known sessions."""
    sessions = [
        dict(
            project="alpha",
            sid="a1",
            started="2026-05-21 07:00:00",
            ended="2026-05-21 07:00:10",
            dur=10,
            status="completed",
        ),
        dict(
            project="alpha",
            sid="a2",
            started="2026-05-21 08:00:00",
            ended="2026-05-21 08:00:05",
            dur=5,
            status="failed",
        ),
        dict(
            project="beta",
            sid="b1",
            started="2026-05-21 06:00:00",
            ended="2026-05-21 06:00:20",
            dur=20,
            status="completed",
        ),
    ]
    for s in sessions:
        _write(tmp_path / f"{s['sid']}.fn", _FN_TMPL.format(**s))

    scanner = FennScanner(extra_dirs=[str(tmp_path)])
    return scanner


@pytest.fixture()
def client(scanner_with_sessions):
    """Flask test client wired to a fresh scanner with known data."""
    import fenn.dashboard.app as app_module

    original = app_module.scanner
    app_module.scanner = scanner_with_sessions
    app.config["TESTING"] = True
    with app.test_client() as c:
        with c.session_transaction() as sess:
            sess["user"] = {"email": "test@example.com"}  # любой dict
        yield c
    app_module.scanner = original


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


class TestApiSessionsShape:
    """The /api/sessions response must always include the standard envelope."""

    def test_default_response_shape(self, client):
        """Default call must return items, total, limit, and offset keys."""
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    def test_items_omit_entries(self, client):
        """Session items in listing must not include the 'entries' key."""
        resp = client.get("/api/sessions")
        data = resp.get_json()
        for item in data["items"]:
            assert "entries" not in item

    def test_default_limit_is_20(self, client):
        """Default limit must equal 20."""
        resp = client.get("/api/sessions")
        data = resp.get_json()
        assert data["limit"] == 20

    def test_default_offset_is_0(self, client):
        """Default offset must equal 0."""
        resp = client.get("/api/sessions")
        data = resp.get_json()
        assert data["offset"] == 0


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestApiSessionsFilter:
    """project= and status= query parameters must filter results."""

    def test_filter_by_project(self, client):
        """?project=alpha should return only alpha sessions."""
        resp = client.get("/api/sessions?project=alpha")
        data = resp.get_json()
        assert all(s["project"] == "alpha" for s in data["items"])
        assert data["total"] == 2

    def test_filter_by_status_completed(self, client):
        """?status=completed should return only completed sessions."""
        resp = client.get("/api/sessions?status=completed")
        data = resp.get_json()
        assert all(s["status"] == "completed" for s in data["items"])

    def test_filter_by_project_and_status(self, client):
        """Combined project + status filter must AND the two predicates."""
        resp = client.get("/api/sessions?project=alpha&status=failed")
        data = resp.get_json()
        assert data["total"] == 1
        assert data["items"][0]["session_id"] == "a2"

    def test_unknown_project_returns_empty(self, client):
        """Filter for a non-existent project must return empty items list."""
        resp = client.get("/api/sessions?project=nope")
        data = resp.get_json()
        assert data["total"] == 0
        assert data["items"] == []


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestApiSessionsPagination:
    """limit= and offset= must slice results correctly."""

    def test_limit_respected(self, client):
        """?limit=1 should return exactly one item."""
        resp = client.get("/api/sessions?limit=1")
        data = resp.get_json()
        assert len(data["items"]) == 1
        assert data["limit"] == 1

    def test_offset_skips_items(self, client):
        """?offset=2 should skip the first two results."""
        all_resp = client.get("/api/sessions").get_json()
        paged_resp = client.get("/api/sessions?offset=2").get_json()
        if all_resp["total"] >= 3:
            assert (
                paged_resp["items"][0]["session_id"]
                == all_resp["items"][2]["session_id"]
            )

    def test_offset_beyond_total_returns_empty(self, client):
        """offset larger than total must return zero items."""
        resp = client.get("/api/sessions?offset=1000")
        data = resp.get_json()
        assert data["items"] == []
        assert data["total"] == 3  # total is unaffected by offset


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class TestApiSessionsSort:
    """sort= parameter must reorder results."""

    def test_sort_by_started_descending(self, client):
        """Default sort (-started) should put the newest session first."""
        resp = client.get("/api/sessions?sort=-started")
        data = resp.get_json()
        starts = [s["started"] for s in data["items"]]
        assert starts == sorted(starts, reverse=True)

    def test_sort_by_duration_ascending(self, client):
        """?sort=duration_s should order shortest first."""
        resp = client.get("/api/sessions?sort=duration_s")
        data = resp.get_json()
        durations = [
            s["duration_s"] for s in data["items"] if s["duration_s"] is not None
        ]
        assert durations == sorted(durations)

    def test_sort_by_started_ascending(self, client):
        """?sort=started should order oldest first."""
        resp = client.get("/api/sessions?sort=started")
        data = resp.get_json()
        starts = [s["started"] for s in data["items"]]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestApiSessionsErrors:
    """Invalid inputs must return 400 with a structured error envelope."""

    def _assert_error(self, resp, expected_code, expected_param=None):
        assert resp.status_code == 400
        body = resp.get_json()
        assert "error" in body
        assert body["error"]["code"] == expected_code
        if expected_param is not None:
            assert body["error"]["param"] == expected_param

    def test_invalid_status_value(self, client):
        """Unknown status string must return 400 with param='status'."""
        resp = client.get("/api/sessions?status=unknown")
        self._assert_error(resp, "invalid_param", "status")

    def test_invalid_sort_field(self, client):
        """Unknown sort field must return 400 with param='sort'."""
        resp = client.get("/api/sessions?sort=nosuchfield")
        self._assert_error(resp, "invalid_param", "sort")

    def test_non_integer_limit(self, client):
        """Non-numeric limit must return 400 with param='limit'."""
        resp = client.get("/api/sessions?limit=abc")
        self._assert_error(resp, "invalid_param", "limit")

    def test_limit_out_of_range(self, client):
        """limit=0 is below the minimum (1) and must be rejected."""
        resp = client.get("/api/sessions?limit=0")
        self._assert_error(resp, "invalid_param", "limit")

    def test_negative_offset(self, client):
        """Negative offset must return 400."""
        resp = client.get("/api/sessions?offset=-1")
        self._assert_error(resp, "invalid_param", "offset")


# ---------------------------------------------------------------------------
# Backward-compat: existing endpoints still work
# ---------------------------------------------------------------------------


class TestExistingEndpoints:
    """Existing API routes must not be broken by Pick 3 additions."""

    def test_api_overview_still_returns_200(self, client):
        """/api/overview should remain functional."""
        resp = client.get("/api/overview")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "projects" in data
        assert "total_sessions" in data


# ---------------------------------------------------------------------------
# Date range filtering
# ---------------------------------------------------------------------------


class TestApiSessionsDateRangeFilter:
    """started_after= and started_before= query parameters must filter by
    the session's started timestamp."""

    def test_started_after_filters_out_earlier_sessions(self, client):
        """?started_after should exclude sessions that started before it."""
        resp = client.get("/api/sessions?started_after=2026-05-21 07:00:00")
        data = resp.get_json()
        sids = {s["session_id"] for s in data["items"]}
        assert sids == {"a1", "a2"}
        assert data["total"] == 2

    def test_started_before_filters_out_later_sessions(self, client):
        """?started_before should exclude sessions that started after it."""
        resp = client.get("/api/sessions?started_before=2026-05-21 07:00:00")
        data = resp.get_json()
        sids = {s["session_id"] for s in data["items"]}
        assert sids == {"a1", "b1"}
        assert data["total"] == 2

    def test_started_before_is_inclusive(self, client):
        """A session started exactly at started_before must be included."""
        resp = client.get("/api/sessions?started_before=2026-05-21 07:00:00")
        data = resp.get_json()
        sids = {s["session_id"] for s in data["items"]}
        assert "a1" in sids

    def test_combined_after_and_before_range(self, client):
        """Both bounds together must return only sessions inside the range."""
        resp = client.get(
            "/api/sessions"
            "?started_after=2026-05-21 06:30:00"
            "&started_before=2026-05-21 07:30:00"
        )
        data = resp.get_json()
        sids = {s["session_id"] for s in data["items"]}
        assert sids == {"a1"}
        assert data["total"] == 1

    def test_inverted_range_returns_empty_without_error(self, client):
        """started_after later than started_before must yield an empty
        result set, not a 400."""
        resp = client.get(
            "/api/sessions"
            "?started_after=2026-05-21 08:00:00"
            "&started_before=2026-05-21 06:00:00"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_date_filter_combined_with_project_and_status(self, client):
        """Date range must AND together with existing project/status filters."""
        resp = client.get(
            "/api/sessions"
            "?project=alpha&status=completed"
            "&started_after=2026-05-21 00:00:00"
        )
        data = resp.get_json()
        assert data["total"] == 1
        assert data["items"][0]["session_id"] == "a1"

    def test_no_date_filters_returns_all_sessions(self, client):
        """Omitting both date params must not affect results (backward
        compatibility)."""
        resp = client.get("/api/sessions")
        data = resp.get_json()
        assert data["total"] == 3

    def test_invalid_started_after_format_returns_400(self, client):
        """Malformed started_after must return 400 with param='started_after'."""
        resp = client.get("/api/sessions?started_after=2026-05-21")
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["error"]["code"] == "invalid_param"
        assert body["error"]["param"] == "started_after"

    def test_invalid_started_before_format_returns_400(self, client):
        """Malformed started_before must return 400 with param='started_before'."""
        resp = client.get("/api/sessions?started_before=not-a-date")
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["error"]["code"] == "invalid_param"
        assert body["error"]["param"] == "started_before"

    def test_session_with_unparsable_started_is_skipped(self, tmp_path):
        """A session whose 'started' field can't be parsed must be skipped
        when a date filter is active, not raise an error."""
        sessions_dir = tmp_path
        _write(
            sessions_dir / "bad.fn",
            _FN_TMPL.format(
                project="alpha",
                sid="bad1",
                started="not-a-real-timestamp",
                ended="2026-05-21 07:00:10",
                dur=10,
                status="completed",
            ),
        )
        _write(
            sessions_dir / "good.fn",
            _FN_TMPL.format(
                project="alpha",
                sid="good1",
                started="2026-05-21 07:00:00",
                ended="2026-05-21 07:00:10",
                dur=10,
                status="completed",
            ),
        )
        scanner = FennScanner(extra_dirs=[str(sessions_dir)])

        import fenn.dashboard.app as app_module

        original = app_module.scanner
        app_module.scanner = scanner
        app.config["TESTING"] = True
        try:
            with app.test_client() as c:
                with c.session_transaction() as sess:
                    sess["user"] = {"email": "test@example.com"}
                resp = c.get("/api/sessions?started_after=2026-05-21 00:00:00")
                assert resp.status_code == 200
                data = resp.get_json()
                sids = {s["session_id"] for s in data["items"]}
                assert "bad1" not in sids
                assert "good1" in sids
        finally:
            app_module.scanner = original
