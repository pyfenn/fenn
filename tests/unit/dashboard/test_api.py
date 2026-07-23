"""Unit tests for the /api/sessions endpoint (Pick 3)."""

import re
import shutil
import zipfile
from io import BytesIO

import pytest
import requests

from fenn.dashboard.app import app
from fenn.dashboard.runner import RunningTemplate, TemplateLaunchError
from fenn.dashboard.scanner import FennScanner
from fenn.dashboard.templates_registry import TemplateEntry, TemplatesRegistry

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


def _extract_csrf_token(html: str) -> str | None:
    meta = re.search(r'<meta name="csrf-token" content="([^"]+)"', html)
    if meta:
        return meta.group(1)
    hidden = re.search(r'name="csrf_token" value="([^"]+)"', html)
    if hidden:
        return hidden.group(1)
    return None


@pytest.fixture()
def scanner_with_sessions(tmp_path, monkeypatch):
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

    monkeypatch.setenv(
        "FENN_DASHBOARD_OVERRIDES_PATH", str(tmp_path / "dashboard_overrides.json")
    )
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


@pytest.fixture()
def client_no_auth(scanner_with_sessions, monkeypatch):
    """Flask test client using scanner fixture, but without logged-in user."""
    import fenn.dashboard.app as app_module

    monkeypatch.setattr(app_module.token_store, "load", lambda: None)
    original = app_module.scanner
    app_module.scanner = scanner_with_sessions
    app.config["TESTING"] = True
    with app.test_client() as c:
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


class TestSessionMutationRoutes:
    """Rename/delete routes should enforce validation, CSRF, and auth."""

    def test_rename_session_success(self, client):
        token = _extract_csrf_token(client.get("/").get_data(as_text=True))
        assert token

        resp = client.post(
            "/api/session/alpha/a1/rename",
            json={"display_name": "Alpha baseline"},
            headers={"X-CSRFToken": token},
        )
        assert resp.status_code == 200
        assert resp.get_json()["display_name"] == "Alpha baseline"

        check = client.get("/api/session/alpha/a1")
        assert check.status_code == 200
        assert check.get_json()["display_name"] == "Alpha baseline"

    def test_rename_empty_name_returns_400(self, client):
        token = _extract_csrf_token(client.get("/").get_data(as_text=True))
        assert token

        resp = client.post(
            "/api/session/alpha/a1/rename",
            json={"display_name": "   "},
            headers={"X-CSRFToken": token},
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["error"]["code"] == "invalid_param"
        assert body["error"]["param"] == "display_name"

    def test_delete_session_success(self, client):
        token = _extract_csrf_token(client.get("/").get_data(as_text=True))
        assert token

        resp = client.post(
            "/api/session/alpha/a2/delete",
            json={},
            headers={"X-CSRFToken": token},
        )
        assert resp.status_code == 200
        assert resp.get_json()["deleted"] is True

        check = client.get("/api/session/alpha/a2")
        assert check.status_code == 404

    def test_delete_unknown_session_returns_404(self, client):
        token = _extract_csrf_token(client.get("/").get_data(as_text=True))
        assert token

        resp = client.post(
            "/api/session/alpha/nope/delete",
            json={},
            headers={"X-CSRFToken": token},
        )
        assert resp.status_code == 404

    def test_archive_and_restore_session_success(self, client):
        token = _extract_csrf_token(client.get("/").get_data(as_text=True))
        assert token

        archive_resp = client.post(
            "/api/session/alpha/a1/archive",
            json={},
            headers={"X-CSRFToken": token},
        )
        assert archive_resp.status_code == 200
        assert archive_resp.get_json()["archived"] is True

        detail = client.get("/api/session/alpha/a1")
        assert detail.status_code == 200
        assert detail.get_json()["archived"] is True

        restore_resp = client.post(
            "/api/session/alpha/a1/restore",
            json={},
            headers={"X-CSRFToken": token},
        )
        assert restore_resp.status_code == 200
        assert restore_resp.get_json()["archived"] is False

        detail_after = client.get("/api/session/alpha/a1")
        assert detail_after.status_code == 200
        assert detail_after.get_json()["archived"] is False

    def test_archive_unknown_session_returns_404(self, client):
        token = _extract_csrf_token(client.get("/").get_data(as_text=True))
        assert token

        resp = client.post(
            "/api/session/alpha/nope/archive",
            json={},
            headers={"X-CSRFToken": token},
        )
        assert resp.status_code == 404

    def test_mutation_routes_without_auth_redirect_to_connect(self, client_no_auth):
        token = _extract_csrf_token(
            client_no_auth.get("/connect").get_data(as_text=True)
        )
        assert token

        rename_resp = client_no_auth.post(
            "/api/session/alpha/a1/rename",
            json={"display_name": "x"},
            headers={"X-CSRFToken": token},
        )
        delete_resp = client_no_auth.post(
            "/api/session/alpha/a1/delete",
            json={},
            headers={"X-CSRFToken": token},
        )

        assert rename_resp.status_code == 302
        assert "/connect" in rename_resp.headers.get("Location", "")
        assert delete_resp.status_code == 302
        assert "/connect" in delete_resp.headers.get("Location", "")

    def test_mutation_routes_without_csrf_return_400(self, client):
        rename_resp = client.post(
            "/api/session/alpha/a1/rename",
            json={"display_name": "x"},
        )
        delete_resp = client.post("/api/session/alpha/a1/delete", json={})
        archive_resp = client.post("/api/session/alpha/a1/archive", json={})
        restore_resp = client.post("/api/session/alpha/a1/restore", json={})

        assert rename_resp.status_code == 400
        assert delete_resp.status_code == 400
        assert archive_resp.status_code == 400
        assert restore_resp.status_code == 400

    def test_api_sessions_include_archived_query_param(self, client):
        token = _extract_csrf_token(client.get("/").get_data(as_text=True))
        assert token
        archive_resp = client.post(
            "/api/session/alpha/a1/archive",
            json={},
            headers={"X-CSRFToken": token},
        )
        assert archive_resp.status_code == 200

        default_resp = client.get("/api/sessions")
        default_ids = {item["session_id"] for item in default_resp.get_json()["items"]}
        assert "a1" not in default_ids

        include_resp = client.get("/api/sessions?include_archived=true")
        include_ids = {item["session_id"] for item in include_resp.get_json()["items"]}
        assert "a1" in include_ids


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


class TestApiTemplates:
    """Tests for the template-listing and template-pull endpoints."""

    def test_returns_sorted_public_templates(
        self,
        client,
        requests_mock,
    ):
        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents",
            status_code=200,
            json=[
                {"name": "chatbot", "type": "dir"},
                {"name": "base", "type": "dir"},
                {"name": "internal-dev-only", "type": "dir"},
                {"name": "README.md", "type": "file"},
            ],
        )

        response = client.get("/api/templates")

        assert response.status_code == 200
        assert response.get_json() == {
            "templates": ["base", "chatbot"],
            "total": 2,
        }

    def test_returns_502_when_github_is_unavailable(
        self,
        client,
        requests_mock,
    ):
        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents",
            exc=requests.exceptions.ConnectionError("offline"),
        )

        response = client.get("/api/templates")

        assert response.status_code == 502

        body = response.get_json()

        assert body["error"]["code"] == "template_list_unavailable"
        assert "Failed to fetch template list" in body["error"]["message"]

    def test_requires_authentication(
        self,
        client_no_auth,
    ):
        response = client_no_auth.get("/api/templates")

        assert response.status_code == 302
        assert "/connect" in response.headers["Location"]

    def test_pull_template_success(
        self,
        client,
        requests_mock,
        tmp_path,
    ):
        target_dir = tmp_path / "downloaded-template"
        archive = BytesIO()

        with zipfile.ZipFile(archive, "w") as zip_file:
            zip_file.writestr(
                "templates-main/base/main.py",
                "print('hello')",
            )
            zip_file.writestr(
                "templates-main/base/fenn.yaml",
                "project: test",
            )

        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents/base",
            status_code=200,
            json={
                "name": "base",
                "type": "dir",
            },
        )

        requests_mock.get(
            "https://github.com/pyfenn/templates/archive/refs/heads/main.zip",
            status_code=200,
            content=archive.getvalue(),
        )

        token = _extract_csrf_token(client.get("/").get_data(as_text=True))
        assert token

        response = client.post(
            "/api/templates/pull",
            json={
                "template": "base",
                "path": str(target_dir),
                "force": False,
            },
            headers={
                "X-CSRFToken": token,
            },
        )

        assert response.status_code == 200

        body = response.get_json()

        assert body["template"] == "base"
        assert body["downloaded"] is True
        assert body["path"] == str(target_dir.resolve())

        assert (target_dir / "main.py").read_text(encoding="utf-8") == "print('hello')"

        assert (target_dir / "fenn.yaml").exists()

    def test_pull_rejects_non_empty_directory(
        self,
        client,
        tmp_path,
    ):
        target_dir = tmp_path / "non-empty"
        target_dir.mkdir()

        (target_dir / "existing.txt").write_text(
            "existing",
            encoding="utf-8",
        )

        token = _extract_csrf_token(client.get("/").get_data(as_text=True))
        assert token

        response = client.post(
            "/api/templates/pull",
            json={
                "template": "base",
                "path": str(target_dir),
                "force": False,
            },
            headers={
                "X-CSRFToken": token,
            },
        )

        assert response.status_code == 409

        body = response.get_json()

        assert body["error"]["code"] == "target_not_empty"
        assert body["error"]["param"] == "path"

    def test_pull_rejects_invalid_payload(
        self,
        client,
    ):
        token = _extract_csrf_token(client.get("/").get_data(as_text=True))
        assert token

        response = client.post(
            "/api/templates/pull",
            json={
                "template": "",
                "path": "",
                "force": "false",
            },
            headers={
                "X-CSRFToken": token,
            },
        )

        assert response.status_code == 400

        body = response.get_json()

        assert body["error"]["code"] == "invalid_param"
        assert body["error"]["param"] == "template"

    def test_pull_template_not_found(
        self,
        client,
        requests_mock,
        tmp_path,
    ):
        target_dir = tmp_path / "missing-template"

        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents/missing",
            status_code=404,
        )

        token = _extract_csrf_token(client.get("/").get_data(as_text=True))
        assert token

        response = client.post(
            "/api/templates/pull",
            json={
                "template": "missing",
                "path": str(target_dir),
                "force": False,
            },
            headers={
                "X-CSRFToken": token,
            },
        )

        assert response.status_code == 404

        body = response.get_json()

        assert body["error"]["code"] == "template_not_found"
        assert body["error"]["param"] == "template"

    def test_pull_requires_csrf(
        self,
        client,
        tmp_path,
    ):
        target_dir = tmp_path / "csrf-target"

        response = client.post(
            "/api/templates/pull",
            json={
                "template": "base",
                "path": str(target_dir),
                "force": False,
            },
        )

        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Templates page and local templates API
# ---------------------------------------------------------------------------


def _add_template(
    base_dir,
    registry,
    name,
    source="base",
    pulled_at="2026-07-21T10:00:00+00:00",
):
    template_dir = base_dir / name
    template_dir.mkdir()
    registry.add_or_update(
        TemplateEntry(
            name=name,
            path=str(template_dir),
            source_template=source,
            pulled_at=pulled_at,
        )
    )
    return template_dir


@pytest.fixture()
def templates_registry_with_entries(tmp_path):
    """A TemplatesRegistry pre-populated with two locally pulled templates."""
    registry = TemplatesRegistry(registry_path=tmp_path / "templates_registry.json")
    _add_template(
        tmp_path,
        registry,
        "tmpl_a",
        source="base",
        pulled_at="2026-07-20T10:00:00+00:00",
    )
    _add_template(
        tmp_path,
        registry,
        "tmpl_b",
        source="vision",
        pulled_at="2026-07-21T10:00:00+00:00",
    )
    return registry


@pytest.fixture()
def empty_templates_registry(tmp_path):
    return TemplatesRegistry(registry_path=tmp_path / "templates_registry.json")


@pytest.fixture()
def templates_client(templates_registry_with_entries):
    """Flask test client, logged in, wired to a registry with known templates."""
    import fenn.dashboard.app as app_module

    original = app_module.templates_registry
    app_module.templates_registry = templates_registry_with_entries
    app.config["TESTING"] = True
    with app.test_client() as c:
        with c.session_transaction() as sess:
            sess["user"] = {"email": "test@example.com"}
        yield c
    app_module.templates_registry = original


@pytest.fixture()
def empty_templates_client(empty_templates_registry):
    """Flask test client, logged in, wired to an empty registry."""
    import fenn.dashboard.app as app_module

    original = app_module.templates_registry
    app_module.templates_registry = empty_templates_registry
    app.config["TESTING"] = True
    with app.test_client() as c:
        with c.session_transaction() as sess:
            sess["user"] = {"email": "test@example.com"}
        yield c
    app_module.templates_registry = original


@pytest.fixture()
def templates_client_no_auth(templates_registry_with_entries, monkeypatch):
    """Flask test client wired to a populated registry, but not logged in."""
    import fenn.dashboard.app as app_module

    monkeypatch.setattr(app_module.token_store, "load", lambda: None)
    original = app_module.templates_registry
    app_module.templates_registry = templates_registry_with_entries
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
    app_module.templates_registry = original


class TestApiLocalTemplates:
    def test_returns_registered_templates(self, templates_client):
        resp = templates_client.get("/api/templates/local")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 2
        names = {t["name"] for t in data["templates"]}
        assert names == {"tmpl_a", "tmpl_b"}

    def test_response_entry_shape(self, templates_client):
        resp = templates_client.get("/api/templates/local")
        entries = resp.get_json()["templates"]
        [entry] = [e for e in entries if e["name"] == "tmpl_a"]
        assert set(entry.keys()) == {"name", "path", "source_template", "pulled_at"}
        assert entry["source_template"] == "base"

    def test_returns_empty_list_when_no_templates_pulled(self, empty_templates_client):
        resp = empty_templates_client.get("/api/templates/local")
        assert resp.status_code == 200
        assert resp.get_json() == {"templates": [], "total": 0}

    def test_deleted_template_directory_is_not_returned(self, templates_client):
        initial = templates_client.get("/api/templates/local").get_json()
        tmpl_a_path = next(
            entry["path"] for entry in initial["templates"] if entry["name"] == "tmpl_a"
        )
        shutil.rmtree(tmpl_a_path)

        resp = templates_client.get("/api/templates/local")
        data = resp.get_json()
        names = {t["name"] for t in data["templates"]}
        assert names == {"tmpl_b"}
        assert data["total"] == 1

    def test_requires_authentication(self, templates_client_no_auth):
        resp = templates_client_no_auth.get("/api/templates/local")
        assert resp.status_code == 302
        assert "/connect" in resp.headers["Location"]

    def test_does_not_collide_with_remote_templates_endpoint(
        self, templates_client, requests_mock
    ):
        """`/api/templates` (remote-available) and `/api/templates/local`
        (locally downloaded) must stay distinct routes with distinct data."""
        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents",
            status_code=200,
            json=[{"name": "base", "type": "dir"}],
        )

        remote_resp = templates_client.get("/api/templates")
        local_resp = templates_client.get("/api/templates/local")

        assert remote_resp.status_code == 200
        assert local_resp.status_code == 200
        assert remote_resp.get_json()["templates"] == ["base"]
        assert {t["name"] for t in local_resp.get_json()["templates"]} == {
            "tmpl_a",
            "tmpl_b",
        }


class TestTemplatesPage:
    def test_requires_authentication(self, templates_client_no_auth):
        resp = templates_client_no_auth.get("/templates")
        assert resp.status_code == 302
        assert "/connect" in resp.headers["Location"]

    def test_renders_downloaded_templates(self, templates_client):
        resp = templates_client.get("/templates")
        assert resp.status_code == 200
        assert b"tmpl_a" in resp.data
        assert b"tmpl_b" in resp.data

    def test_renders_empty_state_when_no_templates(self, empty_templates_client):
        resp = empty_templates_client.get("/templates")
        assert resp.status_code == 200
        assert b"No templates downloaded yet" in resp.data

    def test_does_not_render_deleted_template(self, templates_client):
        initial = templates_client.get("/api/templates/local").get_json()
        tmpl_a_path = next(
            entry["path"] for entry in initial["templates"] if entry["name"] == "tmpl_a"
        )
        shutil.rmtree(tmpl_a_path)

        resp = templates_client.get("/templates")
        assert b"tmpl_a" not in resp.data
        assert b"tmpl_b" in resp.data

    def test_active_page_marks_nav_item_active(self, templates_client):
        resp = templates_client.get("/templates")
        assert b'href="/templates" class="nav-item active"' in resp.data


class TestTemplatesNavigation:
    def test_templates_link_present_on_home_page(self, templates_client):
        resp = templates_client.get("/")
        assert resp.status_code == 200
        assert b'href="/templates"' in resp.data


class TestApiTemplateRun:
    """Tests for POST /api/templates/run."""

    def test_run_rejects_unregistered_path(self, templates_client, tmp_path):
        token = _extract_csrf_token(templates_client.get("/").get_data(as_text=True))
        resp = templates_client.post(
            "/api/templates/run",
            json={"path": str(tmp_path / "not-registered")},
            headers={"X-CSRFToken": token},
        )
        assert resp.status_code == 404
        body = resp.get_json()
        assert body["error"]["code"] == "template_not_registered"
        assert body["error"]["param"] == "path"

    def test_run_rejects_invalid_payload(self, templates_client):
        token = _extract_csrf_token(templates_client.get("/").get_data(as_text=True))
        resp = templates_client.post(
            "/api/templates/run",
            json={"path": ""},
            headers={"X-CSRFToken": token},
        )
        assert resp.status_code == 400
        assert resp.get_json()["error"]["param"] == "path"

    def test_run_requires_csrf(self, templates_client, tmp_path):
        resp = templates_client.post(
            "/api/templates/run",
            json={"path": str(tmp_path)},
        )
        assert resp.status_code == 400

    def test_run_requires_authentication(self, templates_client_no_auth, tmp_path):
        token = _extract_csrf_token(
            templates_client_no_auth.get("/connect").get_data(as_text=True)
        )
        resp = templates_client_no_auth.post(
            "/api/templates/run",
            json={"path": str(tmp_path)},
            headers={"X-CSRFToken": token},
        )
        assert resp.status_code == 302
        assert "/connect" in resp.headers["Location"]

    def test_run_success_launches_and_returns_payload(
        self, templates_client, templates_registry_with_entries, monkeypatch
    ):
        import fenn.dashboard.app as app_module

        entries = templates_registry_with_entries.list_templates()
        target = entries[0]["path"]

        fake_process = type(
            "FakeProcess", (), {"pid": 4242, "poll": lambda self: None}
        )()
        fake_running = RunningTemplate(
            run_id="abc123",
            template_path=app_module.Path(target),
            log_dir=app_module.Path(target) / "logger",
            process=fake_process,
            started_at=0.0,
        )
        monkeypatch.setattr(
            app_module.template_runner, "launch", lambda *a, **k: fake_running
        )

        token = _extract_csrf_token(templates_client.get("/").get_data(as_text=True))
        resp = templates_client.post(
            "/api/templates/run",
            json={"path": target},
            headers={"X-CSRFToken": token},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["run_id"] == "abc123"
        assert body["pid"] == 4242
        assert body["launched"] is True

    def test_run_launch_failure_returns_502(
        self, templates_client, templates_registry_with_entries, monkeypatch
    ):
        import fenn.dashboard.app as app_module

        entries = templates_registry_with_entries.list_templates()
        target = entries[0]["path"]

        def _raise(*a, **k):
            raise TemplateLaunchError("boom: crashed on startup")

        monkeypatch.setattr(app_module.template_runner, "launch", _raise)

        token = _extract_csrf_token(templates_client.get("/").get_data(as_text=True))
        resp = templates_client.post(
            "/api/templates/run",
            json={"path": target},
            headers={"X-CSRFToken": token},
        )
        assert resp.status_code == 502
        body = resp.get_json()
        assert body["error"]["code"] == "template_launch_failed"
        assert "boom: crashed on startup" in body["error"]["message"]
