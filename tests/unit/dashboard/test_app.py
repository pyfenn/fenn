"""Tests for fenn/dashboard/app.py"""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from fenn.dashboard.app import (
    _ApiBadRequest,
    _try_stored_session,
    duration_filter,
    filesize_filter,
    short_id_filter,
)
from fenn.exceptions import AuthUnreachableError, InvalidTokenError

# ══════════════════════════════════════════════════════════════════════════════
# App fixture
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def app():
    from fenn.dashboard.app import app as flask_app

    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def authed_client(app):
    """Test client with a pre-set user session."""
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user"] = {"user_id": "u1", "email": "test@example.com"}
    return client


# ══════════════════════════════════════════════════════════════════════════════
# Template filters
# ══════════════════════════════════════════════════════════════════════════════


class TestTemplateFilters:
    def test_short_id_filter_truncates(self, app):
        with app.test_request_context("/"):
            assert short_id_filter("abcdefghijk") == "abcdefgh"

    def test_short_id_filter_passthrough_short(self, app):
        with app.test_request_context("/"):
            assert short_id_filter("abc") == "abc"

    def test_duration_filter_delegates_to_scanner(self, app):
        with app.test_request_context("/"):
            from fenn.dashboard import app as app_mod

            app_mod.scanner.format_duration = MagicMock(return_value="1m 30s")
            result = duration_filter(90)
        assert result == "1m 30s"

    def test_filesize_filter_delegates_to_scanner(self, app):
        with app.test_request_context("/"):
            from fenn.dashboard import app as app_mod

            app_mod.scanner.format_size = MagicMock(return_value="1.5 KB")
            result = filesize_filter(1536)
        assert result == "1.5 KB"


# ══════════════════════════════════════════════════════════════════════════════
# _parse_int_arg
# ══════════════════════════════════════════════════════════════════════════════


class TestParseIntArg:
    def _parse(self, name, raw, default=20, min_v=1, max_v=200):
        from fenn.dashboard.app import _parse_int_arg

        return _parse_int_arg(name, raw, default, min_v, max_v)

    def test_none_returns_default(self):
        assert self._parse("limit", None) == 20

    def test_empty_string_returns_default(self):
        assert self._parse("limit", "") == 20

    def test_valid_integer(self):
        assert self._parse("limit", "50") == 50

    def test_non_integer_raises(self):
        with pytest.raises(_ApiBadRequest, match="must be an integer"):
            self._parse("limit", "abc")

    def test_below_min_raises(self):
        with pytest.raises(_ApiBadRequest, match="must be between"):
            self._parse("limit", "0", min_v=1)

    def test_above_max_raises(self):
        with pytest.raises(_ApiBadRequest, match="must be between"):
            self._parse("limit", "201", max_v=200)

    def test_boundary_min_valid(self):
        assert self._parse("limit", "1", min_v=1) == 1

    def test_boundary_max_valid(self):
        assert self._parse("limit", "200", max_v=200) == 200


# ══════════════════════════════════════════════════════════════════════════════
# Auth routes
# ══════════════════════════════════════════════════════════════════════════════


class TestConnectRoute:
    def test_redirects_to_index_when_logged_in(self, authed_client):
        with patch("fenn.dashboard.app.scanner") as mock_scanner:
            mock_scanner.get_overview.return_value = {}
            with patch("fenn.dashboard.app.render_template", return_value=""):
                resp = authed_client.get("/connect")
        assert resp.status_code == 302
        assert "/connect" not in resp.headers["Location"]

    def test_renders_connect_page_when_not_logged_in(self, client):
        with patch("fenn.dashboard.app.token_store.load", return_value=None):
            resp = client.get("/connect")
        assert resp.status_code == 200

    def test_pops_pending_info_message(self, client):
        with client.session_transaction() as sess:
            sess["pending_info"] = "Your session expired."
        with patch("fenn.dashboard.app.token_store.load", return_value=None):
            with patch(
                "fenn.dashboard.app.render_template", return_value="ok"
            ) as mock_render:
                client.get("/connect")
        _, kwargs = mock_render.call_args
        assert kwargs.get("info_message") == "Your session expired."


class TestConnectStart:
    def test_redirects_to_pyfenn_with_state(self, client):
        resp = client.post("/connect/start")
        assert resp.status_code == 302
        assert "pyfenn.com" in resp.headers["Location"]
        assert "state=" in resp.headers["Location"]

    def test_stores_oauth_state_in_session(self, client):
        client.post("/connect/start")
        with client.session_transaction() as sess:
            assert "oauth_state" in sess


class TestConnectCallback:
    def test_state_mismatch_returns_400(self, client):
        with client.session_transaction() as sess:
            sess["oauth_state"] = "correct-state"
        resp = client.get("/connect/callback?state=wrong-state&code=abc")
        assert resp.status_code == 400

    def test_missing_state_returns_400(self, client):
        with patch("fenn.dashboard.app.render_template", return_value="err"):
            resp = client.get("/connect/callback?code=abc")
        assert resp.status_code == 400

    def test_invalid_token_returns_401(self, client):
        state = "test-state"
        with client.session_transaction() as sess:
            sess["oauth_state"] = state
        with patch(
            "fenn.dashboard.app.dashboard_auth.exchange_code",
            side_effect=InvalidTokenError("bad code"),
        ):
            with patch("fenn.dashboard.app.render_template", return_value="err"):
                resp = client.get(f"/connect/callback?state={state}&code=bad")
        assert resp.status_code == 401

    def test_auth_unreachable_returns_503(self, client):
        state = "test-state"
        with client.session_transaction() as sess:
            sess["oauth_state"] = state
        with patch(
            "fenn.dashboard.app.dashboard_auth.exchange_code",
            side_effect=AuthUnreachableError("down"),
        ):
            with patch("fenn.dashboard.app.render_template", return_value="err"):
                resp = client.get(f"/connect/callback?state={state}&code=abc")
        assert resp.status_code == 503

    def test_success_sets_session_and_redirects(self, client):
        state = "test-state"
        with client.session_transaction() as sess:
            sess["oauth_state"] = state

        fake_result = {
            "user_id": "u1",
            "email": "a@b.com",
            "token": "fdt_" + "x" * 43,
        }
        with patch(
            "fenn.dashboard.app.dashboard_auth.exchange_code", return_value=fake_result
        ):
            with patch("fenn.dashboard.app.token_store.save") as mock_save:
                resp = client.get(f"/connect/callback?state={state}&code=valid-code")

        assert resp.status_code == 302
        mock_save.assert_called_once()
        with client.session_transaction() as sess:
            assert sess.get("user", {}).get("user_id") == "u1"


class TestLogout:
    def test_clears_session_and_redirects(self, authed_client):
        with patch("fenn.dashboard.app.token_store.clear") as mock_clear:
            resp = authed_client.post("/logout")
        assert resp.status_code == 302
        mock_clear.assert_called_once()
        with authed_client.session_transaction() as sess:
            assert "user" not in sess


# ══════════════════════════════════════════════════════════════════════════════
# _try_stored_session
# ══════════════════════════════════════════════════════════════════════════════


class TestTryStoredSession:
    def test_returns_none_when_no_stored_session(self, app):
        with app.test_request_context("/"):
            with patch("fenn.dashboard.app.token_store.load", return_value=None):
                result = _try_stored_session()
        assert result is None

    def test_clears_and_redirects_on_invalid_token(self, app):
        stored = {
            "token": "fdt_" + "a" * 43,
            "user": {"user_id": "u1", "email": "a@b.com"},
        }
        with app.test_request_context("/"):
            with patch("fenn.dashboard.app.token_store.load", return_value=stored):
                with patch(
                    "fenn.dashboard.app.dashboard_auth.validate_token",
                    side_effect=InvalidTokenError("revoked"),
                ):
                    with patch("fenn.dashboard.app.token_store.clear") as mock_clear:
                        result = _try_stored_session()
        assert result is not None  # redirect response
        mock_clear.assert_called_once()

    def test_uses_cached_user_on_unreachable(self, app):
        stored = {
            "token": "fdt_" + "a" * 43,
            "user": {"user_id": "u1", "email": "a@b.com"},
        }
        with app.test_request_context("/"):
            with patch("fenn.dashboard.app.token_store.load", return_value=stored):
                with patch(
                    "fenn.dashboard.app.dashboard_auth.validate_token",
                    side_effect=AuthUnreachableError("offline"),
                ):
                    result = _try_stored_session()
        assert result is None  # no redirect — used cached identity

    def test_refreshes_token_on_success(self, app):
        stored = {
            "token": "fdt_" + "a" * 43,
            "user": {"user_id": "u1", "email": "old@b.com"},
        }
        fresh_user = {"user_id": "u1", "email": "new@b.com"}
        with app.test_request_context("/"):
            with patch("fenn.dashboard.app.token_store.load", return_value=stored):
                with patch(
                    "fenn.dashboard.app.dashboard_auth.validate_token",
                    return_value=fresh_user,
                ):
                    with patch("fenn.dashboard.app.token_store.save") as mock_save:
                        _try_stored_session()
        mock_save.assert_called_once_with(stored["token"], fresh_user)


# ══════════════════════════════════════════════════════════════════════════════
# API routes
# ══════════════════════════════════════════════════════════════════════════════


class TestUploadFile:
    def test_upload_file_success(self, app, authed_client, tmp_path):
        app.config["UPLOAD_FOLDER"] = tmp_path

        response = authed_client.post(
            "/api/uploads",
            data={
                "file": (BytesIO(b"sample data"), "train.csv"),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 201
        assert response.get_json() == {
            "uploaded": True,
            "filename": "train.csv",
            "size": 11,
        }

        uploaded_file = tmp_path / "train.csv"
        assert uploaded_file.exists()
        assert uploaded_file.read_bytes() == b"sample data"

    def test_upload_without_file_returns_400(
        self,
        app,
        authed_client,
        tmp_path,
    ):
        app.config["UPLOAD_FOLDER"] = tmp_path

        response = authed_client.post(
            "/api/uploads",
            data={},
            content_type="multipart/form-data",
        )

        assert response.status_code == 400

        body = response.get_json()
        assert body["error"]["code"] == "missing_file"
        assert body["error"]["param"] == "file"

    def test_upload_with_empty_filename_returns_400(
        self,
        app,
        authed_client,
        tmp_path,
    ):
        app.config["UPLOAD_FOLDER"] = tmp_path

        response = authed_client.post(
            "/api/uploads",
            data={
                "file": (BytesIO(b"sample data"), ""),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 400

        body = response.get_json()
        assert body["error"]["code"] == "invalid_file"
        assert body["error"]["param"] == "file"

    def test_upload_duplicate_file_returns_409(
        self,
        app,
        authed_client,
        tmp_path,
    ):
        app.config["UPLOAD_FOLDER"] = tmp_path
        (tmp_path / "train.csv").write_bytes(b"existing data")

        response = authed_client.post(
            "/api/uploads",
            data={
                "file": (BytesIO(b"new data"), "train.csv"),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 409

        body = response.get_json()
        assert body["error"]["code"] == "file_exists"

        assert (tmp_path / "train.csv").read_bytes() == b"existing data"

    def test_upload_sanitizes_filename(self, app, authed_client, tmp_path):
        app.config["UPLOAD_FOLDER"] = tmp_path

        response = authed_client.post(
            "/api/uploads",
            data={
                "file": (BytesIO(b"sample data"), "../../train.csv"),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 201
        assert response.get_json()["filename"] == "train.csv"
        assert (tmp_path / "train.csv").exists()

    def test_upload_invalid_filename_returns_400(
        self,
        app,
        authed_client,
        tmp_path,
    ):
        app.config["UPLOAD_FOLDER"] = tmp_path

        response = authed_client.post(
            "/api/uploads",
            data={
                "file": (BytesIO(b"sample data"), "../../../"),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 400

        body = response.get_json()
        assert body["error"]["code"] == "invalid_filename"

    def test_list_uploaded_files(self, app, authed_client, tmp_path):
        app.config["UPLOAD_FOLDER"] = tmp_path

        (tmp_path / "first.csv").write_bytes(b"abc")
        (tmp_path / "second.txt").write_bytes(b"hello")

        response = authed_client.get("/api/uploads")

        assert response.status_code == 200

        body = response.get_json()
        assert body["total"] == 2

        filenames = {file["filename"] for file in body["files"]}
        assert filenames == {"first.csv", "second.txt"}

        for file in body["files"]:
            assert "size" in file
            assert "modified_at" in file

    def test_list_uploaded_files_empty(self, app, authed_client, tmp_path):
        app.config["UPLOAD_FOLDER"] = tmp_path

        response = authed_client.get("/api/uploads")

        assert response.status_code == 200
        assert response.get_json() == {
            "files": [],
            "total": 0,
        }


class TestApiSessions:
    def test_returns_200_with_default_params(self, authed_client):
        from fenn.dashboard import app as app_mod

        app_mod.scanner.list_sessions = MagicMock(
            return_value={"sessions": [], "total": 0}
        )
        resp = authed_client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "sessions" in data

    def test_limit_out_of_range_returns_400(self, authed_client):
        resp = authed_client.get("/api/sessions?limit=999")
        assert resp.status_code == 400

    def test_invalid_started_after_returns_400(self, authed_client):
        resp = authed_client.get("/api/sessions?started_after=not-a-date")
        assert resp.status_code == 400

    def test_invalid_started_before_returns_400(self, authed_client):
        resp = authed_client.get("/api/sessions?started_before=not-a-date")
        assert resp.status_code == 400

    def test_valid_started_after(self, authed_client):
        from fenn.dashboard import app as app_mod

        app_mod.scanner.list_sessions = MagicMock(
            return_value={"sessions": [], "total": 0}
        )
        resp = authed_client.get("/api/sessions?started_after=2024-01-01 00:00:00")
        assert resp.status_code == 200

    def test_scanner_value_error_returns_400(self, authed_client):
        from fenn.dashboard import app as app_mod

        app_mod.scanner.list_sessions = MagicMock(
            side_effect=ValueError("status must be one of...")
        )
        resp = authed_client.get("/api/sessions?status=bad")
        assert resp.status_code == 400


class TestApiSessionArchiveRestore:
    def test_archive_existing_session(self, authed_client):
        from fenn.dashboard import app as app_mod

        app_mod.scanner.archive_session = MagicMock(return_value=True)
        resp = authed_client.post("/api/session/myproject/sess123/archive")
        assert resp.status_code == 200
        assert resp.get_json() == {"archived": True}

    def test_archive_nonexistent_returns_404(self, authed_client):
        from fenn.dashboard import app as app_mod

        app_mod.scanner.archive_session = MagicMock(return_value=False)
        resp = authed_client.post("/api/session/myproject/sess999/archive")
        assert resp.status_code == 404

    def test_restore_existing_session(self, authed_client):
        from fenn.dashboard import app as app_mod

        app_mod.scanner.restore_session = MagicMock(return_value=True)
        resp = authed_client.post("/api/session/myproject/sess123/restore")
        assert resp.status_code == 200
        assert resp.get_json() == {"archived": False}

    def test_restore_nonexistent_returns_404(self, authed_client):
        from fenn.dashboard import app as app_mod

        app_mod.scanner.restore_session = MagicMock(return_value=False)
        resp = authed_client.post("/api/session/myproject/sess999/restore")
        assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# _require_login
# ══════════════════════════════════════════════════════════════════════════════


class TestRequireLogin:
    def test_public_endpoints_accessible_without_auth(self, client):
        with patch("fenn.dashboard.app.token_store.load", return_value=None):
            with patch("fenn.dashboard.app.render_template", return_value="ok"):
                resp = client.get("/connect")
        assert resp.status_code == 200

    def test_protected_endpoint_redirects_without_auth(self, client):
        with patch("fenn.dashboard.app.token_store.load", return_value=None):
            resp = client.get("/api/sessions")
        assert resp.status_code == 302

    def test_protected_endpoint_accessible_with_auth(self, authed_client):
        from fenn.dashboard import app as app_mod

        app_mod.scanner.list_sessions = MagicMock(
            return_value={"sessions": [], "total": 0}
        )
        resp = authed_client.get("/api/sessions")
        assert resp.status_code == 200


class TestUploadsPage:
    def test_uploads_page_renders(self, authed_client):
        response = authed_client.get("/uploads")

        assert response.status_code == 200
        assert b"Upload Files" in response.data
        assert b'id="upload-form"' in response.data
        assert b'action="/api/uploads"' in response.data
        assert b'method="POST"' in response.data
        assert b'name="csrf_token"' in response.data
