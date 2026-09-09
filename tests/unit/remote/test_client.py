"""Tests for `fenn.remote.client.RemoteClient._request` error handling."""

from unittest.mock import Mock

import pytest
import requests

from fenn.exceptions import AuthError, InsufficientCreditsError, NetworkError
from fenn.remote.client import RemoteClient


def _client(response_or_exc) -> RemoteClient:
    client = RemoteClient(host="https://example.test", api_key="fk_test")
    if isinstance(response_or_exc, BaseException):
        client._session.request = Mock(side_effect=response_or_exc)
    else:
        client._session.request = Mock(return_value=response_or_exc)
    return client


@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.ConnectionError("connection refused"),
        requests.exceptions.SSLError("certificate verify failed"),
        requests.exceptions.ConnectTimeout("connect timed out"),
        requests.exceptions.ReadTimeout("read timed out"),
    ],
    ids=["connection", "ssl", "connect-timeout", "read-timeout"],
)
def test_request_wraps_transport_errors_as_network_error(exc):
    client = _client(exc)

    with pytest.raises(NetworkError):
        client.me()


def test_request_returns_response_on_success():
    ok = Mock(status_code=200)
    ok.json.return_value = {"credits": 10}

    assert _client(ok).me() == {"credits": 10}


def test_request_maps_http_status_to_typed_errors():
    unauthorized = Mock(status_code=401)
    unauthorized.json.return_value = {"detail": "revoked"}
    with pytest.raises(AuthError):
        _client(unauthorized).me()

    broke = Mock(status_code=402)
    broke.json.return_value = {"detail": "no credits"}
    with pytest.raises(InsufficientCreditsError):
        _client(broke).me()
