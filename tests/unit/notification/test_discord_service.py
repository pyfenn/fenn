import os

import pytest
import requests

from fenn.notification.services.discord import Discord


@pytest.fixture(scope="class")
def message(fake):
    return fake.sentence()


@pytest.fixture
def mock_discord_response(monkeypatch, requests_mock):
    def _mock(url, status_code, response_body):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", url)
        requests_mock.post(url, status_code=status_code, json=response_body)

        return requests_mock

    return _mock


@pytest.fixture
def send_message_to_discord(mock_discord_response):
    def _send(url, message, status_code, response_body):
        mock_result = mock_discord_response(url, status_code, response_body)

        Discord(url).send_notification(message)

        return mock_result

    return _send


class TestDiscord:
    def test_discord_service_setup_error(self, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

        with pytest.raises(KeyError):
            Discord(
                os.environ["DISCORD_WEBHOOK_URL"],
            )

    def test_send_notification_success(self, send_message_to_discord, message):
        request = send_message_to_discord(
            "https://discord.com/api/webhooks/123/abc", message, 204, {}
        ).last_request

        assert request.json() == {"content": message, "username": "fenn"}

    def test_send_notification_error(self, send_message_to_discord, message):
        with pytest.raises(requests.exceptions.RequestException) as exc_info:
            send_message_to_discord(
                "https://discord.com/api/webhooks/123/abc",
                message,
                400,
                {"error": "test"},
            )

        assert "400" in str(exc_info.value)

    def test_notify_on_success_false_blocks_success(self):
        """Test that notify_on_success=False blocks success notifications."""
        discord = Discord(
            "https://discord.com/api/webhooks/123/abc", notify_on_success=False
        )
        assert discord.should_notify(is_success=True) is False

    def test_notify_on_success_false_allows_failure(self):
        """Test that notify_on_success=False allows failure notifications."""
        discord = Discord(
            "https://discord.com/api/webhooks/123/abc", notify_on_success=False
        )
        assert discord.should_notify(is_success=False) is True

    def test_notify_on_success_true_allows_both(self):
        """Test that notify_on_success=True allows both success and failure notifications."""
        discord = Discord(
            "https://discord.com/api/webhooks/123/abc", notify_on_success=True
        )
        assert discord.should_notify(is_success=True) is True
        assert discord.should_notify(is_success=False) is True

    def test_default_notify_on_success_is_false(self):
        """Test that the default value of notify_on_success is False."""
        discord = Discord("https://discord.com/api/webhooks/123/abc")
        assert discord.should_notify(is_success=True) is False
