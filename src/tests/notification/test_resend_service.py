from unittest.mock import patch

import pytest

from fenn.notification.services.resend import Resend


@pytest.fixture(scope="class")
def message(fake):
    return fake.sentence()


@pytest.fixture
def mock_resend_config(monkeypatch):
    """Mock environment variables for Resend configuration."""

    def _mock(
        api_key="test_api_key",
        from_email="test@example.com",
        to_emails="recipient1@example.com,recipient2@example.com",
    ):
        monkeypatch.setenv("RESEND_API_KEY", api_key)
        monkeypatch.setenv("RESEND_FROM_EMAIL", from_email)
        monkeypatch.setenv("RESEND_TO_EMAILS", to_emails)

    return _mock


class TestResend:
    def test_resend_service_setup_error_missing_api_key(self, monkeypatch):
        """Test that Resend raises KeyError when RESEND_API_KEY is missing."""
        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        monkeypatch.setenv("RESEND_FROM_EMAIL", "test@example.com")
        monkeypatch.setenv("RESEND_TO_EMAILS", "recipient@example.com")

        with pytest.raises(KeyError):
            Resend()

    def test_resend_service_setup_error_missing_from_email(self, monkeypatch):
        """Test that Resend raises KeyError when RESEND_FROM_EMAIL is missing."""
        monkeypatch.setenv("RESEND_API_KEY", "test_key")
        monkeypatch.delenv("RESEND_FROM_EMAIL", raising=False)
        monkeypatch.setenv("RESEND_TO_EMAILS", "recipient@example.com")

        with pytest.raises(KeyError):
            Resend()

    def test_resend_service_setup_error_missing_to_emails(self, monkeypatch):
        """Test that Resend raises KeyError when RESEND_TO_EMAILS is missing."""
        monkeypatch.setenv("RESEND_API_KEY", "test_key")
        monkeypatch.setenv("RESEND_FROM_EMAIL", "test@example.com")
        monkeypatch.delenv("RESEND_TO_EMAILS", raising=False)

        with pytest.raises(KeyError):
            Resend()

    def test_resend_service_initialization(self, mock_resend_config):
        """Test that Resend service initializes correctly with valid configuration."""
        mock_resend_config()

        with patch("resend.api_key", None):
            service = Resend(subject="Test Subject")

            assert service._api_key == "test_api_key"
            assert service._from_email == "test@example.com"
            assert service._to_emails == [
                "recipient1@example.com",
                "recipient2@example.com",
            ]
            assert service._subject == "Test Subject"

    def test_send_notification_success(self, mock_resend_config, message):
        """Test successful email sending."""
        mock_resend_config()

        with patch("resend.Emails.send") as mock_send:
            mock_send.return_value = {"id": "test_email_id"}

            service = Resend(subject="Test Email")
            service.send_notification(message)

            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]

            assert call_args["from"] == "test@example.com"
            assert call_args["to"] == [
                "recipient1@example.com",
                "recipient2@example.com",
            ]
            assert call_args["subject"] == "Test Email"
            assert message in call_args["html"]

    def test_send_notification_multiple_recipients(self, mock_resend_config, message):
        """Test that emails are sent to multiple recipients."""
        mock_resend_config(to_emails="user1@test.com,user2@test.com,user3@test.com")

        with patch("resend.Emails.send") as mock_send:
            mock_send.return_value = {"id": "test_email_id"}

            service = Resend()
            service.send_notification(message)

            call_args = mock_send.call_args[0][0]
            assert len(call_args["to"]) == 3
            assert "user1@test.com" in call_args["to"]
            assert "user2@test.com" in call_args["to"]
            assert "user3@test.com" in call_args["to"]

    def test_send_notification_api_error_response(self, mock_resend_config, message):
        """Test handling of Resend API error response."""
        mock_resend_config()

        with patch("resend.Emails.send") as mock_send:
            mock_send.return_value = {"error": "Invalid API key"}

            service = Resend()

            with pytest.raises(Exception) as exc_info:
                service.send_notification(message)

            assert "Resend API error" in str(exc_info.value)
            assert "Invalid API key" in str(exc_info.value)

    def test_send_notification_exception(self, mock_resend_config, message):
        """Test handling of exceptions during email sending."""
        mock_resend_config()

        with patch("resend.Emails.send") as mock_send:
            mock_send.side_effect = Exception("Network error")

            service = Resend()

            with pytest.raises(Exception) as exc_info:
                service.send_notification(message)

            assert "Failed to send email notification" in str(exc_info.value)
            assert "Network error" in str(exc_info.value)

    def test_send_notification_rate_limit_error(self, mock_resend_config, message):
        """Test handling of rate limit errors."""
        mock_resend_config()

        with patch("resend.Emails.send") as mock_send:
            mock_send.return_value = {"error": "Rate limit exceeded"}

            service = Resend()

            with pytest.raises(Exception) as exc_info:
                service.send_notification(message)

            assert "Resend API error" in str(exc_info.value)
            assert "Rate limit exceeded" in str(exc_info.value)
