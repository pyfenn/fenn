from unittest.mock import Mock, patch

import pytest
import requests

from fenn.notification.services.telegram import Telegram


def test_telegram_message():
    """Test Telegram.send_notification"""
    # Bypass __init__ to avoid KeyStore singleton
    telegram = object.__new__(Telegram)
    telegram._telegram_api_url = "https://api.telegram.org/bot123/sendMessage"
    telegram._chat_id = Mock()  # chat_id is read from .env in constructor
    telegram._parse_mode = None

    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        telegram.send_notification("hello telegram test")

        mock_post.assert_called_once_with(
            "https://api.telegram.org/bot123/sendMessage",
            json={
                "chat_id": telegram._chat_id,
                "text": "hello telegram test",
                "disable_notification": False,
            },
            timeout=10,
        )


def test_telegram_message_with_parse_mode():
    """Test Telegram.send_notification with parse_mode"""
    telegram = object.__new__(Telegram)
    telegram._telegram_api_url = "https://api.telegram.org/bot123/sendMessage"
    telegram._chat_id = Mock()  # chat_id is read from .env in constructor
    telegram._parse_mode = "Markdown"

    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        telegram.send_notification("*hello* telegram test")

        mock_post.assert_called_once_with(
            "https://api.telegram.org/bot123/sendMessage",
            json={
                "chat_id": telegram._chat_id,
                "text": "*hello* telegram test",
                "disable_notification": False,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )


def test_telegram_send_notification_error():
    """Test Telegram.send_notification error handling"""
    telegram = object.__new__(Telegram)
    telegram._telegram_api_url = "https://api.telegram.org/bot123/sendMessage"
    telegram._chat_id = Mock()  # chat_id is read from .env in constructor
    telegram._parse_mode = None

    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")

        with pytest.raises(requests.exceptions.RequestException) as exc_info:
            telegram.send_notification("hello telegram test")

        assert "Failed to send Telegram notification" in str(exc_info.value)
