from unittest.mock import Mock, patch

from fenn.notification.services.slack import Slack


def test_slack_message():
    """Test Slack.send_notification"""
    # Bypass __init__ to avoid KeyStore singleton
    slack = object.__new__(Slack)
    slack._slack_webhook_url = "https://slack.test"

    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        slack.send_notification("hello test")

        mock_post.assert_called_once_with(
            "https://slack.test", json={"text": "hello test"}, timeout=10
        )
