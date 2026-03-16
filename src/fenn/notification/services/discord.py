import requests

from fenn.notification.service import Service


class Discord(Service):
    """Discord notification service using webhooks."""

    def __init__(self):
        """Initialize Discord service."""
        super().__init__()

        self._discord_webhook_url = self._keystore.get_key("DISCORD_WEBHOOK_URL")

    def send_notification(self, message: str) -> None:
        """Send notification to Discord channel.

        Args:
            message: The message to send.

        Raises:
            requests.exceptions.RequestException: If the request fails.
        """
        data = {"content": message, "username": "fenn"}

        try:
            result = requests.post(self._discord_webhook_url, json=data, timeout=10)
            result.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise requests.exceptions.RequestException(
                f"Failed to send Discord notification: {err}"
            )
