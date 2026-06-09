import requests

from fenn.notification.service import Service


class Discord(Service):
    """Discord notification service using webhooks."""

    def __init__(self, webhook_url: str, notify_on_success: bool = False):
        """Initialize Discord service.

        Args:
            webhook_url: Discord webhook URL.
            notify_on_success: If False (default), only send notifications on failure.
                              If True, send notifications for both success and failure.
        """
        super().__init__()

        self._discord_webhook_url = webhook_url
        self._notify_on_success = notify_on_success

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

    def should_notify(self, is_success: bool) -> bool:
        """Determine if a notification should be sent based on status.

        Args:
            is_success: True if the operation succeeded, False if it failed.

        Returns:
            True if the notification should be sent, False otherwise.
        """
        if is_success:
            return self._notify_on_success
        return True
