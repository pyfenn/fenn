from abc import ABC, abstractmethod


class Service(ABC):
    """Abstract base class for notification services."""

    @abstractmethod
    def send_notification(self, message: str) -> None:
        """Send a notification message.

        Args:
            message: The message to send.

        Raises:
            Exception: If the notification fails to send.
        """
        pass

    def should_notify(self, is_success: bool) -> bool:
        """Determine if a notification should be sent based on status.

        This method can be overridden by subclasses to filter notifications
        based on success/failure status.

        Args:
            is_success: True if the operation succeeded, False if it failed.

        Returns:
            True if the notification should be sent, False otherwise.
        """
        return True
