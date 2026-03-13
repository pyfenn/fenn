from abc import ABC, abstractmethod

from fenn.secrets.keystore import KeyStore


class Service(ABC):
    """Abstract base class for notification services."""

    def __init__(self):
        self._keystore = KeyStore()

    @abstractmethod
    def send_notification(self, message: str) -> None:
        """Send a notification message.

        Args:
            message: The message to send.

        Raises:
            Exception: If the notification fails to send.
        """
        pass
