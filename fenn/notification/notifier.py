from collections.abc import Iterable

from fenn.logging import logger

from .service import Service


class Notifier:
    """Central notification manager that handles multiple notification services."""

    def __init__(self) -> None:
        """Initialize the notifier with an empty list of services."""
        self._services: list[Service] = []

    def add_services(
        self,
        services: Iterable[Service | type[Service]],
    ) -> None:
        """
        Add a list of notification services.

        Example:
            notifier.add_services([Discord(webhook_url=url), Telegram(...)])
        """

        for service in services:
            self.add_service(service)

    def add_service(self, service: Service | type[Service]) -> None:
        """Add a notification service.

        Args:
            service: A configured Service instance. A Service class is also
                accepted and instantiated without arguments (only works for
                services that need no constructor parameters).
        """
        if isinstance(service, type):
            service = service()
        self._services.append(service)

    def remove_service(self, service: Service | type[Service]) -> None:
        """Remove a notification service.

        Args:
            service: The instance to remove, or a Service class to remove
                the first registered instance of that class.

        Raises:
            ValueError: If the service is not found.
        """
        for i, registered in enumerate(self._services):
            if registered is service or (
                isinstance(service, type) and isinstance(registered, service)
            ):
                del self._services[i]
                return
        name = service.__name__ if isinstance(service, type) else type(service).__name__
        raise ValueError(f"Service {name} not found in services list")

    def notify(self, message: str) -> None:
        """Send notification to all registered services.

        Args:
            message: The message to send.
        """
        if not self._services:
            return

        for service in self._services:
            try:
                service.send_notification(message)
                logger.info(
                    "Successfully sent notification via %s", service.__class__.__name__
                )
            except Exception as e:
                logger.error(
                    "Failed to send notification via %s: %s",
                    service.__class__.__name__,
                    e,
                )

    def get_services(self) -> list[str]:
        """Get list of registered service names.

        Returns:
            List of service class names.
        """
        return [service.__class__.__name__ for service in self._services]

    def clear_services(self) -> None:
        """Remove all registered services."""
        self._services.clear()
