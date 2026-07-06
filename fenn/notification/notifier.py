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
        services: Iterable[type[Service]],
    ) -> None:
        """
        Add a list of notification services.

        Example:
            app.register_notification_services([Discord, Telegram])
        """

        for service_cls in services:
            self.add_service(service_cls)

    def add_service(self, service: type[Service]) -> None:
        """Add a notification service.

        Args:
            service: A service implementing the Service interface.
        """
        self._services.append(service())

    def remove_service(self, service: type[Service]) -> None:
        """Remove a notification service.

        Args:
            service: The service to remove.

        Raises:
            ValueError: If the service is not found.
        """
        try:
            self._services.remove(service())
        except ValueError as err:
            raise ValueError(
                f"Service {service.__name__} not found in services list"
            ) from err

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
