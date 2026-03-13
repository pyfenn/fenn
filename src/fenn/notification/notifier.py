from typing import Iterable, List, Type

from fenn.notification.service import Service


class Notifier:
    """Central notification manager that handles multiple notification services."""

    def __init__(self):
        """Initialize the notifier with an empty list of services."""
        self._services: List[Service] = []

    def add_services(
        self,
        services: Iterable[Type[Service]],
    ) -> None:
        """
        Add a list of notification services.

        Example:
            app.register_notification_services([Discord, Telegram])
        """

        for service_cls in services:
            self.add_service(service_cls)

    def add_service(self, service: Type[Service]) -> None:
        """Add a notification service.

        Args:
            service: A service implementing the Service interface.
        """
        self._services.append(service())

    def remove_service(self, service: Type[Service]) -> None:
        """Remove a notification service.

        Args:
            service: The service to remove.

        Raises:
            ValueError: If the service is not found.
        """
        try:
            self._services.remove(service())
        except ValueError:
            ValueError(
                f"Service {service.__class__.__name__} not found in services list"
            )
            raise

    def notify(self, message: str) -> None:
        """Send notification to all registered services.

        Args:
            message: The message to send.
        """
        if not self._services:
            return

        successful_services = []
        failed_services = []

        for service in self._services:
            try:
                service.send_notification(message)
                successful_services.append(service.__class__.__name__)
                # logger.info(f"Successfully sent notification via {service.__class__.__name__}")
            except Exception as e:
                failed_services.append((service.__class__.__name__, str(e)))
                # logger.error(f"Failed to send notification via {service.__class__.__name__}: {e}")

    def get_services(self) -> List[str]:
        """Get list of registered service names.

        Returns:
            List of service class names.
        """
        return [service.__class__.__name__ for service in self._services]

    def clear_services(self) -> None:
        """Remove all registered services."""
        self._services.clear()
