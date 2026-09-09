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
        services: Iterable[Service],
    ) -> None:
        """
        Add a list of notification services.

        Example:
            notifier.add_services([Discord(webhook_url), Telegram(token, chat_id)])
        """

        for service in services:
            self.add_service(service)

    def add_service(self, service: Service) -> None:
        """Add a notification service.

        Args:
            service: A configured service instance, e.g. ``Discord(webhook_url)``.
        """
        if isinstance(service, type):
            raise TypeError(
                "add_service expects a configured instance "
                f"(e.g. {service.__name__}(...)), not the class itself."
            )

        self._services.append(service)

    def remove_service(self, service: Service) -> None:
        """Remove a previously added notification service.

        Args:
            service: The same service instance that was passed to ``add_service``.

        Raises:
            ValueError: If the service is not found.
        """
        try:
            self._services.remove(service)
        except ValueError as err:
            name = getattr(type(service), "__name__", repr(service))

            raise ValueError(f"Service {name} not found in services list") from err

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
