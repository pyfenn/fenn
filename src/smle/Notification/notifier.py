
from src.smle.Notification.service import Service


class Notifier:
    def __init__(self):
        self.services = []

    def add_service(self, service : Service) -> None:
        self.services.append(service)

    def remove_service(self,service : Service) -> None:
        self.services.remove(service)

    def notify(self, message:str) -> None:
        try:
            for service in self.services:
                service.send_notification(message)
        except Exception as e:
            print(f"Failed to notify service: {e}")