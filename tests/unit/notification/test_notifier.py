import pytest

from fenn.notification.notifier import Notifier
from fenn.notification.service import Service


class RecordingService(Service):
    def __init__(self, name: str = "recorder") -> None:
        self.name = name
        self.sent: list[str] = []

    def send_notification(self, message: str) -> None:
        self.sent.append(message)


class BoomService(Service):
    def send_notification(self, message: str) -> None:
        raise RuntimeError("boom")


class TestNotifier:
    def test_add_service_and_notify(self):
        notifier = Notifier()
        service = RecordingService()

        notifier.add_service(service)
        notifier.notify("hello")

        assert service.sent == ["hello"]
        assert notifier.get_services() == ["RecordingService"]

    def test_add_services_accepts_instances(self):
        notifier = Notifier()
        a, b = RecordingService("a"), RecordingService("b")

        notifier.add_services([a, b])
        notifier.notify("ping")

        assert a.sent == ["ping"]
        assert b.sent == ["ping"]

    def test_add_service_rejects_a_class(self):
        with pytest.raises(TypeError, match="configured instance"):
            Notifier().add_service(RecordingService)

    def test_notify_with_no_services_is_a_noop(self):
        Notifier().notify("nobody listening")

    def test_notify_isolates_a_failing_service(self):
        notifier = Notifier()
        ok = RecordingService()

        notifier.add_services([BoomService(), ok])
        notifier.notify("still delivered")

        assert ok.sent == ["still delivered"]

    def test_remove_service(self):
        notifier = Notifier()
        service = RecordingService()

        notifier.add_service(service)
        notifier.remove_service(service)
        notifier.notify("gone")

        assert service.sent == []
        assert notifier.get_services() == []

    def test_remove_service_not_present_raises(self):
        with pytest.raises(ValueError, match="not found"):
            Notifier().remove_service(RecordingService())

    def test_clear_services(self):
        notifier = Notifier()
        notifier.add_services([RecordingService(), RecordingService()])

        notifier.clear_services()

        assert notifier.get_services() == []
