import pytest

from fenn.notification import Notifier, Service


class Recorder(Service):
    def __init__(self):
        self.messages = []

    def send_notification(self, message: str) -> None:
        self.messages.append(message)


class Broken(Service):
    def send_notification(self, message: str) -> None:
        raise RuntimeError("down")


def test_add_instance():
    n, s = Notifier(), Recorder()
    n.add_service(s)
    assert n._services == [s]


def test_add_class_without_args():
    n = Notifier()
    n.add_service(Recorder)
    assert n.get_services() == ["Recorder"]
    assert isinstance(n._services[0], Recorder)


def test_add_services_mixed():
    n, s = Notifier(), Recorder()
    n.add_services([s, Recorder])
    assert len(n._services) == 2 and n._services[0] is s


def test_remove_instance_only_that_one():
    n, a, b = Notifier(), Recorder(), Recorder()
    n.add_services([a, b])
    n.remove_service(b)
    assert n._services == [a]


def test_remove_by_class_removes_first_instance():
    n, a, b = Notifier(), Recorder(), Recorder()
    n.add_services([a, b])
    n.remove_service(Recorder)
    assert n._services == [b]


def test_remove_missing_raises():
    n = Notifier()
    with pytest.raises(ValueError):
        n.remove_service(Recorder)
    with pytest.raises(ValueError):
        n.remove_service(Recorder())
    n.add_service(Recorder())
    with pytest.raises(ValueError):
        n.remove_service(Recorder())  # different instance


def test_notify_calls_all():
    n, a, b = Notifier(), Recorder(), Recorder()
    n.add_services([a, b])
    n.notify("hi")
    assert a.messages == ["hi"] and b.messages == ["hi"]


def test_notify_survives_failing_service():
    n, ok = Notifier(), Recorder()
    n.add_services([Broken(), ok])
    n.notify("hi")
    assert ok.messages == ["hi"]


def test_notify_without_services():
    Notifier().notify("hi")


def test_get_and_clear_services():
    n = Notifier()
    n.add_services([Recorder(), Broken()])
    assert n.get_services() == ["Recorder", "Broken"]
    n.clear_services()
    assert n.get_services() == []
