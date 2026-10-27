import pytest

from fenn import Fenn
from fenn.notification import Service
from fenn.parser import Parser


class Recorder(Service):
    def __init__(self):
        self.messages = []

    def send_notification(self, message: str) -> None:
        self.messages.append(message)


class Broken(Service):
    def send_notification(self, message: str) -> None:
        raise RuntimeError("down")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fenn.yaml").write_text(
        "session_id: test-session\nlogger:\n  dir: logs\nexport:\n  dir: exports\n",
        encoding="utf-8",
    )
    Parser.reset()
    a = Fenn()
    a.disable_disclaimer()
    yield a
    Parser.reset()


def test_success_sends_started_then_completed(app):
    svc = Recorder()
    app.add_notification_service(svc)

    @app.entrypoint
    def main(args):
        return 42

    assert app.run() == 42
    assert len(svc.messages) == 2
    assert svc.messages[0].endswith("started")
    assert svc.messages[1].endswith("completed")


def test_failure_notifies_type_not_message_and_reraises(app):
    svc = Recorder()
    app.add_notification_service(svc)

    @app.entrypoint
    def main(args):
        raise ValueError("secret")

    with pytest.raises(ValueError, match="secret"):
        app.run()
    assert len(svc.messages) == 2
    assert svc.messages[0].endswith("started")
    assert "failed" in svc.messages[1]
    assert "ValueError" in svc.messages[1]
    assert "secret" not in svc.messages[1]


def test_no_service_registered(app):
    @app.entrypoint
    def main(args):
        return "ok"

    assert app.run() == "ok"
    assert app._notifier.get_services() == []


def test_failing_service_does_not_break_run(app):
    app.add_notification_service(Broken())

    @app.entrypoint
    def main(args):
        return "ok"

    assert app.run() == "ok"
