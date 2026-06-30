# tests/test_utils.py
import re

from fenn.reproducibility import generate_session_id


def test_session_id_returns_string():
    result = generate_session_id()
    assert isinstance(result, str)


def test_session_id_format():
    result = generate_session_id()
    # Should match: 20260412_1233_a3f9
    assert re.match(r"\d{8}_\d{4}_[a-f0-9]{4}", result), f"Unexpected format: {result}"


def test_session_id_is_unique():
    id1 = generate_session_id()
    id2 = generate_session_id()
    assert id1 != id2
