"""
Additional tests for LLMClient._openai_client, .chat_complete, .ask, and .stream.
Merge with or run alongside test_llm.py.
"""

import builtins
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from fenn.agents.llm import LLMClient

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_client(provider="openai", model="gpt-4o-mini", api_key="test-key"):
    """Build an LLMClient bypassing env-var lookup."""
    client = LLMClient.__new__(LLMClient)
    client.provider = provider
    client.model = model
    client.api_key = api_key
    client.base_url = "https://api.openai.com/v1"
    return client


def _make_completion(content="Hello!"):
    """Return a minimal fake openai ChatCompletion object."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _patched_openai_module():
    """
    Build a fake 'openai' module so `from openai import RateLimitError`
    inside chat_complete() succeeds without the real package installed.
    """
    fake_module = MagicMock()
    fake_module.RateLimitError = type("RateLimitError", (Exception,), {})
    return fake_module


# ── _openai_client ─────────────────────────────────────────────────────────────


def test_openai_client_returns_openai_instance(monkeypatch):
    fake_openai_cls = MagicMock()
    fake_instance = MagicMock()
    fake_openai_cls.return_value = fake_instance

    with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=fake_openai_cls)}):
        client = _make_client()
        result = client._openai_client()

    fake_openai_cls.assert_called_once_with(
        api_key="test-key", base_url=client.base_url
    )
    assert result is fake_instance


def test_openai_client_raises_on_missing_package():
    client = _make_client()
    with patch.dict("sys.modules", {"openai": None}):
        with pytest.raises(ImportError, match="openai"):
            client._openai_client()


# ── chat_complete ──────────────────────────────────────────────────────────────


def test_chat_complete_plain_text(monkeypatch):
    """chat_complete without schema returns a string."""
    client = _make_client()
    fake_response = _make_completion("plain response")
    mock_oa = MagicMock()
    mock_oa.chat.completions.create.return_value = fake_response

    with patch.dict("sys.modules", {"openai": _patched_openai_module()}):
        with patch.object(client, "_openai_client", return_value=mock_oa):
            result = client.chat_complete([{"role": "user", "content": "hi"}])

    assert result == "plain response"


def test_chat_complete_with_schema(monkeypatch):
    """When schema is provided, response JSON is validated and returned as model."""

    class Reply(BaseModel):
        answer: str

    payload = json.dumps({"answer": "42"})
    fake_response = _make_completion(payload)
    mock_oa = MagicMock()
    mock_oa.chat.completions.create.return_value = fake_response

    client = _make_client()
    with patch.dict("sys.modules", {"openai": _patched_openai_module()}):
        with patch.object(client, "_openai_client", return_value=mock_oa):
            result = client.chat_complete(
                [{"role": "user", "content": "What is the answer?"}],
                schema=Reply,
            )

    assert isinstance(result, Reply)
    assert result.answer == "42"
    # Confirm response_format was injected
    _, call_kwargs = mock_oa.chat.completions.create.call_args
    assert call_kwargs.get("response_format") == {"type": "json_object"}


def test_chat_complete_with_schema_simple():
    """Schema path: JSON body is parsed and returned as a Pydantic model."""

    class Point(BaseModel):
        x: int
        y: int

    client = _make_client()
    fake_response = _make_completion(json.dumps({"x": 1, "y": 2}))
    mock_oa = MagicMock()
    mock_oa.chat.completions.create.return_value = fake_response

    with patch.dict("sys.modules", {"openai": _patched_openai_module()}):
        with patch.object(client, "_openai_client", return_value=mock_oa):
            result = client.chat_complete(
                [{"role": "user", "content": "give me a point"}],
                schema=Point,
            )

    assert result == Point(x=1, y=2)


def test_chat_complete_does_not_mutate_caller_messages():
    """chat_complete should shallow-copy messages, not mutate the caller's list."""
    client = _make_client()
    msgs = [{"role": "user", "content": "hello"}]
    original_content = msgs[0]["content"]

    fake_response = _make_completion("ok")
    mock_oa = MagicMock()
    mock_oa.chat.completions.create.return_value = fake_response

    with patch.dict("sys.modules", {"openai": _patched_openai_module()}):
        with patch.object(client, "_openai_client", return_value=mock_oa):
            client.chat_complete(msgs)

    assert msgs[0]["content"] == original_content


def test_chat_complete_retries_on_rate_limit():
    """Should retry up to `retries` times on RateLimitError, then raise."""
    import fenn.agents.llm as llm_module

    client = _make_client()
    RLE = type("RateLimitError", (Exception,), {})

    mock_oa = MagicMock()
    mock_oa.chat.completions.create.side_effect = RLE("rate limited")
    original_import = builtins.__import__
    with patch.object(client, "_openai_client", return_value=mock_oa):
        with patch.object(llm_module, "time") as mock_time:
            # patch the RateLimitError import inside chat_complete
            with patch("builtins.__import__") as mock_import:

                def side_import(name, *args, **kwargs):
                    if name == "openai":
                        return SimpleNamespace(RateLimitError=RLE)
                    return original_import(name, *args, **kwargs)

                mock_import.side_effect = side_import

                with pytest.raises(RLE):
                    client.chat_complete([{"role": "user", "content": "hi"}], retries=3)

            assert (
                mock_time.sleep.call_count == 2
            )  # retries-1 sleeps before final raise


def test_chat_complete_rate_limit_retry_succeeds():
    """Should succeed if a retry after RateLimitError returns a valid response."""
    import fenn.agents.llm as llm_module

    client = _make_client()
    RLE = type("RateLimitError", (Exception,), {})
    fake_response = _make_completion("eventually ok")

    mock_oa = MagicMock()
    mock_oa.chat.completions.create.side_effect = [RLE("limited"), fake_response]
    original_import = builtins.__import__

    with patch.object(client, "_openai_client", return_value=mock_oa):
        with patch.object(llm_module, "time"):
            with patch("builtins.__import__") as mock_import:

                def side_import(name, *args, **kwargs):
                    if name == "openai":
                        return SimpleNamespace(RateLimitError=RLE)
                    return original_import(name, *args, **kwargs)

                mock_import.side_effect = side_import

                result = client.chat_complete(
                    [{"role": "user", "content": "hi"}], retries=3
                )

    assert result == "eventually ok"


def test_chat_complete_raises_import_error_without_openai():
    """When openai is not installed, chat_complete raises ImportError."""
    client = _make_client()
    with patch.dict("sys.modules", {"openai": None}):
        with pytest.raises(ImportError, match="openai"):
            client.chat_complete([{"role": "user", "content": "hi"}])


# ── ask ────────────────────────────────────────────────────────────────────────


def test_ask_delegates_to_chat_complete():
    client = _make_client()
    with patch.object(client, "chat_complete", return_value="answer") as mock_cc:
        result = client.ask("What is 2+2?")

    assert result == "answer"
    mock_cc.assert_called_once_with(
        [{"role": "user", "content": "What is 2+2?"}],
        schema=None,
        retries=3,
    )


def test_ask_passes_schema_and_retries():
    class Ans(BaseModel):
        value: int

    client = _make_client()
    with patch.object(client, "chat_complete", return_value=Ans(value=4)) as mock_cc:
        result = client.ask("2+2?", schema=Ans, retries=5)

    assert result == Ans(value=4)
    mock_cc.assert_called_once_with(
        [{"role": "user", "content": "2+2?"}], schema=Ans, retries=5
    )


# ── stream ─────────────────────────────────────────────────────────────────────


def _make_chunk(content):
    delta = SimpleNamespace(content=content)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


def test_stream_yields_tokens():
    client = _make_client()
    chunks = [_make_chunk("Hello"), _make_chunk(", "), _make_chunk("world")]
    mock_oa = MagicMock()
    mock_oa.chat.completions.create.return_value = iter(chunks)

    with patch.object(client, "_openai_client", return_value=mock_oa):
        tokens = list(client.stream("Say hello"))

    assert tokens == ["Hello", ", ", "world"]


def test_stream_skips_empty_delta():
    client = _make_client()
    chunks = [_make_chunk("Hi"), _make_chunk(""), _make_chunk(None), _make_chunk("!")]
    mock_oa = MagicMock()
    mock_oa.chat.completions.create.return_value = iter(chunks)

    with patch.object(client, "_openai_client", return_value=mock_oa):
        tokens = list(client.stream("hi"))

    assert tokens == ["Hi", "!"]


def test_stream_skips_chunk_without_choices():
    client = _make_client()
    no_choices = SimpleNamespace(choices=[])
    normal = _make_chunk("ok")
    mock_oa = MagicMock()
    mock_oa.chat.completions.create.return_value = iter([no_choices, normal])

    with patch.object(client, "_openai_client", return_value=mock_oa):
        tokens = list(client.stream("hi"))

    assert tokens == ["ok"]


def test_stream_passes_correct_params():
    client = _make_client()
    mock_oa = MagicMock()
    mock_oa.chat.completions.create.return_value = iter([])

    with patch.object(client, "_openai_client", return_value=mock_oa):
        list(client.stream("test prompt"))

    call_kwargs = mock_oa.chat.completions.create.call_args[1]
    assert call_kwargs["stream"] is True
    assert call_kwargs["model"] == client.model
    assert call_kwargs["messages"] == [{"role": "user", "content": "test prompt"}]
