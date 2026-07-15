"""Tests for fenn/agents/rag/llm.py (module-level ask/stream functions)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fenn.agents.rag.llm import (
    DEFAULT_MODELS,
    LOCAL_PROVIDERS,
    PROVIDERS,
    _detect_provider,
    _resolve_key,
    ask,
    stream,
)

# ── _detect_provider ───────────────────────────────────────────────────────────


class TestDetectProvider:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-4o-mini", "openai"),
            ("gpt-3.5-turbo", "openai"),
            ("o1-mini", "openai"),
            ("o3-large", "openai"),
            ("o4-preview", "openai"),
            ("gemini-2.0-flash", "gemini"),
            ("claude-3-5-haiku", "anthropic"),
            ("mistral-small", "mistral"),
            ("codestral-latest", "mistral"),
            ("command-r-plus", "cohere"),
            ("grok-beta", "xai"),
            ("deepseek-chat", "deepseek"),
            ("llama-3.1-8b", "groq"),
            ("mixtral-8x7b", "groq"),
            ("openai/gpt-4o", "openrouter"),
            (None, "openrouter"),
        ],
    )
    def test_from_model_name(self, model, expected):
        assert _detect_provider(None, model, None) == expected

    def test_explicit_provider_wins_over_model(self):
        assert _detect_provider("anthropic", "gpt-4o", None) == "anthropic"

    def test_from_known_base_url(self):
        assert _detect_provider(None, None, PROVIDERS["groq"]) == "groq"

    def test_from_base_url_partial_match(self):
        # URL contains the provider URL as a substring
        assert _detect_provider(None, None, PROVIDERS["openai"] + "/extra") == "openai"

    def test_unknown_base_url_defaults_to_openrouter(self):
        assert (
            _detect_provider(None, None, "https://custom.endpoint/v1") == "openrouter"
        )

    def test_base_url_takes_priority_over_model(self):
        assert _detect_provider(None, "gpt-4o", PROVIDERS["groq"]) == "groq"

    def test_all_none_defaults_to_openrouter(self):
        assert _detect_provider(None, None, None) == "openrouter"


# ── _resolve_key ───────────────────────────────────────────────────────────────


class TestResolveKey:
    def test_explicit_key_wins(self):
        assert _resolve_key("my-key", "openai") == "my-key"

    @pytest.mark.parametrize("provider", sorted(LOCAL_PROVIDERS))
    def test_local_providers_return_local(self, provider):
        assert _resolve_key(None, provider) == "local"

    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        assert _resolve_key(None, "openai") == "env-key"

    def test_missing_env_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key not found"):
            _resolve_key(None, "openai")

    def test_unknown_provider_returns_local(self):
        assert _resolve_key(None, "unknown") == "local"

    def test_explicit_key_beats_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        assert _resolve_key("direct-key", "openai") == "direct-key"


# ── Helpers for mocking openai responses ──────────────────────────────────────


def _make_completion(content="response text"):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _make_chunk(content):
    delta = SimpleNamespace(content=content)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


def _mock_openai(content="response text"):
    """Return a mock OpenAI client whose chat.completions.create returns a completion."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_completion(content)
    return mock_client


# ── ask ────────────────────────────────────────────────────────────────────────


class TestAsk:
    def _patched_ask(self, prompt, mock_client, **kwargs):
        RLE = type("RateLimitError", (Exception,), {})
        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = mock_client
        fake_openai.RateLimitError = RLE
        with patch.dict("sys.modules", {"openai": fake_openai}):
            return ask(
                prompt, model_api_key="test-key", model_provider="openai", **kwargs
            )

    def test_returns_text(self):
        result = self._patched_ask("hello", _mock_openai("hi there"))
        assert result == "hi there"

    def test_uses_correct_model(self):
        mock_client = _mock_openai()
        self._patched_ask("hello", mock_client, model="gpt-4o")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    def test_defaults_to_provider_default_model(self):
        mock_client = _mock_openai()
        self._patched_ask("hello", mock_client)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == DEFAULT_MODELS["openai"]

    def test_sends_user_message(self):
        mock_client = _mock_openai()
        self._patched_ask("my prompt", mock_client)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert "my prompt" in messages[0]["content"]

    def test_schema_appends_json_instruction(self):
        from pydantic import BaseModel

        class Reply(BaseModel):
            answer: str

        payload = json.dumps({"answer": "42"})
        mock_client = _mock_openai(payload)
        result = self._patched_ask("question", mock_client, schema=Reply)
        assert isinstance(result, Reply)
        assert result.answer == "42"

    def test_schema_sets_response_format(self):
        from pydantic import BaseModel

        class Reply(BaseModel):
            value: int

        mock_client = _mock_openai(json.dumps({"value": 1}))
        self._patched_ask("q", mock_client, schema=Reply)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    def test_retries_on_rate_limit_then_raises(self):
        import fenn.agents.rag.llm as llm_module

        RLE = type("RateLimitError", (Exception,), {})
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RLE("limited")

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = mock_client
        fake_openai.RateLimitError = RLE

        with patch.dict("sys.modules", {"openai": fake_openai}):
            with patch.object(llm_module, "time") as mock_time:
                with pytest.raises(RLE):
                    ask("hi", model_api_key="key", model_provider="openai", retries=3)
            assert mock_time.sleep.call_count == 2

    def test_retries_succeeds_after_rate_limit(self):
        import fenn.agents.rag.llm as llm_module

        RLE = type("RateLimitError", (Exception,), {})
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            RLE("limited"),
            _make_completion("ok"),
        ]

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = mock_client
        fake_openai.RateLimitError = RLE

        with patch.dict("sys.modules", {"openai": fake_openai}):
            with patch.object(llm_module, "time"):
                result = ask(
                    "hi", model_api_key="key", model_provider="openai", retries=3
                )
        assert result == "ok"

    def test_raises_import_error_without_openai(self):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                ask("hello", model_api_key="key", model_provider="openai")

    def test_custom_base_url_is_used(self):
        mock_client = _mock_openai()
        RLE = type("RateLimitError", (Exception,), {})
        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = mock_client
        fake_openai.RateLimitError = RLE

        custom_url = "https://my.proxy/v1"
        with patch.dict("sys.modules", {"openai": fake_openai}):
            ask("hi", model_api_key="key", model_provider="openai", base_url=custom_url)

        fake_openai.OpenAI.assert_called_once_with(api_key="key", base_url=custom_url)


# ── stream ─────────────────────────────────────────────────────────────────────


class TestStream:
    def _patched_stream(self, prompt, chunks, **kwargs):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = mock_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            return list(
                stream(
                    prompt, model_api_key="test-key", model_provider="openai", **kwargs
                )
            ), mock_client

    def test_yields_tokens(self):
        chunks = [_make_chunk("Hello"), _make_chunk(", "), _make_chunk("world")]
        tokens, _ = self._patched_stream("hi", chunks)
        assert tokens == ["Hello", ", ", "world"]

    def test_skips_empty_delta(self):
        chunks = [
            _make_chunk("Hi"),
            _make_chunk(""),
            _make_chunk(None),
            _make_chunk("!"),
        ]
        tokens, _ = self._patched_stream("hi", chunks)
        assert tokens == ["Hi", "!"]

    def test_skips_chunk_without_choices(self):
        no_choices = SimpleNamespace(choices=[])
        normal = _make_chunk("ok")
        tokens, _ = self._patched_stream("hi", [no_choices, normal])
        assert tokens == ["ok"]

    def test_passes_stream_true(self):
        _, mock_client = self._patched_stream("hi", [])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True

    def test_sends_correct_prompt(self):
        _, mock_client = self._patched_stream("my question", [])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["content"] == "my question"

    def test_raises_import_error_without_openai(self):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                list(stream("hi", model_api_key="key", model_provider="openai"))
