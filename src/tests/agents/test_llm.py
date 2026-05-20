import os
import pytest
from fenn.agents.llm import (
    LLMClient,
    _detect_provider,
    PROVIDERS,
    DEFAULT_MODELS,
    ENV_KEYS,
    LOCAL_PROVIDERS,
)


# ── Provider detection ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("model,expected", [
    ("gpt-4o-mini",        "openai"),
    ("gpt-3.5-turbo",      "openai"),
    ("o1-mini",            "openai"),
    ("gemini-2.0-flash",   "gemini"),
    ("claude-3-5-haiku",   "anthropic"),
    ("mistral-small",      "mistral"),
    ("codestral-latest",   "mistral"),
    ("command-r-plus",     "cohere"),
    ("grok-beta",          "xai"),
    ("deepseek-chat",      "deepseek"),
    ("llama-3.1-8b",       "groq"),
    ("mixtral-8x7b",       "groq"),
    ("openai/gpt-4o",      "openrouter"),
    (None,                 "openrouter"),
])
def test_detect_provider_from_model(model, expected):
    assert _detect_provider(None, model, None) == expected


def test_detect_provider_explicit_wins():
    assert _detect_provider("anthropic", "gpt-4o", None) == "anthropic"


def test_detect_provider_from_base_url():
    url = PROVIDERS["groq"]
    assert _detect_provider(None, None, url) == "groq"


def test_detect_provider_unknown_url_defaults_openrouter():
    assert _detect_provider(None, None, "https://custom.endpoint/v1") == "openrouter"


# ── Default model selection ────────────────────────────────────────────────────

def test_default_model_per_provider():
    for provider, model in DEFAULT_MODELS.items():
        client = LLMClient.__new__(LLMClient)
        client.provider = provider
        client.model    = None or DEFAULT_MODELS.get(provider, "gpt-4o-mini")
        assert client.model == model


# ── Local provider key handling ────────────────────────────────────────────────

@pytest.mark.parametrize("provider", sorted(LOCAL_PROVIDERS))
def test_local_providers_need_no_key(provider):
    client = LLMClient.__new__(LLMClient)
    client.provider = provider
    key = client._resolve_key(None, None)
    assert key == "local"


# ── Direct API key takes priority ─────────────────────────────────────────────

def test_direct_api_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    client = LLMClient.__new__(LLMClient)
    client.provider = "openai"
    assert client._resolve_key("direct-key", None) == "direct-key"


# ── Custom env var lookup ──────────────────────────────────────────────────────

def test_custom_api_key_env(monkeypatch):
    monkeypatch.setenv("MY_CUSTOM_KEY", "secret-123")
    client = LLMClient.__new__(LLMClient)
    client.provider = "openrouter"
    assert client._resolve_key(None, "MY_CUSTOM_KEY") == "secret-123"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient.__new__(LLMClient)
    client.provider = "openai"
    with pytest.raises(ValueError, match="API key not found"):
        client._resolve_key(None, None)


# ── LLMClient construction ────────────────────────────────────────────────────

def test_llmclient_resolves_provider_and_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = LLMClient(provider="openai")
    assert client.provider == "openai"
    assert client.model == DEFAULT_MODELS["openai"]
    assert client.base_url == PROVIDERS["openai"]


def test_llmclient_explicit_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = LLMClient(provider="openai", model="gpt-4o")
    assert client.model == "gpt-4o"
