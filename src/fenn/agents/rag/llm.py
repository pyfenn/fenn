import os
import time


# ── LLM Providers ─────────────────────────────────────────────────────────────
#
# All providers use an OpenAI-compatible chat completions API.
# Local providers (ollama, lmstudio, llamacpp) require no API key.
#
PROVIDERS = {
    # ── Aggregators (many models, one key) ────────────────
    "openrouter":  "https://openrouter.ai/api/v1",
    "together":    "https://api.together.xyz/v1",
    "groq":        "https://api.groq.com/openai/v1",
    "fireworks":   "https://api.fireworks.ai/inference/v1",
    "deepinfra":   "https://api.deepinfra.com/v1/openai",
    "anyscale":    "https://api.endpoints.anyscale.com/v1",
    "perplexity":  "https://api.perplexity.ai",
    # ── Direct providers ──────────────────────────────────
    "openai":      "https://api.openai.com/v1",
    "anthropic":   "https://api.anthropic.com/v1",
    "gemini":      "https://generativelanguage.googleapis.com/v1beta/openai",
    "mistral":     "https://api.mistral.ai/v1",
    "cohere":      "https://api.cohere.ai/compatibility/v1",
    "xai":         "https://api.x.ai/v1",
    "deepseek":    "https://api.deepseek.com/v1",
    "cerebras":    "https://api.cerebras.ai/v1",
    "nvidia":      "https://integrate.api.nvidia.com/v1",
    # ── Local (no key, no internet) ───────────────────────
    "ollama":      "http://localhost:11434/v1",
    "lmstudio":    "http://localhost:1234/v1",
    "llamacpp":    "http://localhost:8080/v1",
}

# Default model for each provider when model= is not specified
DEFAULT_MODELS = {
    "openrouter":  "arcee-ai/trinity-large-preview:free",   # free tier
    "together":    "meta-llama/Llama-3-8b-chat-hf",
    "groq":        "llama-3.1-8b-instant",
    "fireworks":   "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "deepinfra":   "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "perplexity":  "llama-3.1-sonar-small-128k-online",
    "openai":      "gpt-4o-mini",
    "anthropic":   "claude-3-5-haiku-20241022",
    "gemini":      "gemini-2.0-flash",
    "mistral":     "mistral-small-latest",
    "cohere":      "command-r-plus",
    "xai":         "grok-beta",
    "deepseek":    "deepseek-chat",
    "cerebras":    "llama3.1-8b",
    "nvidia":      "meta/llama-3.1-8b-instruct",
    "ollama":      "llama3",
    "lmstudio":    "local-model",
    "llamacpp":    "local-model",
}

# Environment variable name for each provider's API key
ENV_KEYS = {
    "openrouter":  "OPENROUTER_API_KEY",
    "together":    "TOGETHER_API_KEY",
    "groq":        "GROQ_API_KEY",
    "fireworks":   "FIREWORKS_API_KEY",
    "deepinfra":   "DEEPINFRA_API_KEY",
    "anyscale":    "ANYSCALE_API_KEY",
    "perplexity":  "PERPLEXITY_API_KEY",
    "openai":      "OPENAI_API_KEY",
    "anthropic":   "ANTHROPIC_API_KEY",
    "gemini":      "GEMINI_API_KEY",
    "mistral":     "MISTRAL_API_KEY",
    "cohere":      "COHERE_API_KEY",
    "xai":         "XAI_API_KEY",
    "deepseek":    "DEEPSEEK_API_KEY",
    "cerebras":    "CEREBRAS_API_KEY",
    "nvidia":      "NVIDIA_API_KEY",
    "ollama":      None,   # local, no key
    "lmstudio":    None,   # local, no key
    "llamacpp":    None,   # local, no key
}

# Providers that run locally and never need an API key
LOCAL_PROVIDERS = {"ollama", "lmstudio", "llamacpp"}


def ask(prompt, model=None, model_api_key=None, base_url=None,
        model_provider=None, retries=3, schema=None):
    """
    Send a prompt to an LLM and return the response.

    Parameters
    ----------
    prompt : str
        The full prompt to send.
    model : str, optional
        Model identifier. Defaults to the provider's default model.
    model_api_key : str, optional
        API key. If None, reads from the environment variable.
    base_url : str, optional
        Custom API base URL (for custom or proxy endpoints).
    model_provider : str, optional
        Provider name. Auto-detected from model name if not given.
    retries : int, optional
        Number of retry attempts on rate limit errors. Default: 3.
    schema : pydantic.BaseModel, optional
        If provided, instructs the LLM to return JSON matching this schema
        and returns a validated Pydantic model instance.

    Returns
    -------
    str or pydantic.BaseModel
    """
    try:
        from openai import OpenAI, RateLimitError

        model_provider = _detect_provider(model_provider, model, base_url)
        model          = model    or DEFAULT_MODELS.get(model_provider, "gpt-4o-mini")
        base_url       = base_url or PROVIDERS.get(model_provider, PROVIDERS["openrouter"])
        model_api_key  = _resolve_key(model_api_key, model_provider)
        client         = OpenAI(api_key=model_api_key or "local", base_url=base_url)

        if schema:
            prompt += (
                f"\n\nRespond ONLY with a valid JSON object matching this schema:\n"
                f"{schema.model_json_schema()}\n"
                f"Do not include any text outside the JSON object."
            )

        kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}])
        if schema:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(retries):
            try:
                response = client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content
                if schema:
                    import json
                    return schema.model_validate(json.loads(text))
                return text
            except RateLimitError:
                if attempt < retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"[cofone] rate limit hit, retrying in {wait}s... ({attempt+1}/{retries})")
                    time.sleep(wait)
                else:
                    raise

    except ImportError:
        raise ImportError(
            "[cofone] 'openai' package not found.\n"
            "Run: pip install openai"
        )


def stream(prompt, model=None, model_api_key=None, base_url=None, model_provider=None):
    """
    Send a prompt to an LLM and yield response tokens one by one.

    Parameters
    ----------
    prompt : str
        The full prompt to send.
    model : str, optional
        Model identifier.
    model_api_key : str, optional
        API key.
    base_url : str, optional
        Custom API base URL.
    model_provider : str, optional
        Provider name.

    Yields
    ------
    str
        Individual tokens from the LLM response.
    """
    try:
        from openai import OpenAI

        model_provider = _detect_provider(model_provider, model, base_url)
        model          = model    or DEFAULT_MODELS.get(model_provider, "gpt-4o-mini")
        base_url       = base_url or PROVIDERS.get(model_provider, PROVIDERS["openrouter"])
        model_api_key  = _resolve_key(model_api_key, model_provider)
        client         = OpenAI(api_key=model_api_key or "local", base_url=base_url)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in response:
            if not hasattr(chunk, "choices") or len(chunk.choices) == 0:
                continue
            delta = getattr(chunk.choices[0].delta, "content", "") or ""
            if delta:
                yield delta

    except ImportError:
        raise ImportError(
            "[cofone] 'openai' package not found.\n"
            "Run: pip install openai"
        )


def _detect_provider(model_provider, model, base_url):
    """
    Infer the provider from model name or base_url if not explicitly given.

    Priority: explicit model_provider > base_url match > model name prefix.
    Falls back to "openrouter" if nothing matches.
    """
    if model_provider:
        return model_provider
    if base_url:
        for name, url in PROVIDERS.items():
            if url in base_url:
                return name
        return "openrouter"
    if model:
        if model.startswith(("gpt-", "o1-", "o3-", "o4-")):    return "openai"
        if model.startswith("gemini"):                          return "gemini"
        if model.startswith("claude"):                          return "anthropic"
        if model.startswith(("mistral", "codestral")):          return "mistral"
        if model.startswith("command"):                         return "cohere"
        if model.startswith("grok"):                            return "xai"
        if model.startswith("deepseek"):                        return "deepseek"
        if model.startswith(("llama", "mixtral")):              return "groq"
        if "/" in model:                                        return "openrouter"
    return "openrouter"


def _resolve_key(model_api_key, model_provider):
    """
    Resolve the API key from direct parameter or environment variable.

    Priority: direct parameter > environment variable.
    Local providers never need a key.

    Raises ValueError if the key is required but not found.
    """
    if model_api_key:
        return model_api_key
    if model_provider in LOCAL_PROVIDERS:
        return "local"
    env = ENV_KEYS.get(model_provider)
    if env:
        key = os.environ.get(env)
        if not key:
            raise ValueError(
                f"[cofone] API key not found for provider '{model_provider}'.\n"
                f"Set {env} in your .env file and call load_dotenv(),\n"
                f"or pass model_api_key='...' directly to RAG()."
            )
        return key
    return "local"