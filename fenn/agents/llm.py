import os
import time

from fenn.utils.logging import logger

PROVIDERS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "groq": "https://api.groq.com/openai/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "anyscale": "https://api.endpoints.anyscale.com/v1",
    "perplexity": "https://api.perplexity.ai",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "mistral": "https://api.mistral.ai/v1",
    "cohere": "https://api.cohere.ai/compatibility/v1",
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "llamacpp": "http://localhost:8080/v1",
}

DEFAULT_MODELS = {
    "openrouter": "arcee-ai/trinity-large-preview:free",
    "together": "meta-llama/Llama-3-8b-chat-hf",
    "groq": "llama-3.1-8b-instant",
    "fireworks": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "deepinfra": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "perplexity": "llama-3.1-sonar-small-128k-online",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "gemini": "gemini-2.0-flash",
    "mistral": "mistral-small-latest",
    "cohere": "command-r-plus",
    "xai": "grok-beta",
    "deepseek": "deepseek-chat",
    "cerebras": "llama3.1-8b",
    "nvidia": "meta/llama-3.1-8b-instruct",
    "ollama": "llama3",
    "lmstudio": "local-model",
    "llamacpp": "local-model",
}

ENV_KEYS = {
    "openrouter": "OPENROUTER_API_KEY",
    "together": "TOGETHER_API_KEY",
    "groq": "GROQ_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "anyscale": "ANYSCALE_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "ollama": None,
    "lmstudio": None,
    "llamacpp": None,
}

LOCAL_PROVIDERS = {"ollama", "lmstudio", "llamacpp"}


def _detect_provider(provider, model, base_url):
    if provider:
        return provider
    if base_url:
        for name, url in PROVIDERS.items():
            if url in base_url:
                return name
        return "openrouter"
    if model:
        if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
            return "openai"
        if model.startswith("gemini"):
            return "gemini"
        if model.startswith("claude"):
            return "anthropic"
        if model.startswith(("mistral", "codestral")):
            return "mistral"
        if model.startswith("command"):
            return "cohere"
        if model.startswith("grok"):
            return "xai"
        if model.startswith("deepseek"):
            return "deepseek"
        if model.startswith(("llama", "mixtral")):
            return "groq"
        if "/" in model:
            return "openrouter"
    return "openrouter"


class LLMClient:
    """
    Unified LLM client supporting all major providers via an OpenAI-compatible API.

    Parameters
    ----------
    provider : str, optional
        Provider name (e.g. "openai", "anthropic", "openrouter", "ollama").
        Auto-detected from model name or base_url when omitted.
    model : str, optional
        Model identifier. Defaults to the provider's recommended default.
    api_key : str, optional
        API key. Takes priority over api_key_env and environment lookup.
    api_key_env : str, optional
        Environment variable name to read the API key from.
        Overrides the provider's default env var (e.g. OPENROUTER_API_KEY).
    base_url : str, optional
        Custom API base URL. Overrides the provider's default endpoint.
    """

    def __init__(
        self, provider=None, model=None, api_key=None, api_key_env=None, base_url=None
    ):
        self.provider = _detect_provider(provider, model, base_url)
        self.model = model or DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")
        self.base_url = base_url or PROVIDERS.get(
            self.provider, PROVIDERS["openrouter"]
        )
        self.api_key = self._resolve_key(api_key, api_key_env)

    def _resolve_key(self, api_key, api_key_env):
        if api_key:
            return api_key
        if self.provider in LOCAL_PROVIDERS:
            return "local"
        env_var = api_key_env or ENV_KEYS.get(self.provider)
        if env_var:
            key = os.environ.get(env_var)
            if not key:
                raise ValueError(
                    f"[fenn] API key not found for provider '{self.provider}'.\n"
                    f"Set {env_var} in your .env file or pass api_key='...' to LLMClient()."
                )
            return key
        return "local"

    def _openai_client(self):
        try:
            from openai import OpenAI

            return OpenAI(api_key=self.api_key or "local", base_url=self.base_url)
        except ImportError:
            raise ImportError(
                "[fenn] 'openai' package not found. Run: pip install openai"
            )

    def chat_complete(self, messages, schema=None, retries=3):
        """
        Call the chat completions API with a list of message dicts.

        Parameters
        ----------
        messages : list of dict
            Messages in OpenAI format: [{"role": "user", "content": "..."}].
        schema : pydantic.BaseModel, optional
            If provided, instructs the model to return JSON matching this schema.
        retries : int
            Number of retry attempts on rate limit errors.

        Returns
        -------
        str or pydantic.BaseModel
        """
        try:
            from openai import RateLimitError
        except ImportError:
            raise ImportError(
                "[fenn] 'openai' package not found. Run: pip install openai"
            )

        client = self._openai_client()
        msgs = [
            dict(m) for m in messages
        ]  # shallow-copy to avoid mutating caller's list
        kwargs = dict(model=self.model, messages=msgs)

        if schema:
            msgs[-1]["content"] += (
                f"\n\nRespond ONLY with a valid JSON object matching this schema:\n"
                f"{schema.model_json_schema()}\n"
                f"Do not include any text outside the JSON object."
            )
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
                    logger.warning(
                        f"[fenn] rate limit hit, retrying in {wait}s... ({attempt + 1}/{retries})"
                    )
                    time.sleep(wait)
                else:
                    raise

    def ask(self, prompt, schema=None, retries=3):
        """
        Send a single prompt and return the response.

        Parameters
        ----------
        prompt : str
            The user message to send.
        schema : pydantic.BaseModel, optional
            If provided, validates the response against this schema.
        retries : int
            Retry attempts on rate limit errors.

        Returns
        -------
        str or pydantic.BaseModel
        """
        return self.chat_complete(
            [{"role": "user", "content": prompt}], schema=schema, retries=retries
        )

    def stream(self, prompt):
        """
        Send a prompt and yield response tokens one by one.

        Parameters
        ----------
        prompt : str
            The user message to send.

        Yields
        ------
        str
            Individual tokens from the LLM response.
        """
        client = self._openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in response:
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue
            delta = getattr(chunk.choices[0].delta, "content", "") or ""
            if delta:
                yield delta
