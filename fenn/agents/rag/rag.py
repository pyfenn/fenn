from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fenn.agents.llm import LLMClient
from fenn.logging import logger

from .loader import load_documents
from .retriever import Retriever


class RAG:
    """
    Main RAG (Retrieval-Augmented Generation) class.

    Loads documents, indexes them, retrieves relevant chunks for a query,
    and sends them to an LLM to generate an answer.

    Parameters
    ----------
    model_provider : str, optional
        LLM provider name. One of: openrouter, openai, anthropic, gemini,
        mistral, groq, cohere, deepseek, xai, together, perplexity,
        fireworks, cerebras, nvidia, deepinfra, anyscale,
        ollama, lmstudio, llamacpp.
        Auto-detected from `model` name when possible.
        Default: "openrouter".

    model : str, optional
        Model identifier for the chosen provider.
        Default: provider's default model (e.g. "arcee-ai/trinity-large-preview:free"
        for openrouter, "gpt-4o-mini" for openai).

    model_api_key : str, optional
        API key for the LLM provider. If omitted, reads from the
        corresponding environment variable (e.g. OPENROUTER_API_KEY).
        Not required for local providers (ollama, lmstudio, llamacpp).

    base_url : str, optional
        Custom base URL for any OpenAI-compatible API endpoint.
        Overrides the default URL for the detected provider.

    faiss : bool, optional
        If True, uses FAISS semantic vector search instead of BM25 keyword
        search. Requires `pip install "cofone[faiss]"`. Default: False.

    embedding_provider : str, optional
        Provider for text embeddings (used only when faiss=True).
        One of: local, openai, gemini, cohere, mistral, voyage, jina,
        nvidia, together, openrouter, ollama.
        Default: "local" (sentence-transformers, no API key needed).

    embedding_model : str, optional
        Embedding model identifier for the chosen embedding provider.
        Default: "all-MiniLM-L6-v2" (local sentence-transformers).

    embedding_api_key : str, optional
        API key for the embedding provider. If omitted, reads from the
        corresponding environment variable. Not required for local/ollama.

    chunk_mode : str, optional
        How to split documents before indexing.
        One of: "smart" (default), "paragraphs", "sentences", "fixed".

    persist_path : str or Path, optional
        Folder path to save/load the FAISS index to/from disk.
        Avoids recomputing embeddings on subsequent runs.
        Only used when faiss=True.

    system_prompt : str, optional
        Instructions prepended to every prompt as a system message.
        Tells the LLM how to behave: role, tone, language, format, etc.
        If None, a sensible default is used.
        Default: None (uses built-in default prompt).

    memory : bool, optional
        If True, keeps conversation history across .run() calls.
        Default: False.

    Examples
    --------
    Minimal (reads OPENROUTER_API_KEY from .env):
        >>> from dotenv import load_dotenv
        >>> from cofone import RAG
        >>> load_dotenv()
        >>> answer = RAG().add_source("docs/").run("What is this about?")

    Explicit provider:
        >>> RAG(model_provider="openai", model="gpt-4o-mini",
        ...     model_api_key="sk-...").add_source("notes.txt").run("Summarize")

    Fully local (no internet, no keys):
        >>> RAG(model_provider="ollama", model="llama3",
        ...     faiss=True,
        ...     embedding_provider="ollama",
        ...     embedding_model="nomic-embed-text").add_source("docs/").run("question")
    """

    def __init__(
        self,
        # ── LLM ───────────────────────────────────────────
        model_provider: str | None = None,
        model: str | None = None,
        model_api_key: str | None = None,
        base_url: str | None = None,
        # ── Embeddings ────────────────────────────────────
        faiss: bool = False,
        embedding_provider: str = "local",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_api_key: str | None = None,
        # ── RAG behaviour ─────────────────────────────────
        chunk_mode: str = "smart",
        persist_path: str | Path | None = None,
        memory: bool = False,
        max_history: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._llm = LLMClient(
            provider=model_provider,
            model=model,
            api_key=model_api_key,
            base_url=base_url,
        )
        self.model_provider = self._llm.provider
        self.model = self._llm.model
        self.model_api_key = self._llm.api_key
        self.base_url = self._llm.base_url

        self._retriever = Retriever(
            use_faiss=faiss,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            chunk_mode=chunk_mode,
            persist_path=persist_path,
        )
        self._tools = []
        self._debug = False
        self._memory = memory
        self._history = []  # list of (query, answer) tuples
        self._max_history = max_history
        self._system_prompt = system_prompt or (
            "You are a helpful assistant. "
            "Answer the user's question using ONLY the information provided in the context below. "
            "If the answer is not in the context, say you don't know. "
            "Be concise, accurate, and respond in the same language as the user's question."
        )

    def add_source(self, source: str | Path) -> "RAG":
        """
        Load and index a document source.

        Accepts: file path (.txt, .md, .pdf), folder path (recursive),
        web URL, Wikipedia URL, or YouTube URL.

        Can be chained: RAG().add_source("a.txt").add_source("b.txt").run(...)

        Parameters
        ----------
        source : str or Path
            Path to a file/folder, or a URL.

        Returns
        -------
        self : RAG
            Returns self for fluent chaining.
        """
        docs = load_documents(source)
        if self._debug:
            logger.info(f"[cofone] loaded {len(docs)} doc(s) from: {source}")
        self._retriever.index(docs)
        return self

    def add_tool(self, fn: Callable[..., Any]) -> "RAG":
        """
        Attach a custom Python function as a tool.

        The function's name and docstring are passed to the LLM as
        additional context alongside the retrieved chunks.

        Parameters
        ----------
        fn : callable
            A Python function. Should have a docstring describing what it does.

        Returns
        -------
        self : RAG
        """
        self._tools.append(fn)
        return self

    def debug(self) -> "RAG":
        """
        Enable verbose logging.

        Prints: provider, model, number of loaded docs, number of retrieved
        chunks, and an 80-character preview of each retrieved chunk.

        Returns
        -------
        self : RAG
        """
        self._debug = True
        return self

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the final prompt sent to the LLM, with optional memory."""
        tool_ctx = ""
        if self._tools:
            tool_descriptions = "\n".join(
                f"- {fn.__name__}: {(fn.__doc__ or '').strip()}" for fn in self._tools
            )
            tool_ctx = f"\n\nAvailable tools:\n{tool_descriptions}"

        if self._memory and self._history:
            history = (
                self._history[-self._max_history :]
                if self._max_history
                else self._history
            )
            history_str = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)
            return (
                f"{self._system_prompt}\n\n"
                f"Conversation so far:\n{history_str}\n\n"
                f"Context:\n{context}{tool_ctx}\n\n"
                f"User: {query}\nAssistant:"
            )
        return (
            f"{self._system_prompt}\n\n"
            f"Context:\n{context}{tool_ctx}\n\n"
            f"Question: {query}\nAnswer:"
        )

    def run(self, query: str, schema: Any = None) -> Any:
        """
        Run a single query against the indexed documents.

        Retrieves the most relevant chunks, builds a prompt, and calls
        the LLM. Stateless by default (no memory between calls).

        Parameters
        ----------
        query : str
            The question or instruction to send to the LLM.

        schema : pydantic.BaseModel, optional
            If provided, the LLM is instructed to return a JSON object
            matching this schema. Returns a validated Pydantic model
            instance instead of a string.

        Returns
        -------
        str or pydantic.BaseModel
            The LLM's answer, either as a string or a validated schema instance.
        """
        chunks = self._retriever.query(query)

        if self._debug:
            logger.info(
                f"\n[cofone] model_provider: {self.model_provider} | model: {self.model}"
            )
            logger.info(f"[cofone] query: {query}")
            logger.info(f"[cofone] chunks found: {len(chunks)}")
            for i, c in enumerate(chunks):
                logger.info(f"  [{i}] {c[:80]}...")

        context = "\n\n".join(chunks)
        prompt = self._build_prompt(query, context)
        answer = self._llm.ask(prompt, schema=schema)

        if self._memory:
            self._history.append(
                (query, answer if isinstance(answer, str) else str(answer))
            )

        return answer

    def chat(self, query: str) -> str:
        """
        Run a query with memory enabled.

        Equivalent to calling .run(query) with memory=True.
        Conversation history is preserved across calls on the same instance.

        Parameters
        ----------
        query : str
            The question or instruction.

        Returns
        -------
        str
            The LLM's answer.
        """
        self._memory = True
        return self.run(query)

    def stream(self, query: str) -> Iterator[str]:
        """
        Run a query and stream the response token by token.

        Returns a generator that yields string tokens as they arrive
        from the LLM. No waiting for the full response.

        Parameters
        ----------
        query : str
            The question or instruction.

        Yields
        ------
        str
            Individual tokens from the LLM response.

        Example
        -------
            for token in rag.stream("Explain this document"):
                print(token, end="", flush=True)
            print()
        """
        chunks = self._retriever.query(query)
        context = "\n\n".join(chunks)
        prompt = self._build_prompt(query, context)

        if self._debug:
            logger.info(
                f"\n[cofone] model_provider: {self.model_provider} | model: {self.model}"
            )
            logger.info(f"[cofone] query: {query}")
            logger.info(f"[cofone] chunks found: {len(chunks)}")
            for i, c in enumerate(chunks):
                logger.info(f"  [{i + 1}] {c[:80]}...")

        yield from self._llm.stream(prompt)

    def reset_memory(self) -> "RAG":
        """
        Clear the conversation history.

        After calling this, the next .chat() or .run() call will have
        no knowledge of previous exchanges.

        Returns
        -------
        self : RAG
        """
        self._history = []
        return self
