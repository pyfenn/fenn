from typing import Any

from fenn.agents import Node

from .loader import load_documents
from .retriever import Retriever


class RAGNode(Node):
    """
    Flow node that retrieves relevant context from indexed sources.

    Loads and indexes all sources once at construction time, then per run
    queries the index using ``shared[query_key]`` and writes the results
    into ``shared[chunks_key]`` and ``shared[context_key]``.

    Parameters
    ----------
    sources : str or list of str, optional
        File paths, folder paths, or URLs to load and index on init.
        Additional sources can be indexed later with :meth:`add_source`.
    query_key : str
        Key in ``shared`` that holds the user query. Default: ``"query"``.
    context_key : str
        Key written into ``shared`` with the concatenated chunk text.
        Default: ``"rag_context"``.
    chunks_key : str
        Key written into ``shared`` with the raw list of chunks.
        Default: ``"rag_chunks"``.
    top_k : int
        Maximum number of chunks to retrieve. Default: 5.
    next_action : str
        Action string returned by ``post()``, used by ``Flow.get_next_node()``.
        Default: ``"default"``.
    faiss : bool
        Use FAISS semantic search instead of BM25. Default: False.
    embedding_provider : str
        Embedding provider (only used when faiss=True). Default: ``"local"``.
    embedding_model : str
        Embedding model (only used when faiss=True). Default: ``"all-MiniLM-L6-v2"``.
    embedding_api_key : str, optional
        API key for the embedding provider.
    chunk_mode : str
        Document chunking strategy. One of ``"smart"``, ``"paragraphs"``,
        ``"sentences"``, ``"fixed"``. Default: ``"smart"``.
    persist_path : str or Path, optional
        Directory to save/load the FAISS index. Only used when faiss=True.
    """

    def __init__(
        self,
        sources: str | list[str] | None = None,
        query_key: str = "query",
        context_key: str = "rag_context",
        chunks_key: str = "rag_chunks",
        top_k: int = 5,
        next_action: str = "default",
        faiss: bool = False,
        embedding_provider: str = "local",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_api_key: str | None = None,
        chunk_mode: str = "smart",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self._query_key = query_key
        self._context_key = context_key
        self._chunks_key = chunks_key
        self._top_k = top_k
        self._next_action = next_action

        self._retriever = Retriever(
            use_faiss=faiss,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            chunk_mode=chunk_mode,
            persist_path=persist_path,
        )

        if sources:
            if isinstance(sources, str):
                sources = [sources]
            for s in sources:
                self._retriever.index(load_documents(s))

    def add_source(self, source: str) -> "RAGNode":
        """Index an additional source. Returns self for chaining."""
        self._retriever.index(load_documents(source))
        return self

    def prep(self, shared: dict[str, Any]) -> str:
        return shared.get(self._query_key, "")

    def exec(self, query: str) -> list[str]:
        return self._retriever.query(query, top_k=self._top_k)

    def post(self, shared: dict[str, Any], query: str, chunks: list[str]) -> str:
        shared[self._chunks_key] = chunks
        shared[self._context_key] = "\n\n".join(chunks)
        return self._next_action
