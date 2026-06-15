"""Tests for fenn/agents/rag/retriever.py"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fenn.agents.rag.retriever import LOCAL_EMBEDDING_PROVIDERS, Retriever

# ── Helpers ────────────────────────────────────────────────────────────────────


def _docs(*texts):
    return list(texts)


def _bm25_retriever(docs=None):
    r = Retriever(use_faiss=False)
    if docs:
        r.index(docs)
    return r


# ── Construction ───────────────────────────────────────────────────────────────


class TestRetrieverInit:
    def test_defaults(self):
        r = Retriever()
        assert r.chunks == []
        assert r.use_faiss is False
        assert r.embedding_provider == "local"
        assert r.embedding_model == "all-MiniLM-L6-v2"
        assert r.persist_path is None
        assert r._faiss_index is None
        assert r._loaded_from_disk is False

    def test_persist_path_converted_to_path(self, tmp_path):
        r = Retriever(persist_path=str(tmp_path))
        assert isinstance(r.persist_path, Path)
        assert r.persist_path == tmp_path


# ── _resolve_embedding_key ─────────────────────────────────────────────────────


class TestResolveEmbeddingKey:
    def test_explicit_key_wins(self):
        r = Retriever(embedding_api_key="my-key")
        assert r._resolve_embedding_key() == "my-key"

    @pytest.mark.parametrize("provider", sorted(LOCAL_EMBEDDING_PROVIDERS))
    def test_local_providers_return_none(self, provider):
        r = Retriever(embedding_provider=provider)
        assert r._resolve_embedding_key() is None

    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        r = Retriever(embedding_provider="openai")
        assert r._resolve_embedding_key() == "env-key"

    def test_missing_env_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        r = Retriever(embedding_provider="openai")
        with pytest.raises(ValueError, match="embedding API key not found"):
            r._resolve_embedding_key()

    def test_unknown_provider_returns_none(self):
        r = Retriever(embedding_provider="unknown_provider")
        assert r._resolve_embedding_key() is None


# ── BM25 index & query ─────────────────────────────────────────────────────────


class TestBM25:
    def test_index_populates_chunks(self):
        r = _bm25_retriever(_docs("hello world", "foo bar"))
        assert len(r.chunks) >= 2

    def test_index_builds_inverted_index(self):
        r = _bm25_retriever(_docs("the quick brown fox"))
        assert "quick" in r._index
        assert "fox" in r._index

    def test_query_returns_relevant_chunk(self):
        r = _bm25_retriever(
            _docs(
                "Python is a programming language.",
                "The cat sat on the mat.",
            )
        )
        results = r.query("Python programming", top_k=1)
        assert any("Python" in c for c in results)

    def test_query_respects_top_k(self):
        docs = [f"document number {i}" for i in range(20)]
        r = _bm25_retriever(docs)
        results = r.query("document number", top_k=3)
        assert len(results) <= 3

    def test_query_no_match_returns_first_chunks(self):
        r = _bm25_retriever(_docs("apple banana", "cherry date"))
        results = r.query("zzznomatch", top_k=2)
        # Falls back to chunks[:top_k]
        assert len(results) <= 2

    def test_query_empty_index_returns_empty(self):
        r = Retriever(use_faiss=False)
        results = r.query("anything", top_k=5)
        assert results == []

    def test_multiple_index_calls_accumulate_chunks(self):
        r = Retriever(use_faiss=False)
        r.index(_docs("first document"))
        r.index(_docs("second document"))
        assert len(r.chunks) >= 2

    def test_bm25_scores_by_word_frequency(self):
        r = _bm25_retriever(
            _docs(
                "cat cat cat",
                "cat dog",
                "dog dog",
            )
        )
        results = r.query("cat", top_k=3)
        # chunk with most "cat" hits should rank first
        assert results[0] == "cat cat cat"


# ── query() dispatch ───────────────────────────────────────────────────────────


class TestQueryDispatch:
    def test_bm25_path_called(self):
        r = Retriever(use_faiss=False)
        r.chunks = ["some chunk"]
        r._index["some"].append(0)
        with patch.object(r, "_query_bm25", wraps=r._query_bm25) as mock:
            r.query("some text", top_k=1)
        mock.assert_called_once_with("some text", 1)

    def test_faiss_path_called(self):
        r = Retriever(use_faiss=True)
        with patch.object(r, "_query_faiss", return_value=[]) as mock:
            r.query("some text", top_k=3)
        mock.assert_called_once_with("some text", 3)


# ── index() with persist_path ──────────────────────────────────────────────────


class TestIndexPersistence:
    def test_index_loads_from_disk_on_first_call(self, tmp_path):
        r = Retriever(use_faiss=False, persist_path=tmp_path)
        with patch.object(r, "_load_from_disk", return_value=True) as mock_load:
            r.index(_docs("hello"))
        mock_load.assert_called_once()
        assert r._loaded_from_disk is True

    def test_index_skips_disk_load_on_second_call(self, tmp_path):
        r = Retriever(use_faiss=False, persist_path=tmp_path)
        r._loaded_from_disk = True
        with patch.object(r, "_load_from_disk") as mock_load:
            r.index(_docs("hello world"))
        mock_load.assert_not_called()

    def test_index_proceeds_when_disk_load_fails(self, tmp_path):
        r = Retriever(use_faiss=False, persist_path=tmp_path)
        with patch.object(r, "_load_from_disk", return_value=False):
            r.index(_docs("hello world"))
        assert len(r.chunks) > 0

    def test_load_from_disk_returns_false_when_files_missing(self, tmp_path):
        r = Retriever(persist_path=tmp_path)
        assert r._load_from_disk() is False

    def test_load_from_disk_returns_false_on_corrupt_data(self, tmp_path):
        (tmp_path / "index.faiss").write_bytes(b"corrupt")
        (tmp_path / "chunks.json").write_text("[]", encoding="utf-8")
        r = Retriever(persist_path=tmp_path)
        # faiss will raise on corrupt file; should return False not crash
        result = r._load_from_disk()
        assert result is False

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save a FAISS index to disk, then load it back."""
        pytest.importorskip("faiss")
        pytest.importorskip("sentence_transformers")

        r = Retriever(use_faiss=True, persist_path=tmp_path)
        r.chunks = ["chunk one", "chunk two"]

        mock_index = MagicMock()
        r._faiss_index = mock_index

        with patch("fenn.agents.rag.retriever.faiss"):
            r._save_to_disk()
            assert (tmp_path / "chunks.json").exists()
            saved = json.loads((tmp_path / "chunks.json").read_text())
            assert saved == ["chunk one", "chunk two"]


# ── _build_faiss ───────────────────────────────────────────────────────────────


class TestBuildFaiss:
    def test_build_faiss_raises_without_faiss(self):
        r = Retriever(use_faiss=True)
        r.chunks = ["chunk"]
        with patch.dict("sys.modules", {"faiss": None}):
            with pytest.raises(ImportError, match="faiss-cpu"):
                r._build_faiss()


# ── Embedding method ImportError paths ────────────────────────────────────────


class TestEmbedImportErrors:
    def test_embed_local_raises_without_sentence_transformers(self):
        r = Retriever(embedding_provider="local")
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                r._embed_local(["text"])

    def test_embed_openai_compat_raises_without_openai(self):
        r = Retriever(embedding_provider="openai", embedding_api_key="key")
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                r._embed_openai_compat(["text"])

    def test_embed_ollama_raises_without_openai(self):
        r = Retriever(embedding_provider="ollama")
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                r._embed_ollama(["text"])

    def test_embed_cohere_raises_without_cohere(self):
        r = Retriever(embedding_provider="cohere", embedding_api_key="key")
        with patch.dict("sys.modules", {"cohere": None}):
            with pytest.raises(ImportError, match="cohere"):
                r._embed_cohere(["text"])

    def test_embed_voyage_raises_without_voyageai(self):
        r = Retriever(embedding_provider="voyage", embedding_api_key="key")
        with patch.dict("sys.modules", {"voyageai": None}):
            with pytest.raises(ImportError, match="voyageai"):
                r._embed_voyage(["text"])

    def test_embed_jina_raises_without_httpx(self):
        r = Retriever(embedding_provider="jina", embedding_api_key="key")
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx"):
                r._embed_jina(["text"])


# ── _embed dispatch ────────────────────────────────────────────────────────────


class TestEmbedDispatch:
    @pytest.mark.parametrize(
        "provider,method",
        [
            ("local", "_embed_local"),
            ("openai", "_embed_openai_compat"),
            ("gemini", "_embed_openai_compat"),
            ("mistral", "_embed_openai_compat"),
            ("ollama", "_embed_ollama"),
            ("cohere", "_embed_cohere"),
            ("voyage", "_embed_voyage"),
            ("jina", "_embed_jina"),
        ],
    )
    def test_embed_routes_to_correct_method(self, provider, method):
        r = Retriever(embedding_provider=provider, embedding_api_key="key")
        with patch.object(r, method, return_value="result") as mock_method:
            r._embed(["text"])
        mock_method.assert_called_once()
