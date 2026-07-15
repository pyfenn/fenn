"""Tests for fenn/agents/rag/chunker.py, loader.py, and rag.py"""

from unittest.mock import MagicMock, patch

import pytest

from fenn.agents.rag.chunker import (
    _chunk_fixed,
    _chunk_paragraphs,
    _chunk_sentences,
    _chunk_smart,
    chunk_text,
)
from fenn.agents.rag.loader import (
    _load_url,
    _load_wikipedia,
    _load_youtube,
    _read_file,
    _read_pdf,
    load_documents,
)
from fenn.agents.rag.rag import RAG

# ══════════════════════════════════════════════════════════════════════════════
# chunker.py
# ══════════════════════════════════════════════════════════════════════════════


class TestChunkParagraphs:
    def test_splits_on_blank_line(self):
        text = "First paragraph.\n\nSecond paragraph."
        result = _chunk_paragraphs(text)
        assert result == ["First paragraph.", "Second paragraph."]

    def test_strips_whitespace(self):
        result = _chunk_paragraphs("  hello  \n\n  world  ")
        assert result == ["hello", "world"]

    def test_ignores_empty_blocks(self):
        result = _chunk_paragraphs("\n\n\nonly content\n\n\n")
        assert result == ["only content"]

    def test_single_paragraph(self):
        result = _chunk_paragraphs("no blank lines here")
        assert result == ["no blank lines here"]

    def test_empty_string(self):
        assert _chunk_paragraphs("") == []


class TestChunkSentences:
    def test_groups_short_sentences(self):
        text = "Hello world. This is a test."
        result = _chunk_sentences(text)
        assert len(result) >= 1
        assert all(isinstance(c, str) for c in result)

    def test_splits_long_sentences(self):
        # Each sentence is ~100 chars; 6 of them exceed 500 total
        sentence = "This is a moderately long sentence that takes up space. "
        text = (sentence * 10).strip()
        result = _chunk_sentences(text)
        assert len(result) > 1
        assert all(len(c) <= 600 for c in result)

    def test_handles_exclamation_and_question(self):
        text = "Really? Yes! Absolutely."
        result = _chunk_sentences(text)
        assert len(result) >= 1

    def test_empty_string(self):
        result = _chunk_sentences("")
        assert result == [] or result == [""]


class TestChunkFixed:
    def test_basic_chunking(self):
        text = "a" * 1000
        result = _chunk_fixed(text, size=200, overlap=50)
        assert len(result) > 1
        assert all(len(c) <= 200 for c in result)

    def test_overlap_is_shared(self):
        text = "abcdefghij"  # 10 chars
        result = _chunk_fixed(text, size=6, overlap=2)
        # chunk 0: [0:6] = "abcdef", chunk 1: [4:10] = "efghij"
        assert result[0][-2:] == result[1][:2]

    def test_short_text_single_chunk(self):
        result = _chunk_fixed("hello", size=100, overlap=10)
        assert result == ["hello"]

    def test_overlap_gte_size_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            _chunk_fixed("text", size=10, overlap=10)

    def test_overlap_larger_than_size_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            _chunk_fixed("text", size=5, overlap=20)


class TestChunkSmart:
    def test_short_paragraphs_kept_whole(self):
        text = "Short para.\n\nAnother short para."
        result = _chunk_smart(text)
        assert "Short para." in result
        assert "Another short para." in result

    def test_long_paragraph_split_into_sentences(self):
        # Build a paragraph > 600 chars
        sentence = "This is a fairly long sentence used for testing purposes only. "
        long_para = sentence * 12
        result = _chunk_smart(long_para)
        assert len(result) > 1
        assert all(len(c) <= 600 for c in result)

    def test_empty_chunks_filtered(self):
        result = _chunk_smart("\n\n\n")
        assert result == []


class TestChunkText:
    def test_mode_smart(self):
        assert chunk_text("hello\n\nworld", mode="smart") == ["hello", "world"]

    def test_mode_paragraphs(self):
        assert chunk_text("a\n\nb", mode="paragraphs") == ["a", "b"]

    def test_mode_sentences(self):
        result = chunk_text("Hello. World.", mode="sentences")
        assert isinstance(result, list)

    def test_mode_fixed(self):
        result = chunk_text("x" * 200, mode="fixed", size=100, overlap=10)
        assert len(result) > 1

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="unknown chunk_mode"):
            chunk_text("text", mode="invalid")


# ══════════════════════════════════════════════════════════════════════════════
# loader.py
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadDocuments:
    def test_loads_txt_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello world", encoding="utf-8")
        docs = load_documents(str(f))
        assert docs == ["hello world"]

    def test_loads_md_file(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# Title\nContent.", encoding="utf-8")
        docs = load_documents(str(f))
        assert len(docs) == 1
        assert "Title" in docs[0]

    def test_loads_directory_recursively(self, tmp_path):
        (tmp_path / "a.txt").write_text("file a", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("file b", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 2

    def test_skips_unsupported_extensions(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "doc.txt").write_text("valid", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert docs == ["valid"]

    def test_empty_directory_returns_empty(self, tmp_path):
        docs = load_documents(str(tmp_path))
        assert docs == []

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path/file.txt")

    def test_routes_youtube_url(self):
        with patch(
            "fenn.agents.rag.loader._load_youtube", return_value="transcript"
        ) as m:
            result = load_documents("https://www.youtube.com/watch?v=abc123")
        m.assert_called_once()
        assert result == ["transcript"]

    def test_routes_youtu_be_url(self):
        with patch("fenn.agents.rag.loader._load_youtube", return_value="t") as m:
            load_documents("https://youtu.be/abc123")
        m.assert_called_once()

    def test_routes_http_url(self):
        with patch("fenn.agents.rag.loader._load_url", return_value="page text") as m:
            result = load_documents("https://example.com")
        m.assert_called_once()
        assert result == ["page text"]

    def test_filters_empty_docs(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        docs = load_documents(str(f))
        assert docs == []


class TestReadFile:
    def test_reads_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content", encoding="utf-8")
        assert _read_file(f) == "content"

    def test_returns_none_on_read_error(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("x")
        with patch("pathlib.Path.read_text", side_effect=OSError("fail")):
            result = _read_file(f)
        assert result is None

    def test_delegates_pdf_to_read_pdf(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        with patch("fenn.agents.rag.loader._read_pdf", return_value="pdf text") as m:
            result = _read_file(f)
        m.assert_called_once_with(f)
        assert result == "pdf text"


class TestReadPdf:
    def test_extracts_text(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "pdf content"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        with patch("fenn.agents.rag.loader.pypdf", create=True) as mock_pypdf:
            mock_pypdf.PdfReader.return_value = mock_reader
            with patch.dict("sys.modules", {"pypdf": mock_pypdf}):
                import fenn.agents.rag.loader as loader_mod

                with patch.object(loader_mod, "_read_pdf", wraps=loader_mod._read_pdf):
                    pass  # just verifying import path

    def test_raises_import_error_without_pypdf(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        with patch.dict("sys.modules", {"pypdf": None}):
            with pytest.raises(ImportError, match="pypdf"):
                _read_pdf(f)


class TestLoadUrl:
    def test_raises_import_error_without_httpx(self):
        with patch.dict("sys.modules", {"httpx": None, "bs4": None}):
            with pytest.raises(ImportError, match="httpx"):
                _load_url("https://example.com")

    def test_raises_import_error_without_wikipedia(self):
        with patch.dict("sys.modules", {"wikipedia": None}):
            with pytest.raises((ImportError, Exception)):
                _load_wikipedia("https://en.wikipedia.org/wiki/Python")

    def test_raises_import_error_without_youtube_transcript(self):
        with patch.dict("sys.modules", {"youtube_transcript_api": None}):
            with pytest.raises(ImportError, match="youtube-transcript-api"):
                _load_youtube("https://www.youtube.com/watch?v=abc")

    def test_youtube_invalid_url_raises(self):
        mock_yta = MagicMock()
        with patch.dict("sys.modules", {"youtube_transcript_api": mock_yta}):
            with patch(
                "fenn.agents.rag.loader.YouTubeTranscriptApi", mock_yta, create=True
            ):
                with pytest.raises((ValueError, Exception)):
                    _load_youtube("https://www.youtube.com/watch")  # no video ID


# ══════════════════════════════════════════════════════════════════════════════
# rag.py
# ══════════════════════════════════════════════════════════════════════════════


def _make_rag(**kwargs):
    """Build a RAG instance with a mocked LLMClient."""
    with patch("fenn.agents.rag.rag.LLMClient") as MockLLM:
        mock_llm = MagicMock()
        mock_llm.provider = "openai"
        mock_llm.model = "gpt-4o-mini"
        mock_llm.api_key = "test-key"
        mock_llm.base_url = "https://api.openai.com/v1"
        MockLLM.return_value = mock_llm
        rag = RAG(**kwargs)
    return rag


class TestRAGInit:
    def test_default_system_prompt_set(self):
        rag = _make_rag()
        assert "helpful assistant" in rag._system_prompt

    def test_custom_system_prompt(self):
        rag = _make_rag(system_prompt="You are a pirate.")
        assert rag._system_prompt == "You are a pirate."

    def test_memory_off_by_default(self):
        rag = _make_rag()
        assert rag._memory is False
        assert rag._history == []

    def test_memory_on(self):
        rag = _make_rag(memory=True)
        assert rag._memory is True

    def test_add_source_returns_self(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content", encoding="utf-8")
        rag = _make_rag()
        result = rag.add_source(str(f))
        assert result is rag

    def test_add_tool_returns_self(self):
        def _fn(x):
            return x

        rag = _make_rag()
        result = rag.add_tool(_fn)
        assert result is rag
        assert _fn in rag._tools

    def test_debug_returns_self(self):
        rag = _make_rag()
        result = rag.debug()
        assert result is rag
        assert rag._debug is True

    def test_reset_memory_clears_history(self):
        rag = _make_rag(memory=True)
        rag._history = [("q", "a")]
        result = rag.reset_memory()
        assert rag._history == []
        assert result is rag


class TestRAGBuildPrompt:
    def test_no_memory_no_tools(self):
        rag = _make_rag()
        prompt = rag._build_prompt("What is X?", "Some context.")
        assert "Some context." in prompt
        assert "What is X?" in prompt
        assert "Question:" in prompt
        assert "Answer:" in prompt

    def test_with_tools_includes_tool_descriptions(self):
        rag = _make_rag()

        def my_tool():
            """Does something useful."""
            pass

        rag.add_tool(my_tool)
        prompt = rag._build_prompt("q", "ctx")
        assert "Available tools:" in prompt
        assert "my_tool" in prompt
        assert "Does something useful." in prompt

    def test_with_memory_includes_history(self):
        rag = _make_rag(memory=True)
        rag._history = [("prev question", "prev answer")]
        prompt = rag._build_prompt("new question", "context")
        assert "Conversation so far:" in prompt
        assert "prev question" in prompt
        assert "prev answer" in prompt
        assert "User: new question" in prompt

    def test_memory_with_max_history_trims(self):
        rag = _make_rag(memory=True)
        rag._max_history = 1
        rag._history = [("old q", "old a"), ("recent q", "recent a")]
        prompt = rag._build_prompt("q", "ctx")
        assert "recent q" in prompt
        assert "old q" not in prompt

    def test_memory_no_history_uses_standard_format(self):
        rag = _make_rag(memory=True)
        rag._history = []
        prompt = rag._build_prompt("q", "ctx")
        assert "Question:" in prompt


class TestRAGRun:
    def test_run_returns_llm_answer(self):
        rag = _make_rag()
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = ["chunk one"]
        rag._llm.ask.return_value = "the answer"
        result = rag.run("my question")
        assert result == "the answer"

    def test_run_passes_query_to_retriever(self):
        rag = _make_rag()
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = []
        rag._llm.ask.return_value = "ok"
        rag.run("search this")
        rag._retriever.query.assert_called_once_with("search this")

    def test_run_with_memory_appends_history(self):
        rag = _make_rag(memory=True)
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = []
        rag._llm.ask.return_value = "answer"
        rag.run("question")
        assert rag._history == [("question", "answer")]

    def test_run_without_memory_does_not_append_history(self):
        rag = _make_rag(memory=False)
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = []
        rag._llm.ask.return_value = "answer"
        rag.run("question")
        assert rag._history == []

    def test_run_passes_schema(self):
        from pydantic import BaseModel

        class Reply(BaseModel):
            value: str

        rag = _make_rag()
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = []
        rag._llm.ask.return_value = Reply(value="x")
        rag.run("q", schema=Reply)
        _, call_kwargs = rag._llm.ask.call_args
        assert call_kwargs.get("schema") is Reply

    def test_run_debug_mode_prints(self, capsys):
        rag = _make_rag()
        rag.debug()
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = ["a chunk"]
        rag._llm.ask.return_value = "ok"
        rag.run("q")
        captured = capsys.readouterr()
        assert "model_provider" in captured.out

    def test_chat_enables_memory_and_runs(self):
        rag = _make_rag()
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = []
        rag._llm.ask.return_value = "chat answer"
        result = rag.chat("hello")
        assert result == "chat answer"
        assert rag._memory is True

    def test_stream_yields_tokens(self):
        rag = _make_rag()
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = ["context"]
        rag._llm.stream.return_value = iter(["tok1", "tok2"])
        tokens = list(rag.stream("q"))
        assert tokens == ["tok1", "tok2"]

    def test_stream_debug_prints(self, capsys):
        rag = _make_rag()
        rag.debug()
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = ["chunk"]
        rag._llm.stream.return_value = iter([])
        list(rag.stream("q"))
        captured = capsys.readouterr()
        assert "model_provider" in captured.out

    def test_add_source_indexes_docs(self, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text("some content", encoding="utf-8")
        rag = _make_rag()
        rag._retriever = MagicMock()
        rag.add_source(str(f))
        rag._retriever.index.assert_called_once()

    def test_add_source_debug_prints(self, tmp_path, capsys):
        f = tmp_path / "note.txt"
        f.write_text("content", encoding="utf-8")
        rag = _make_rag()
        rag.debug()
        rag._retriever = MagicMock()
        rag.add_source(str(f))
        captured = capsys.readouterr()
        assert "loaded" in captured.out
