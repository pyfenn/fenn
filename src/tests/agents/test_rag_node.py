import tempfile
import textwrap
from pathlib import Path
import pytest
from fenn.agents import Flow
from fenn.agents.rag import RAGNode


@pytest.fixture()
def knowledge_dir(tmp_path):
    (tmp_path / "a.md").write_text(
        textwrap.dedent("""\
        # Fenn
        Fenn is a Python framework for machine learning workflows.
        It provides YAML configuration, logging, and training abstractions.
        """),
        encoding="utf-8",
    )
    (tmp_path / "b.txt").write_text(
        textwrap.dedent("""\
        LLMClient supports OpenAI, Anthropic, Gemini, and local providers.
        Use LLMClient(provider='openai') to create a client.
        """),
        encoding="utf-8",
    )
    return tmp_path


# ── Construction / indexing ────────────────────────────────────────────────────

def test_ragnode_indexes_on_init(knowledge_dir):
    node = RAGNode(sources=str(knowledge_dir))
    assert len(node._retriever.chunks) > 0


def test_ragnode_add_source(tmp_path):
    f = tmp_path / "extra.txt"
    f.write_text("Extra knowledge about neural networks.", encoding="utf-8")
    node = RAGNode()
    node.add_source(str(f))
    assert len(node._retriever.chunks) > 0


# ── BM25 retrieval via exec ────────────────────────────────────────────────────

def test_ragnode_exec_returns_chunks(knowledge_dir):
    node = RAGNode(sources=str(knowledge_dir), top_k=3)
    chunks = node.exec("Python framework")
    assert isinstance(chunks, list)
    assert len(chunks) <= 3


def test_ragnode_retrieves_relevant_chunk(knowledge_dir):
    node = RAGNode(sources=str(knowledge_dir), top_k=5)
    chunks = node.exec("LLMClient provider")
    assert any("LLMClient" in c for c in chunks), f"expected LLMClient in chunks: {chunks}"


# ── post() writes to shared ────────────────────────────────────────────────────

def test_ragnode_post_writes_shared(knowledge_dir):
    node = RAGNode(sources=str(knowledge_dir))
    shared = {}
    chunks = node.exec("Fenn framework")
    action = node.post(shared, "Fenn framework", chunks)

    assert "rag_chunks" in shared
    assert "rag_context" in shared
    assert shared["rag_chunks"] == chunks
    assert shared["rag_context"] == "\n\n".join(chunks)
    assert action == "default"


def test_ragnode_custom_next_action(knowledge_dir):
    node = RAGNode(sources=str(knowledge_dir), next_action="retrieved")
    shared = {}
    chunks = node.exec("anything")
    action = node.post(shared, "anything", chunks)
    assert action == "retrieved"


def test_ragnode_custom_keys(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Fenn framework docs.", encoding="utf-8")
    node = RAGNode(
        sources=str(f),
        query_key="q",
        context_key="ctx",
        chunks_key="raw",
    )
    shared = {"q": "Fenn"}
    query  = node.prep(shared)
    chunks = node.exec(query)
    node.post(shared, query, chunks)

    assert "ctx" in shared
    assert "raw" in shared


# ── Flow integration ───────────────────────────────────────────────────────────

def test_ragnode_in_flow(knowledge_dir):
    from fenn.agents import Node

    class _Answer(Node):
        def prep(self, shared):
            return shared.get("rag_context", "")
        def post(self, shared, prep_res, exec_res):
            shared["used_context"] = prep_res
            return "default"

    rag    = RAGNode(sources=str(knowledge_dir), query_key="query")
    answer = _Answer()

    flow = Flow(start=rag)
    flow.connect(rag, answer).connect(answer, None)

    shared = {"query": "Fenn framework"}
    flow.run(shared)

    assert "rag_context" in shared
    assert shared.get("used_context") == shared["rag_context"]
