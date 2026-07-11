from collections.abc import Iterator
from typing import Any, Generic, TypeVar

_PrepResult = TypeVar("_PrepResult")
_ExecResult = TypeVar("_ExecResult")
_PostResult = TypeVar("_PostResult")

ParamValue = str | int | float | bool | None | list[Any] | dict[str, Any]
SharedData = dict[str, Any]
Params = dict[str, ParamValue]
Messages = list[dict[str, str]]

class BaseNode(Generic[_PrepResult, _ExecResult, _PostResult]):
    params: Params
    successors: dict[str, Any]

    def __init__(self) -> None: ...
    def set_params(self, params: Params) -> None: ...
    def prep(self, shared: SharedData) -> _PrepResult: ...
    def exec(self, prep_res: _PrepResult) -> _ExecResult: ...
    def post(
        self, shared: SharedData, prep_res: _PrepResult, exec_res: _ExecResult
    ) -> _PostResult: ...
    def _exec(self, prep_res: _PrepResult) -> _ExecResult: ...
    def _run(self, shared: SharedData) -> _PostResult: ...
    def run(self, shared: SharedData) -> _PostResult: ...

class Node(BaseNode[_PrepResult, _ExecResult, _PostResult]):
    max_retries: int
    wait: int | float
    cur_retry: int

    def __init__(self, max_retries: int = 1, wait: int | float = 0) -> None: ...
    def exec_fallback(self, prep_res: _PrepResult, exc: Exception) -> _ExecResult: ...
    def _exec(self, prep_res: _PrepResult) -> _ExecResult: ...

class BatchNode(Node[list[_PrepResult] | None, list[_ExecResult], _PostResult]):
    def _exec(self, items: list[_PrepResult] | None) -> list[_ExecResult]: ...

class Flow(BaseNode[_PrepResult, Any, _PostResult]):
    start_node: BaseNode[Any, Any, Any] | None

    def __init__(self, start: BaseNode[Any, Any, Any] | None = None) -> None: ...
    def start(self, start: BaseNode[Any, Any, Any]) -> BaseNode[Any, Any, Any]: ...
    def connect(
        self,
        src: BaseNode[Any, Any, Any],
        dst: BaseNode[Any, Any, Any] | None,
        action: str = "default",
    ) -> "Flow[_PrepResult, _PostResult]": ...
    def get_next_node(
        self, curr: BaseNode[Any, Any, Any], action: str | None
    ) -> BaseNode[Any, Any, Any] | None: ...
    def _orch(self, shared: SharedData, params: Params | None = None) -> Any: ...
    def _run(self, shared: SharedData) -> _PostResult: ...
    def post(
        self, shared: SharedData, prep_res: _PrepResult, exec_res: Any
    ) -> _PostResult: ...

class BatchFlow(Flow[list[Params] | None, Any, _PostResult]):
    def _run(self, shared: SharedData) -> _PostResult: ...

class AsyncNode(Node[_PrepResult, _ExecResult, _PostResult]):
    async def prep_async(self, shared: SharedData) -> _PrepResult: ...
    async def exec_async(self, prep_res: _PrepResult) -> _ExecResult: ...
    async def exec_fallback_async(
        self, prep_res: _PrepResult, exc: Exception
    ) -> _ExecResult: ...
    async def post_async(
        self, shared: SharedData, prep_res: _PrepResult, exec_res: _ExecResult
    ) -> _PostResult: ...
    async def _exec(self, prep_res: _PrepResult) -> _ExecResult: ...
    async def run_async(self, shared: SharedData) -> _PostResult: ...
    async def _run_async(self, shared: SharedData) -> _PostResult: ...
    def _run(self, shared: SharedData) -> _PostResult: ...

class AsyncBatchNode(
    AsyncNode[list[_PrepResult] | None, list[_ExecResult], _PostResult],
    BatchNode[list[_PrepResult] | None, list[_ExecResult], _PostResult],
):
    async def _exec(self, items: list[_PrepResult] | None) -> list[_ExecResult]: ...

class AsyncParallelBatchNode(
    AsyncNode[list[_PrepResult] | None, list[_ExecResult], _PostResult],
    BatchNode[list[_PrepResult] | None, list[_ExecResult], _PostResult],
):
    async def _exec(self, items: list[_PrepResult] | None) -> list[_ExecResult]: ...

class AsyncFlow(
    Flow[_PrepResult, Any, _PostResult], AsyncNode[_PrepResult, Any, _PostResult]
):
    async def _orch_async(
        self, shared: SharedData, params: Params | None = None
    ) -> Any: ...
    async def _run_async(self, shared: SharedData) -> _PostResult: ...
    async def post_async(
        self, shared: SharedData, prep_res: _PrepResult, exec_res: Any
    ) -> _PostResult: ...

class AsyncBatchFlow(
    AsyncFlow[list[Params] | None, Any, _PostResult],
    BatchFlow[list[Params] | None, Any, _PostResult],
):
    async def _run_async(self, shared: SharedData) -> _PostResult: ...

class AsyncParallelBatchFlow(
    AsyncFlow[list[Params] | None, Any, _PostResult],
    BatchFlow[list[Params] | None, Any, _PostResult],
):
    async def _run_async(self, shared: SharedData) -> _PostResult: ...

class LLMClient:
    provider: str
    model: str
    base_url: str
    api_key: str

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        base_url: str | None = None,
    ) -> None: ...
    def chat_complete(
        self,
        messages: Messages,
        schema: Any = None,
        retries: int = 3,
    ) -> Any: ...
    def ask(self, prompt: str, schema: Any = None, retries: int = 3) -> Any: ...
    def stream(self, prompt: str) -> Iterator[str]: ...

class RAGNode(Node[str, list[str], str]):
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
    ) -> None: ...
    def add_source(self, source: str) -> "RAGNode": ...
    def prep(self, shared: SharedData) -> str: ...
    def exec(self, query: str) -> list[str]: ...
    def post(self, shared: SharedData, query: str, chunks: list[str]) -> str: ...
