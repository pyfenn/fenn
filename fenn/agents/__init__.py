import asyncio
import copy
import time
import warnings

_TERMINAL = object()  # sentinel for explicit terminal transitions in Flow.connect


class BaseNode:
    def __init__(self):
        self.params, self.successors = {}, {}

    def set_params(self, params):
        self.params = params

    def prep(self, shared):
        """Prepare data for execution."""
        return None

    def exec(self, prep_res):
        """Execute the node logic."""
        return None

    def post(self, shared, prep_res, exec_res):
        """Post-process execution results."""
        return None

    def _exec(self, prep_res):
        return self.exec(prep_res)

    def _run(self, shared):
        p = self.prep(shared)
        e = self._exec(p)
        return self.post(shared, p, e)

    def run(self, shared):
        if self.successors:
            warnings.warn("Node won't run successors. Use Flow.")
        return self._run(shared)


class Node(BaseNode):
    """
    A retryable unit of work in a Flow.

    Wraps the prep/exec/post lifecycle from BaseNode with automatic
    retry support around the exec step. If all retry attempts are
    exhausted, execution falls back to exec_fallback instead of
    propagating the exception.

    Parameters
    ----------
    max_retries : int
        Maximum number of attempts to run exec before giving up and
        calling exec_fallback. Default: 1 (no retries).
    wait : int or float
        Seconds to sleep between failed attempts. Default: 0.
    """

    def __init__(self, max_retries=1, wait=0):
        """
        Initialize the node's retry configuration.

        Parameters
        ----------
        max_retries : int
            Maximum number of attempts to run exec before giving up.
            Default: 1.
        wait : int or float
            Seconds to sleep between failed attempts. Default: 0.
        """
        super().__init__()
        self.max_retries, self.wait = max_retries, wait

    def exec_fallback(self, prep_res, exc):
        """
        Handle the final exec failure after all retries are exhausted.

        Default behavior is to re-raise the exception. Override this
        to return a fallback result instead of failing.

        Parameters
        ----------
        prep_res : Any
            The result returned by prep, passed through to exec.
        exc : Exception
            The exception raised by the final failed exec attempt.

        Returns
        -------
        Any
            Fallback result to use in place of a successful exec call.

        Raises
        ------
        Exception
            Re-raises `exc` by default.
        """
        raise exc

    def _exec(self, prep_res):
        """
        Run exec with retry handling.

        Attempts to call exec up to max_retries times, sleeping `wait`
        seconds between failed attempts. On the final failed attempt,
        delegates to exec_fallback instead of raising.

        Parameters
        ----------
        prep_res : Any
            The result returned by prep, passed through to exec.

        Returns
        -------
        Any
            The result of a successful exec call, or the result of
            exec_fallback if all retries are exhausted.
        """
        for self.cur_retry in range(self.max_retries):
            try:
                return self.exec(prep_res)
            except Exception as e:
                if self.cur_retry == self.max_retries - 1:
                    return self.exec_fallback(prep_res, e)
                if self.wait > 0:
                    time.sleep(self.wait)


class BatchNode(Node):
    def _exec(self, items):
        return [super(BatchNode, self)._exec(i) for i in (items or [])]


class Flow(BaseNode):
    """
    A directed-graph orchestrator for chaining Node executions.

    Flow manages the traversal of a directed graph of nodes. Each node
    returns an action string after execution, and Flow uses that action
    to look up the next node via the successor mappings created with
    connect(). The graph can branch, terminate explicitly, or warn on
    missing transitions.

    Parameters
    ----------
    start : Node or None
        The first node to execute when the flow is run. Can be set
        later via start().
    """

    def __init__(self, start=None):
        """
        Initialize the flow with an optional start node.

        Parameters
        ----------
        start : Node or None
            The first node to execute, or None if start() will be
            called later.
        """
        super().__init__()
        self.start_node = start

    def start(self, start):
        """
        Set the start node for the flow.

        Parameters
        ----------
        start : Node
            The node that will be executed first.

        Returns
        -------
        Node
            The start node, enabling method chaining.
        """
        self.start_node = start
        return start

    def connect(self, src, dst, action="default"):
        """
        Wire a transition from *src* to *dst* triggered by *action*.

        When *src* returns *action* from its lifecycle methods, the flow
        will move to *dst*. Passing ``None`` for *dst* marks the path
        as an explicit terminal transition.

        Parameters
        ----------
        src : Node
            The source node whose return value selects the destination.
        dst : Node or None
            The next node to execute, or None to terminate the flow.
        action : str
            The action string returned by the source node that triggers
            this transition. Default: ``"default"``.

        Returns
        -------
        Flow
            self, enabling method chaining.
        """
        if action in src.successors:
            warnings.warn(f"Overwriting successor for action '{action}'")
        src.successors[action] = _TERMINAL if dst is None else dst
        return self

    def get_next_node(self, curr, action):
        """
        Resolve the next node given the current node and action.

        Parameters
        ----------
        curr : Node
            The currently executing node.
        action : str or None
            The action returned by the current node's lifecycle.

        Returns
        -------
        Node or None
            The next node to execute, or None if the flow should
            terminate.
        """
        nxt = curr.successors.get(action or "default")
        if nxt is _TERMINAL:
            return None
        if nxt is None and curr.successors:
            warnings.warn(f"Flow ends: '{action}' not found in {list(curr.successors)}")
        return nxt

    def _orch(self, shared, params=None):
        curr, p, last_action = (
            copy.copy(self.start_node),
            (params or {**self.params}),
            None,
        )
        while curr:
            curr.set_params(p)
            last_action = curr._run(shared)
            curr = copy.copy(self.get_next_node(curr, last_action))
        return last_action

    def _run(self, shared):
        p = self.prep(shared)
        o = self._orch(shared)
        return self.post(shared, p, o)

    def post(self, shared, prep_res, exec_res):
        """
        Post-process execution results.

        Returns the result of the final node's execution unchanged.
        Override this method when subclasses need to inspect or
        transform the flow-level result.

        Parameters
        ----------
        shared : dict
            Mutable state shared across all nodes in the flow.
        prep_res : Any
            The value returned by prep().
        exec_res : Any
            The action string returned by the final node's lifecycle.

        Returns
        -------
        Any
            The final action from the last executed node.
        """
        return exec_res


class BatchFlow(Flow):
    def _run(self, shared):
        pr = self.prep(shared) or []
        for bp in pr:
            self._orch(shared, {**self.params, **bp})
        return self.post(shared, pr, None)


class AsyncNode(Node):
    async def prep_async(self, shared):
        """Prepare data asynchronously."""
        return None

    async def exec_async(self, prep_res):
        """Execute logic asynchronously."""
        return None

    async def exec_fallback_async(self, prep_res, exc):
        raise exc

    async def post_async(self, shared, prep_res, exec_res):
        """Post-process results asynchronously."""
        return None

    async def _exec(self, prep_res):
        for self.cur_retry in range(self.max_retries):
            try:
                return await self.exec_async(prep_res)
            except Exception as e:
                if self.cur_retry == self.max_retries - 1:
                    return await self.exec_fallback_async(prep_res, e)
                if self.wait > 0:
                    await asyncio.sleep(self.wait)

    async def run_async(self, shared):
        if self.successors:
            warnings.warn("Node won't run successors. Use AsyncFlow.")
        return await self._run_async(shared)

    async def _run_async(self, shared):
        p = await self.prep_async(shared)
        e = await self._exec(p)
        return await self.post_async(shared, p, e)

    def _run(self, shared):
        raise RuntimeError("Use run_async.")


class AsyncBatchNode(AsyncNode, BatchNode):
    async def _exec(self, items):
        return [await super(AsyncBatchNode, self)._exec(i) for i in items]


class AsyncParallelBatchNode(AsyncNode, BatchNode):
    async def _exec(self, items):
        return await asyncio.gather(
            *(super(AsyncParallelBatchNode, self)._exec(i) for i in items)
        )


class AsyncFlow(Flow, AsyncNode):
    async def _orch_async(self, shared, params=None):
        curr, p, last_action = (
            copy.copy(self.start_node),
            (params or {**self.params}),
            None,
        )
        while curr:
            curr.set_params(p)
            last_action = (
                await curr._run_async(shared)
                if isinstance(curr, AsyncNode)
                else curr._run(shared)
            )
            curr = copy.copy(self.get_next_node(curr, last_action))
        return last_action

    async def _run_async(self, shared):
        p = await self.prep_async(shared)
        o = await self._orch_async(shared)
        return await self.post_async(shared, p, o)

    async def post_async(self, shared, prep_res, exec_res):
        return exec_res


class AsyncBatchFlow(AsyncFlow, BatchFlow):
    async def _run_async(self, shared):
        pr = await self.prep_async(shared) or []
        for bp in pr:
            await self._orch_async(shared, {**self.params, **bp})
        return await self.post_async(shared, pr, None)


class AsyncParallelBatchFlow(AsyncFlow, BatchFlow):
    async def _run_async(self, shared):
        pr = await self.prep_async(shared) or []
        await asyncio.gather(
            *(self._orch_async(shared, {**self.params, **bp}) for bp in pr)
        )
        return await self.post_async(shared, pr, None)


from .llm import LLMClient  # noqa: E402 - avoid circular import with .llm
from .rag import RAGNode  # noqa: E402 - avoid circular import with .rag

__all__ = [
    "BaseNode",
    "Node",
    "BatchNode",
    "Flow",
    "BatchFlow",
    "AsyncNode",
    "AsyncBatchNode",
    "AsyncParallelBatchNode",
    "AsyncFlow",
    "AsyncBatchFlow",
    "AsyncParallelBatchFlow",
    "LLMClient",
    "RAGNode",
]
