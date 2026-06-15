import asyncio
import warnings

import pytest

from fenn.agents import (
    AsyncBatchFlow,
    AsyncBatchNode,
    AsyncFlow,
    AsyncNode,
    AsyncParallelBatchFlow,
    AsyncParallelBatchNode,
    BaseNode,
    BatchFlow,
    BatchNode,
    Flow,
    Node,
)


class TestBaseNode:
    def test_init_sets_empty_params_and_successors(self):
        node = BaseNode()
        assert node.params == {}
        assert node.successors == {}

    def test_set_params(self):
        node = BaseNode()
        node.set_params({"a": 1})
        assert node.params == {"a": 1}

    def test_default_prep_exec_post_return_none(self):
        node = BaseNode()
        assert node.prep({}) is None
        assert node.exec(None) is None
        assert node.post({}, None, None) is None

    def test_run_calls_prep_exec_post_in_order(self):
        calls = []

        class Tracked(BaseNode):
            def prep(self, shared):
                calls.append("prep")
                return "prepped"

            def exec(self, prep_res):
                calls.append(("exec", prep_res))
                return "executed"

            def post(self, shared, prep_res, exec_res):
                calls.append(("post", prep_res, exec_res))
                return "done"

        node = Tracked()
        result = node.run({})
        assert calls == ["prep", ("exec", "prepped"), ("post", "prepped", "executed")]
        assert result == "done"

    def test_run_warns_when_successors_present(self):
        node = BaseNode()
        node.successors["default"] = BaseNode()
        with pytest.warns(UserWarning, match="won't run successors"):
            node.run({})

    def test_run_no_warning_without_successors(self):
        node = BaseNode()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            node.run({})  # should not raise


class TestNode:
    def test_default_max_retries_and_wait(self):
        node = Node()
        assert node.max_retries == 1
        assert node.wait == 0

    def test_exec_succeeds_first_try(self):
        class Succeeds(Node):
            def exec(self, prep_res):
                return "ok"

        node = Succeeds()
        assert node._exec(None) == "ok"

    def test_exec_fallback_raises_by_default(self):
        class AlwaysFails(Node):
            def exec(self, prep_res):
                raise ValueError("boom")

        node = AlwaysFails(max_retries=1)
        with pytest.raises(ValueError, match="boom"):
            node._exec(None)

    def test_retries_then_succeeds(self):
        attempts = {"count": 0}

        class FlakyThenWorks(Node):
            def exec(self, prep_res):
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise RuntimeError("transient")
                return "success"

        node = FlakyThenWorks(max_retries=5)
        result = node._exec(None)
        assert result == "success"
        assert attempts["count"] == 3

    def test_exec_fallback_called_after_exhausting_retries(self):
        class FailsWithFallback(Node):
            def exec(self, prep_res):
                raise ValueError("always fails")

            def exec_fallback(self, prep_res, exc):
                return f"fallback: {exc}"

        node = FailsWithFallback(max_retries=2)
        result = node._exec(None)
        assert result == "fallback: always fails"

    def test_wait_triggers_sleep_between_retries(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("fenn.agents.time.sleep", lambda s: sleeps.append(s))

        class AlwaysFails(Node):
            def exec(self, prep_res):
                raise ValueError("fail")

            def exec_fallback(self, prep_res, exc):
                return "fallback"

        node = AlwaysFails(max_retries=3, wait=2)
        node._exec(None)
        assert sleeps == [2, 2]

    def test_cur_retry_tracks_attempt_number(self):
        attempts = []

        class TrackRetry(Node):
            def exec(self, prep_res):
                attempts.append(self.cur_retry)
                raise ValueError("fail")

            def exec_fallback(self, prep_res, exc):
                return "fallback"

        node = TrackRetry(max_retries=3)
        node._exec(None)
        assert attempts == [0, 1, 2]

    def test_full_run_pipeline(self):
        class Adder(Node):
            def prep(self, shared):
                return shared["value"]

            def exec(self, prep_res):
                return prep_res + 1

            def post(self, shared, prep_res, exec_res):
                shared["result"] = exec_res
                return "done"

        node = Adder()
        shared = {"value": 5}
        action = node._run(shared)
        assert shared["result"] == 6
        assert action == "done"


class TestBatchNode:
    def test_exec_processes_each_item(self):
        class Doubler(BatchNode):
            def exec(self, item):
                return item * 2

        node = Doubler()
        result = node._exec([1, 2, 3])
        assert result == [2, 4, 6]

    def test_exec_with_none_returns_empty_list(self):
        class Doubler(BatchNode):
            def exec(self, item):
                return item * 2

        node = Doubler()
        assert node._exec(None) == []

    def test_exec_with_empty_list(self):
        class Doubler(BatchNode):
            def exec(self, item):
                return item * 2

        node = Doubler()
        assert node._exec([]) == []

    def test_batch_node_retries_per_item(self):
        attempts = {"a": 0, "b": 0}

        class FlakyPerItem(BatchNode):
            def exec(self, item):
                attempts[item] += 1
                if attempts[item] < 2:
                    raise RuntimeError("fail once")
                return item

            def exec_fallback(self, prep_res, exc):
                return "fallback"

        node = FlakyPerItem(max_retries=3)
        result = node._exec(["a", "b"])
        assert result == ["a", "b"]
        assert attempts == {"a": 2, "b": 2}


class _Echo(BaseNode):
    """Simple node: records its label and returns a fixed next action.

    Uses instance attributes (not self.params) because Flow._orch()
    overwrites node.params on every step.
    """

    def __init__(self, label="node", next_action="default"):
        super().__init__()
        self.label = label
        self.next_action = next_action

    def post(self, shared, prep_res, exec_res):
        shared.setdefault("visited", []).append(self.label)
        return self.next_action


class TestFlow:
    def test_start_sets_start_node_and_returns_it(self):
        flow = Flow()
        node = BaseNode()
        result = flow.start(node)
        assert flow.start_node is node
        assert result is node

    def test_connect_returns_self(self):
        flow = Flow()
        a, b = BaseNode(), BaseNode()
        result = flow.connect(a, b)
        assert result is flow

    def test_connect_sets_successor(self):
        flow = Flow()
        a, b = BaseNode(), BaseNode()
        flow.connect(a, b, action="next")
        assert a.successors["next"] is b

    def test_connect_default_action(self):
        flow = Flow()
        a, b = BaseNode(), BaseNode()
        flow.connect(a, b)
        assert a.successors["default"] is b

    def test_connect_none_dest_sets_terminal(self):
        flow = Flow()
        a = BaseNode()
        flow.connect(a, None, action="done")
        assert flow.get_next_node(a, "done") is None

    def test_connect_overwrite_warns(self):
        flow = Flow()
        a, b, c = BaseNode(), BaseNode(), BaseNode()
        flow.connect(a, b, action="x")
        with pytest.warns(UserWarning, match="Overwriting successor"):
            flow.connect(a, c, action="x")

    def test_get_next_node_returns_successor(self):
        flow = Flow()
        a, b = BaseNode(), BaseNode()
        flow.connect(a, b, action="go")
        assert flow.get_next_node(a, "go") is b

    def test_get_next_node_defaults_to_default_action(self):
        flow = Flow()
        a, b = BaseNode(), BaseNode()
        flow.connect(a, b)
        assert flow.get_next_node(a, None) is b

    def test_get_next_node_unknown_action_warns(self):
        flow = Flow()
        a, b = BaseNode(), BaseNode()
        flow.connect(a, b, action="default")
        with pytest.warns(UserWarning, match="Flow ends"):
            result = flow.get_next_node(a, "unknown")
        assert result is None

    def test_get_next_node_no_successors_no_warning(self):
        flow = Flow()
        a = BaseNode()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = flow.get_next_node(a, "anything")
        assert result is None

    def test_orch_traverses_chain(self):
        a = _Echo(label="a")
        b = _Echo(label="b")
        c = _Echo(label="c")
        flow = Flow(start=a)
        flow.connect(a, b).connect(b, c).connect(c, None)

        shared = {}
        flow._orch(shared)
        assert shared["visited"] == ["a", "b", "c"]

    def test_post_returns_exec_res(self):
        flow = Flow()
        assert flow.post({}, "prep", "exec_result") == "exec_result"

    def test_full_flow_run(self):
        a = _Echo(label="a", next_action="next")
        b = _Echo(label="b", next_action="default")

        flow = Flow(start=a)
        flow.connect(a, b, action="next")
        flow.connect(b, None)

        shared = {}
        result = flow._run(shared)
        assert shared["visited"] == ["a", "b"]
        assert result == "default"

    def test_branching_flow(self):
        """Flow takes different paths based on returned action."""

        class Router(BaseNode):
            def post(self, shared, prep_res, exec_res):
                shared.setdefault("visited", []).append("router")
                return shared["route"]

        class Leaf(BaseNode):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def post(self, shared, prep_res, exec_res):
                shared.setdefault("visited", []).append(self.name)
                return "default"

        router = Router()
        left = Leaf("left")
        right = Leaf("right")

        flow = Flow(start=router)
        flow.connect(router, left, action="left")
        flow.connect(router, right, action="right")
        flow.connect(left, None)
        flow.connect(right, None)

        shared = {"route": "right"}
        flow._orch(shared)
        assert shared["visited"] == ["router", "right"]


class TestBatchFlow:
    def test_run_iterates_prep_results(self):
        processed = []

        class Worker(_Echo):
            def post(self, shared, prep_res, exec_res):
                processed.append(self.params.get("item"))
                return "default"

        worker = Worker()
        worker.successors["default"] = (
            None  # no successor, terminal via warning suppressed
        )

        class MyBatchFlow(BatchFlow):
            def prep(self, shared):
                return [{"item": "a"}, {"item": "b"}, {"item": "c"}]

        flow = MyBatchFlow(start=worker)
        # avoid "Flow ends" warning noise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flow._run({})

        assert processed == ["a", "b", "c"]

    def test_run_with_empty_prep_returns_post_result(self):
        class EmptyBatchFlow(BatchFlow):
            def prep(self, shared):
                return None

            def post(self, shared, prep_res, exec_res):
                return ("post", prep_res, exec_res)

        flow = EmptyBatchFlow(start=BaseNode())
        result = flow._run({})
        assert result == ("post", [], None)


class TestAsyncNode:
    def test_sync_run_raises_runtime_error(self):
        node = AsyncNode()
        with pytest.raises(RuntimeError, match="Use run_async"):
            node._run({})

    def test_run_async_calls_pipeline(self):
        calls = []

        class Tracked(AsyncNode):
            async def prep_async(self, shared):
                calls.append("prep")
                return "p"

            async def exec_async(self, prep_res):
                calls.append(("exec", prep_res))
                return "e"

            async def post_async(self, shared, prep_res, exec_res):
                calls.append(("post", prep_res, exec_res))
                return "done"

        node = Tracked()
        result = asyncio.run(node.run_async({}))
        assert calls == ["prep", ("exec", "p"), ("post", "p", "e")]
        assert result == "done"

    def test_run_async_warns_with_successors(self):
        class Simple(AsyncNode):
            async def exec_async(self, prep_res):
                return "e"

        node = Simple()
        node.successors["default"] = AsyncNode()

        async def run_it():
            with pytest.warns(UserWarning, match="won't run successors"):
                await node.run_async({})

        asyncio.run(run_it())

    def test_exec_fallback_async_raises_by_default(self):
        class AlwaysFails(AsyncNode):
            async def exec_async(self, prep_res):
                raise ValueError("async boom")

        node = AlwaysFails(max_retries=1)
        with pytest.raises(ValueError, match="async boom"):
            asyncio.run(node._exec(None))

    def test_exec_fallback_async_called_after_retries(self):
        class FailsWithFallback(AsyncNode):
            async def exec_async(self, prep_res):
                raise ValueError("nope")

            async def exec_fallback_async(self, prep_res, exc):
                return f"fallback: {exc}"

        node = FailsWithFallback(max_retries=2)
        result = asyncio.run(node._exec(None))
        assert result == "fallback: nope"

    def test_async_retries_then_succeeds(self):
        attempts = {"count": 0}

        class FlakyAsync(AsyncNode):
            async def exec_async(self, prep_res):
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise RuntimeError("transient")
                return "success"

        node = FlakyAsync(max_retries=5)
        result = asyncio.run(node._exec(None))
        assert result == "success"
        assert attempts["count"] == 3

    def test_async_wait_triggers_sleep(self, monkeypatch):
        sleeps = []

        async def fake_sleep(s):
            sleeps.append(s)

        monkeypatch.setattr("fenn.agents.asyncio.sleep", fake_sleep)

        class AlwaysFails(AsyncNode):
            async def exec_async(self, prep_res):
                raise ValueError("fail")

            async def exec_fallback_async(self, prep_res, exc):
                return "fallback"

        node = AlwaysFails(max_retries=3, wait=1)
        asyncio.run(node._exec(None))
        assert sleeps == [1, 1]


class TestAsyncBatchNode:
    def test_processes_items_sequentially(self):
        order = []

        class Worker(AsyncBatchNode):
            async def exec_async(self, item):
                order.append(item)
                return item * 2

        node = Worker()
        result = asyncio.run(node._exec([1, 2, 3]))
        assert result == [2, 4, 6]
        assert order == [1, 2, 3]


class TestAsyncParallelBatchNode:
    def test_processes_items_in_parallel(self):
        class Worker(AsyncParallelBatchNode):
            async def exec_async(self, item):
                await asyncio.sleep(0)
                return item * 10

        node = Worker()
        result = asyncio.run(node._exec([1, 2, 3]))
        assert result == [10, 20, 30]


class _AsyncEcho(AsyncNode):
    """Async version of _Echo - uses instance attributes, not self.params."""

    def __init__(self, label="node", next_action="default"):
        super().__init__()
        self.label = label
        self.next_action = next_action

    async def post_async(self, shared, prep_res, exec_res):
        shared.setdefault("visited", []).append(self.label)
        return self.next_action


class TestAsyncFlow:
    def test_orch_async_with_async_nodes(self):
        a = _AsyncEcho(label="a")
        b = _AsyncEcho(label="b")

        flow = AsyncFlow(start=a)
        flow.connect(a, b).connect(b, None)

        shared = {}
        asyncio.run(flow._orch_async(shared))
        assert shared["visited"] == ["a", "b"]

    def test_orch_async_with_mixed_sync_and_async_nodes(self):
        a = _AsyncEcho(label="a")
        b = _Echo(label="b")  # sync node

        flow = AsyncFlow(start=a)
        flow.connect(a, b).connect(b, None)

        shared = {}
        asyncio.run(flow._orch_async(shared))
        assert shared["visited"] == ["a", "b"]

    def test_post_async_default_returns_exec_res(self):
        flow = AsyncFlow()
        result = asyncio.run(flow.post_async({}, "prep", "exec"))
        assert result == "exec"

    def test_run_async_full_pipeline(self):
        a = _AsyncEcho(label="a")

        flow = AsyncFlow(start=a)
        flow.connect(a, None)

        shared = {}
        result = asyncio.run(flow._run_async(shared))
        assert shared["visited"] == ["a"]
        assert result == "default"


class TestAsyncBatchFlow:
    def test_run_async_iterates_prep_results(self):
        processed = []

        class Worker(_AsyncEcho):
            async def post_async(self, shared, prep_res, exec_res):
                processed.append(self.params.get("item"))
                return "default"

        worker = Worker()

        class MyAsyncBatchFlow(AsyncBatchFlow):
            async def prep_async(self, shared):
                return [{"item": "a"}, {"item": "b"}]

        flow = MyAsyncBatchFlow(start=worker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            asyncio.run(flow._run_async({}))

        assert processed == ["a", "b"]

    def test_run_async_with_empty_prep(self):
        class EmptyAsyncBatchFlow(AsyncBatchFlow):
            async def prep_async(self, shared):
                return None

            async def post_async(self, shared, prep_res, exec_res):
                return ("post", prep_res, exec_res)

        flow = EmptyAsyncBatchFlow(start=BaseNode())
        result = asyncio.run(flow._run_async({}))
        assert result == ("post", [], None)


class TestAsyncParallelBatchFlow:
    def test_run_async_processes_concurrently(self):
        processed = []

        class Worker(_AsyncEcho):
            async def post_async(self, shared, prep_res, exec_res):
                await asyncio.sleep(0)
                processed.append(self.params.get("item"))
                return "default"

        worker = Worker()

        class MyParallelBatchFlow(AsyncParallelBatchFlow):
            async def prep_async(self, shared):
                return [{"item": "a"}, {"item": "b"}, {"item": "c"}]

        flow = MyParallelBatchFlow(start=worker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            asyncio.run(flow._run_async({}))

        assert sorted(processed) == ["a", "b", "c"]


class TestModuleExports:
    def test_llmclient_importable(self):
        from fenn.agents import LLMClient

        assert LLMClient is not None

    def test_ragnode_importable(self):
        from fenn.agents import RAGNode

        assert RAGNode is not None

    def test_all_exports_present(self):
        import fenn.agents as agents_module

        for name in agents_module.__all__:
            assert hasattr(agents_module, name), f"{name} missing from module"
