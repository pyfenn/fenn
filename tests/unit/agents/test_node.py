"""Tests for fenn/agents/node.py"""

from unittest.mock import MagicMock, patch

from fenn.agents.node import ActNode, ObserveNode, ThinkNode

# ── Helpers ────────────────────────────────────────────────────────────────────


def _shared(
    messages=None,
    last_thought=None,
    last_observation=None,
    iterations=0,
    max_iterations=5,
):
    mock_llm = MagicMock()
    return {
        "llm": mock_llm,
        "messages": messages or [{"role": "user", "content": "hello"}],
        "last_thought": last_thought,
        "last_observation": last_observation,
        "iterations": iterations,
        "max_iterations": max_iterations,
    }


# ── ThinkNode ──────────────────────────────────────────────────────────────────


class TestThinkNode:
    def test_prep_returns_llm_and_messages(self):
        node = ThinkNode()
        shared = _shared()
        result = node.prep(shared)
        assert result["llm"] is shared["llm"]
        assert result["messages"] is shared["messages"]

    def test_exec_calls_chat_complete(self):
        node = ThinkNode()
        mock_llm = MagicMock()
        mock_llm.chat_complete.return_value = "I need to think."
        prep_res = {"llm": mock_llm, "messages": [{"role": "user", "content": "hi"}]}
        result = node.exec(prep_res)
        mock_llm.chat_complete.assert_called_once_with(prep_res["messages"])
        assert result == "I need to think."

    def test_post_returns_act_when_action_in_response(self):
        node = ThinkNode()
        shared = _shared()
        action = "Thought: I should search.\nAction: search(query)"
        result = node.post(shared, {}, action)
        assert result == "act"

    def test_post_returns_done_when_no_action(self):
        node = ThinkNode()
        shared = _shared()
        result = node.post(shared, {}, "I have the final answer.")
        assert result == "done"

    def test_post_appends_message(self):
        node = ThinkNode()
        shared = _shared()
        node.post(shared, {}, "Some thought.")
        assert shared["messages"][-1] == {
            "role": "assistant",
            "content": "Some thought.",
        }

    def test_post_stores_last_thought(self):
        node = ThinkNode()
        shared = _shared()
        node.post(shared, {}, "My thought")
        assert shared["last_thought"] == "My thought"


# ── ActNode ────────────────────────────────────────────────────────────────────


class TestActNode:
    def test_prep_returns_last_thought(self):
        node = ActNode()
        shared = _shared(last_thought="Action: search(hello)")
        assert node.prep(shared) == "Action: search(hello)"

    def test_exec_parses_action_line_and_calls_tool(self):
        node = ActNode()
        thought = "Thought: search\nAction: search(python)"
        with patch("fenn.agents.node.execute_tool", return_value="result") as mock_tool:
            result = node.exec(thought)
        mock_tool.assert_called_once_with("search", "python")
        assert result == "result"

    def test_exec_without_action_prefix(self):
        node = ActNode()
        thought = "lookup(wikipedia)"
        with patch(
            "fenn.agents.node.execute_tool", return_value="wiki result"
        ) as mock_tool:
            result = node.exec(thought)
        mock_tool.assert_called_once_with("lookup", "wikipedia")
        assert result == "wiki result"

    def test_exec_tool_exception_returns_error_string(self):
        node = ActNode()
        thought = "Action: bad_tool(arg)"
        with patch(
            "fenn.agents.node.execute_tool", side_effect=ValueError("not found")
        ):
            result = node.exec(thought)
        assert result == "Error: not found"

    def test_exec_multi_arg_tool(self):
        node = ActNode()
        thought = "Action: calculate(1, 2)"
        with patch("fenn.agents.node.execute_tool", return_value="3") as mock_tool:
            node.exec(thought)
        mock_tool.assert_called_once_with("calculate", "1", "2")

    def test_post_stores_observation_and_returns_observe(self):
        node = ActNode()
        shared = _shared()
        action = node.post(shared, {}, "tool output")
        assert shared["last_observation"] == "tool output"
        assert action == "observe"


# ── ObserveNode ────────────────────────────────────────────────────────────────


class TestObserveNode:
    def test_prep_returns_last_observation(self):
        node = ObserveNode()
        shared = _shared(last_observation="some result")
        assert node.prep(shared) == "some result"

    def test_exec_returns_observation_unchanged(self):
        node = ObserveNode()
        assert node.exec("anything") == "anything"

    def test_post_appends_observation_message(self):
        node = ObserveNode()
        shared = _shared()
        node.post(shared, {}, "search result")
        assert shared["messages"][-1] == {
            "role": "user",
            "content": "Observation: search result",
        }

    def test_post_increments_iterations(self):
        node = ObserveNode()
        shared = _shared(iterations=2)
        node.post(shared, {}, "obs")
        assert shared["iterations"] == 3

    def test_post_returns_think_when_below_max(self):
        node = ObserveNode()
        shared = _shared(iterations=0, max_iterations=5)
        result = node.post(shared, {}, "obs")
        assert result == "think"

    def test_post_returns_done_when_at_max_iterations(self):
        node = ObserveNode()
        shared = _shared(iterations=4, max_iterations=5)
        result = node.post(shared, {}, "obs")
        assert result == "done"

    def test_post_returns_done_when_exceeds_max(self):
        node = ObserveNode()
        shared = _shared(iterations=10, max_iterations=5)
        result = node.post(shared, {}, "obs")
        assert result == "done"
