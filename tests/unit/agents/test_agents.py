"""Tests for fenn/agents/agent.py"""

from unittest.mock import MagicMock, mock_open, patch

import pytest

FAKE_CONFIG = """
agent:
  system_prompt: "You are a helpful assistant."
  max_iterations: 5
"""


def _make_agent():
    """Construct an Agent with all heavy dependencies mocked out."""
    mock_llm = MagicMock()

    with patch(
        "fenn.agents.agent.yaml.safe_load",
        return_value={
            "agent": {"system_prompt": "You are helpful.", "max_iterations": 3}
        },
    ):
        with patch("builtins.open", mock_open(read_data=FAKE_CONFIG)):
            with (
                patch("fenn.agents.agent.ThinkNode") as MockThink,
                patch("fenn.agents.agent.ActNode") as MockAct,
                patch("fenn.agents.agent.ObserveNode") as MockObserve,
                patch("fenn.agents.agent.Flow") as MockFlow,
                patch("fenn.agents.agent.get_tool_schema", return_value=[]),
            ):
                # Make node - operator return the node itself so chaining doesn't error
                for MockNode in (MockThink, MockAct, MockObserve):
                    instance = MockNode.return_value
                    instance.__sub__ = MagicMock(return_value=instance)
                    instance.__rshift__ = MagicMock(return_value=instance)

                from fenn.agents.agent import Agent

                agent = Agent(config="fake_config.yaml", llm=mock_llm)

    return agent, mock_llm, MockFlow


# ── __init__ ───────────────────────────────────────────────────────────────────


def test_agent_stores_llm():
    agent, mock_llm, _ = _make_agent()
    assert agent.llm is mock_llm


def test_agent_loads_config():
    agent, _, _ = _make_agent()
    assert agent.config["agent"]["system_prompt"] == "You are helpful."
    assert agent.config["agent"]["max_iterations"] == 3


def test_agent_creates_flow():
    agent, _, MockFlow = _make_agent()
    assert agent.flow is MockFlow.return_value


def test_agent_config_file_not_found():
    from fenn.agents.agent import Agent

    with pytest.raises((FileNotFoundError, OSError)):
        Agent(config="/nonexistent/path/config.yaml", llm=MagicMock())


# ── run ────────────────────────────────────────────────────────────────────────


def test_run_calls_flow_run():
    agent, _, _ = _make_agent()
    agent.flow.run = MagicMock()

    # Simulate flow.run appending a response message
    def fake_run(shared):
        shared["messages"].append({"role": "assistant", "content": "Done!"})

    agent.flow.run.side_effect = fake_run

    with patch("fenn.agents.agent.get_tool_schema", return_value=[]):
        result = agent.run("Hello")

    agent.flow.run.assert_called_once()
    assert result == "Done!"


def test_run_returns_last_message_content():
    agent, _, _ = _make_agent()

    def fake_run(shared):
        shared["messages"].append({"role": "assistant", "content": "Final answer"})

    agent.flow.run.side_effect = fake_run

    result = agent.run("What is 2+2?")
    assert result == "Final answer"


def test_run_builds_correct_shared_state():
    agent, mock_llm, _ = _make_agent()
    captured = {}

    def fake_run(shared):
        captured.update(shared)
        shared["messages"].append({"role": "assistant", "content": "ok"})

    agent.flow.run.side_effect = fake_run

    with patch("fenn.agents.agent.get_tool_schema", return_value=["tool1"]):
        agent.run("Test input")

    assert captured["llm"] is mock_llm
    assert captured["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert captured["messages"][1] == {"role": "user", "content": "Test input"}
    assert captured["tools"] == ["tool1"]
    assert captured["iterations"] == 0
    assert captured["max_iterations"] == 3
    assert captured["last_thought"] is None
    assert captured["last_observation"] is None


def test_run_with_different_user_inputs():
    agent, _, _ = _make_agent()

    for user_input in ["Hello", "What time is it?", ""]:

        def fake_run(shared):
            shared["messages"].append({"role": "assistant", "content": "response"})

        agent.flow.run.side_effect = fake_run
        result = agent.run(user_input)
        assert result == "response"
