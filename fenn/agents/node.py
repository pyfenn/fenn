from typing import Any, TypeAlias, cast

from fenn.agents import Node
from fenn.agents.tools import execute_tool

SharedData: TypeAlias = dict[str, Any]
ThinkPrepResult: TypeAlias = dict[str, Any]


def _parse_tool_call(prep_res: str) -> tuple[str, list[str]]:
    """Extract a tool name and arguments from an action line."""
    if "Action:" in prep_res:
        action_line = next(
            line for line in prep_res.split("\n") if line.startswith("Action:")
        )
        tool_call = action_line.replace("Action:", "").strip()
    else:
        tool_call = prep_res.strip()

    tool_name = tool_call.split("(")[0]
    args_str = tool_call.split("(")[1].rstrip(")")
    tool_args = [arg.strip() for arg in args_str.split(",")]
    return tool_name, tool_args


class ThinkNode(Node):
    def prep(self, shared: SharedData) -> ThinkPrepResult:
        return {"llm": shared["llm"], "messages": shared["messages"]}

    def exec(self, prep_res: ThinkPrepResult) -> str:
        llm = prep_res["llm"]
        response = llm.chat_complete(prep_res["messages"])
        return cast(str, response)

    def post(self, shared: SharedData, prep_res: ThinkPrepResult, exec_res: str) -> str:
        shared["last_thought"] = exec_res
        shared["messages"].append({"role": "assistant", "content": exec_res})
        if "Action:" in exec_res:
            return "act"
        return "done"


class ActNode(Node):
    def prep(self, shared: SharedData) -> str:
        return cast(str, shared["last_thought"])

    def exec(self, prep_res: str) -> str:
        tool_name, tool_args = _parse_tool_call(prep_res)

        try:
            result = execute_tool(tool_name, *tool_args)
        except (
            ValueError,
            TypeError,
            RuntimeError,
            KeyError,
            AttributeError,
            OSError,
        ) as e:
            result = f"Error: {e}"
        return cast(str, result)

    def post(self, shared: SharedData, prep_res: str, exec_res: str) -> str:
        shared["last_observation"] = exec_res
        return "observe"


class ObserveNode(Node):
    def prep(self, shared: SharedData) -> str:
        return cast(str, shared["last_observation"])

    def exec(self, prep_res: str) -> str:
        return prep_res

    def post(self, shared: SharedData, prep_res: str, exec_res: str) -> str:
        shared["messages"].append(
            {"role": "user", "content": f"Observation: {exec_res}"}
        )
        shared["iterations"] += 1
        if shared["iterations"] >= shared["max_iterations"]:
            return "done"
        return "think"
