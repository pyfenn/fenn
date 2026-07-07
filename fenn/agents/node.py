from typing import Any

from fenn.agents import Node
from fenn.agents.tools import execute_tool


class ThinkNode(Node):
    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:
        return {"llm": shared["llm"], "messages": shared["messages"]}

    def exec(self, prep_res: dict[str, Any]) -> str:
        llm = prep_res["llm"]
        response = llm.chat_complete(prep_res["messages"])
        return response

    def post(self, shared: dict[str, Any], prep_res: dict[str, Any], exec_res: str) -> str:
        shared["last_thought"] = exec_res
        shared["messages"].append({"role": "assistant", "content": exec_res})
        if "Action:" in exec_res:
            return "act"
        return "done"


class ActNode(Node):
    def prep(self, shared: dict[str, Any]) -> str:
        return shared["last_thought"]

    def exec(self, thought: str) -> str:
        if "Action:" in thought:
            action_line = [
                line for line in thought.split("\n") if line.startswith("Action:")
            ][0]
            tool_call = action_line.replace("Action:", "").strip()
        else:
            tool_call = thought.strip()

        tool_name = tool_call.split("(")[0]
        args_str = tool_call.split("(")[1].rstrip(")")
        tool_args = [arg.strip() for arg in args_str.split(",")]

        try:
            result = execute_tool(tool_name, *tool_args)
        except Exception as e:
            result = f"Error: {e}"
        return str(result)

    def post(self, shared: dict[str, Any], prep_res: str, exec_res: str) -> str:
        shared["last_observation"] = exec_res
        return "observe"


class ObserveNode(Node):
    def prep(self, shared: dict[str, Any]) -> str:
        return shared["last_observation"]

    def exec(self, observation: str) -> str:
        return observation

    def post(self, shared: dict[str, Any], prep_res: str, exec_res: str) -> str:
        shared["messages"].append(
            {"role": "user", "content": f"Observation: {exec_res}"}
        )
        shared["iterations"] += 1
        if shared["iterations"] >= shared["max_iterations"]:
            return "done"
        return "think"