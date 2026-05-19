from fenn.agents import Node
from fenn.agents.tools import TOOLS

class ThinkNode(Node):
    def prep(self, shared):
        return {"llm": shared["llm"], "messages": shared["messages"]}

    def exec(self, prep_res):
        llm = prep_res["llm"]
        response =llm.chat_complete(prep_res["messages"])
        return response
    
    def post(self, shared, prep_res, exec_res):
        shared["last_thought"] = exec_res
        shared["messages"].append({"role": "assistant", "content": exec_res})
        if "Action:" in exec_res:
            return "act"
        return "done"
    
class ActNode(Node):
    def prep(self, shared):
        return shared["last_thought"]
    
    def exec(self, thought):
        line = [l for l in thought.split("\n") if l.startswith("Action:")][0]
        tool_call = line.replace("Action:", "").strip()

        tool_name = tool_call.split("(")[0]
        tool_arg  = tool_call.split("(")[1].rstrip(")")

        result = TOOLS[tool_name](tool_arg)
        return result
    
    def post(self, shared, prep_res, exec_res):
        shared["last_observation"] = exec_res
        return "observe"
    
class ObserveNode(Node):
    def prep(self, shared):
        return shared["last_observation"]
    
    def exec(self, observation):
        return observation
    
    def post(self, shared, prep_res, exec_res):
        shared["messages"].append({
            "role": "user",
            "content": f"Observation: {exec_res}"
        })
        shared["iterations"] += 1
        if shared["iterations"] >= shared["max_iterations"]:
            return "done"
        return "think"