from fenn.agents import Flow
from fenn.agents.node import ThinkNode, ActNode, ObserveNode
import yaml

class Agent:
    def __init__(self, config, llm):
        self.llm = llm
        with open(config) as f:
            self.config = yaml.safe_load(f)

        think = ThinkNode()
        act = ActNode()
        observe = ObserveNode()

        think - "act"     >> act
        think - "done"    >> None
        act   - "observe" >> observe
        observe - "think" >> think
        observe - "done"  >> None

        self.flow = Flow(start=think)

    def run(self, user_input):
        shared = {
            "llm": self.llm,
            "messages": [
                {"role": "system", "content": self.config["agent"]["system_prompt"]},
                {"role": "user",   "content": user_input}
            ],
            "iterations": 0,
            "max_iterations": self.config["agent"]["max_iterations"],
            "last_thought": None,
            "last_observation": None
        }

        self.flow.run(shared)
        return shared["messages"][-1]["content"]