import inspect
from functools import wraps
from typing import Any

TOOLS_REGISTRY: dict[str, dict[str, Any]] = {}


def tool(func):
    """
    A decorator that registers an executable function as a tool
    """

    @wraps(func)
    def decorator(*args, **kwargs):
        return func(*args, **kwargs)

    tool_name = func.__name__
    tools_description = inspect.getdoc(func) or "No description provided"

    TOOLS_REGISTRY[tool_name] = {
        "schema": {"name": tool_name, "description": tools_description},
        "execute": func,
    }
    return decorator


def get_tool_schema():
    return [info["schema"] for info in TOOLS_REGISTRY.values()]


def execute_tool(name: str, *args, **kwargs):
    if name not in TOOLS_REGISTRY:
        raise ValueError(f"Tool '{name}' is not registered.")
    return TOOLS_REGISTRY[name]["execute"](*args, **kwargs)
