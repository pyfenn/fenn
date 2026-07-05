import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeAlias, TypedDict, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class ToolSchema(TypedDict):
    name: str
    description: str


class ToolEntry(TypedDict):
    schema: ToolSchema
    execute: Callable[..., Any]


ToolRegistry: TypeAlias = dict[str, ToolEntry]

TOOLS_REGISTRY: ToolRegistry = {}


def tool(func: Callable[P, R]) -> Callable[P, R]:
    """
    A decorator that registers an executable function as a tool
    """

    @wraps(func)
    def decorator(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)

    tool_name: str = func.__name__
    tools_description: str = inspect.getdoc(func) or "No description provided"

    TOOLS_REGISTRY[tool_name] = {
        "schema": {"name": tool_name, "description": tools_description},
        "execute": func,
    }
    return decorator


def get_tool_schema() -> list[ToolSchema]:
    return [info["schema"] for info in TOOLS_REGISTRY.values()]


def execute_tool(name: str, *args: Any, **kwargs: Any) -> Any:
    if name not in TOOLS_REGISTRY:
        raise ValueError(f"Tool '{name}' is not registered.")
    return TOOLS_REGISTRY[name]["execute"](*args, **kwargs)
