import pytest

from fenn.agents.tools import TOOLS_REGISTRY, execute_tool, get_tool_schema, tool


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure each test starts with a clean tools registry."""
    TOOLS_REGISTRY.clear()
    yield
    TOOLS_REGISTRY.clear()


class TestToolDecorator:
    def test_registers_function_in_registry(self):
        @tool
        def my_tool(x):
            """Does something."""
            return x * 2

        assert "my_tool" in TOOLS_REGISTRY

    def test_registry_schema_contains_name_and_description(self):
        @tool
        def search(query):
            """Search the web for a query."""
            return query

        entry = TOOLS_REGISTRY["search"]
        assert entry["schema"]["name"] == "search"
        assert entry["schema"]["description"] == "Search the web for a query."

    def test_missing_docstring_uses_default_description(self):
        @tool
        def no_doc(x):
            return x

        entry = TOOLS_REGISTRY["no_doc"]
        assert entry["schema"]["description"] == "No description provided"

    def test_execute_key_stores_original_function(self):
        @tool
        def add(a, b):
            """Add two numbers."""
            return a + b

        assert TOOLS_REGISTRY["add"]["execute"](2, 3) == 5

    def test_decorated_function_still_callable(self):
        @tool
        def double(x):
            """Double a number."""
            return x * 2

        assert double(5) == 10

    def test_decorator_preserves_name_and_doc(self):
        @tool
        def greet(name):
            """Greet someone."""
            return f"Hello, {name}"

        assert greet.__name__ == "greet"
        assert greet.__doc__ == "Greet someone."

    def test_multiple_tools_registered_independently(self):
        @tool
        def tool_a():
            """Tool A."""
            return "a"

        @tool
        def tool_b():
            """Tool B."""
            return "b"

        assert set(TOOLS_REGISTRY.keys()) == {"tool_a", "tool_b"}

    def test_decorator_passes_args_and_kwargs(self):
        @tool
        def combine(a, b, sep="-"):
            """Combine two values with a separator."""
            return f"{a}{sep}{b}"

        assert combine("x", "y") == "x-y"
        assert combine("x", "y", sep="_") == "x_y"


class TestGetToolSchema:
    def test_empty_registry_returns_empty_list(self):
        assert get_tool_schema() == []

    def test_returns_list_of_schemas(self):
        @tool
        def search(query):
            """Search something."""
            return query

        @tool
        def calc(expr):
            """Evaluate an expression."""
            return expr

        schemas = get_tool_schema()
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert names == {"search", "calc"}

    def test_schema_does_not_include_execute_key(self):
        @tool
        def my_tool():
            """A tool."""
            return None

        schemas = get_tool_schema()
        assert "execute" not in schemas[0]


class TestExecuteTool:
    def test_executes_registered_tool(self):
        @tool
        def add(a, b):
            """Add two numbers."""
            return a + b

        result = execute_tool("add", 2, 3)
        assert result == 5

    def test_executes_with_kwargs(self):
        @tool
        def greet(name, greeting="Hello"):
            """Greet someone."""
            return f"{greeting}, {name}!"

        result = execute_tool("greet", "World", greeting="Hi")
        assert result == "Hi, World!"

    def test_unregistered_tool_raises_value_error(self):
        with pytest.raises(ValueError, match="not registered"):
            execute_tool("nonexistent_tool")

    def test_error_message_includes_tool_name(self):
        with pytest.raises(ValueError, match="missing_tool"):
            execute_tool("missing_tool", "arg")

    def test_executes_tool_with_no_args(self):
        @tool
        def get_time():
            """Get the current time."""
            return "12:00"

        assert execute_tool("get_time") == "12:00"

    def test_execute_uses_original_function_not_wrapper(self):
        """execute_tool should call the underlying func directly."""
        call_log = []

        @tool
        def logged(x):
            """Logs calls."""
            call_log.append(x)
            return x

        execute_tool("logged", "value")
        assert call_log == ["value"]
