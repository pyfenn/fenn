TOOLS = {
    "search": lambda query: f"Search result for: {query}",
    "calculator": lambda expr: str(eval(expr))
}

# This is what you put in the LLM prompt so it knows what tools exist
TOOL_DESCRIPTIONS = """
- search(query): Search the web
- calculator(expr): Evaluate a math expression
"""