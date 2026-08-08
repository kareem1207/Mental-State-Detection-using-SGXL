"""MCP server exposing the mental-state ML classifier as a tool (stdio transport).

Run standalone for testing: `uv run python mcp_server.py`
Or point an MCP client (e.g. chat_with_tools.py, Claude Desktop, MCP Inspector) at this file.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from ml_pipeline import analyze_mental_state

mcp = MCPServer("brain-analyzer")


@mcp.tool()
def analyze_mental_state_tool(text: str) -> dict:
    """Classify the user's mental/emotional state from a piece of text.

    Returns the predicted state plus supportive tips, yoga suggestions, and a
    short spoken-style message tailored to that state. Call this whenever the
    user's message expresses feelings, mood, or emotional distress and you want
    to ground your reply in an actual assessment rather than guessing.

    Args:
        text: The user's message, in English.
    """
    return analyze_mental_state(text)


if __name__ == "__main__":
    mcp.run(transport="stdio")
