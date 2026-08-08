"""Scenario 2 — LLM chat with the ML mental-state classifier attached as an MCP tool.

Spawns mcp_server.py over stdio, advertises its tool to the Qwen model via
llama-server's OpenAI-compatible tool-calling API, executes any tool call the
model makes through the MCP client, and feeds the result back into the
conversation.

Fallback: this llama-server build (Qwen2.5-Coder-3B, --jinja) does NOT populate
the native OpenAI-style `tool_calls` field — instead it writes the tool call as
plain-text JSON in `content` (e.g. a ```json fenced {"name": ..., "arguments":
{...}} block). We still pass `tools=` so the model knows the tool exists, but
we detect and parse that plain-text JSON call ourselves (ReAct-style dispatch)
since the API-level `tool_calls` field is not usable here.

Run: uv run python chat_with_tools.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sys

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import settings
from llm_runner import ensure_llama_server_running

SYSTEM_PROMPT = (
    "You are a supportive assistant. When the user expresses feelings, mood, "
    "or emotional distress, call the analyze_mental_state_tool to ground your "
    "reply in an actual assessment. In your reply, explicitly state the "
    "detected mental state by name (e.g. \"It sounds like you may be "
    "experiencing Anxiety.\"), then respond warmly using its tips/yoga "
    "suggestions. For neutral messages, just chat normally without the tool."
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def mcp_tool_to_openai(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }


def print_detected_state(tool_name: str, result_text: str) -> None:
    """Print the ML classifier's predicted state directly from the tool result,
    so it's always shown to the user regardless of whether the model's own
    reply mentions it."""
    if tool_name != "analyze_mental_state_tool":
        return
    try:
        data = json.loads(result_text)
    except json.JSONDecodeError:
        return
    prediction = data.get("prediction")
    if prediction:
        print(f"[Detected state: {prediction}]")


def parse_fallback_tool_call(content: str, valid_names: set[str]) -> tuple[str, dict] | None:
    """Best-effort extraction of a tool call the model wrote as plain JSON text
    instead of a native tool_calls entry, e.g. a ```json {"name": ..., "arguments":
    {...}} ``` block."""
    match = _JSON_BLOCK_RE.search(content)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    name = data.get("name") or data.get("tool")
    args = data.get("arguments") or data.get("args") or {}
    if name in valid_names and isinstance(args, dict):
        return name, args
    return None


async def run_chat() -> None:
    ensure_llama_server_running()
    print(f"Connected to llama-server at {settings.llama_server_base_url}.")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            openai_tools = [mcp_tool_to_openai(t) for t in tools_result.tools]
            tool_names = {t.name for t in tools_result.tools}
            print(f"MCP tools available: {[t.name for t in tools_result.tools]}")
            print("Type 'exit' to quit.\n")

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            async with httpx.AsyncClient(timeout=120.0) as client:
                while True:
                    user_input = input("You: ").strip()
                    if user_input.lower() in {"exit", "quit"}:
                        break
                    if not user_input:
                        continue

                    messages.append({"role": "user", "content": user_input})

                    # Loop to allow the model to chain a tool call before its final reply.
                    for _ in range(4):
                        resp = await client.post(
                            f"{settings.llama_server_base_url}/v1/chat/completions",
                            json={
                                "model": "qwen2.5-coder-3b-instruct",
                                "messages": messages,
                                "tools": openai_tools,
                            },
                        )
                        resp.raise_for_status()
                        choice = resp.json()["choices"][0]["message"]
                        tool_calls = choice.get("tool_calls")
                        content = choice.get("content", "") or ""

                        if not tool_calls:
                            fallback = parse_fallback_tool_call(content, tool_names)
                            if fallback is None:
                                print(f"Qwen: {content}\n")
                                messages.append({"role": "assistant", "content": content})
                                break

                            fn_name, fn_args = fallback
                            print(f"[tool call] {fn_name}({fn_args})")
                            result = await session.call_tool(fn_name, fn_args)
                            result_text = "".join(
                                part.text for part in result.content if hasattr(part, "text")
                            )
                            print_detected_state(fn_name, result_text)
                            # No native tool_calls to echo back, so record what happened
                            # as plain conversation turns the model can read and build on.
                            messages.append({"role": "assistant", "content": content})
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        f"[Tool result for {fn_name}]: {result_text}\n\n"
                                        "Now reply to the user directly in plain language "
                                        "(no more tool calls needed)."
                                    ),
                                }
                            )
                            continue

                        messages.append(choice)
                        for call in tool_calls:
                            fn_name = call["function"]["name"]
                            fn_args = json.loads(call["function"]["arguments"] or "{}")
                            print(f"[tool call] {fn_name}({fn_args})")
                            result = await session.call_tool(fn_name, fn_args)
                            result_text = "".join(
                                part.text for part in result.content if hasattr(part, "text")
                            )
                            print_detected_state(fn_name, result_text)
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call["id"],
                                    "content": result_text,
                                }
                            )


def main() -> None:
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()
