"""Scenario 1 — plain LLM chat against the local Qwen model, no tools attached.

Run: uv run python chat_plain.py
"""
from __future__ import annotations

import httpx

from config import settings
from llm_runner import ensure_llama_server_running

SYSTEM_PROMPT = "You are a helpful, concise assistant."


def main() -> None:
    ensure_llama_server_running()
    print(f"Connected to llama-server at {settings.llama_server_base_url}. Type 'exit' to quit.\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    with httpx.Client(timeout=120.0) as client:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            resp = client.post(
                f"{settings.llama_server_base_url}/v1/chat/completions",
                json={"model": "qwen2.5-coder-3b-instruct", "messages": messages},
            )
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"]
            print(f"Qwen: {reply}\n")
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
