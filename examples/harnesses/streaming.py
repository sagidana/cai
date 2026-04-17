"""Streaming example: watch the agent work in real time.

Demonstrates iterating over a Result to stream Event objects as the agent
runs. The SDK yields four event types:

    content       — assistant text chunk (stream token-by-token)
    reasoning     — thinking-model reasoning chunk (if enabled)
    tool_call     — a tool invocation is starting
    tool_result   — a tool invocation has returned

Usage:
    python examples/harnesses/streaming.py "what are the main classes in src/cai/?"
"""
from cai import Harness
from _helpers import get_task


READ_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "symbol_search", "file_code_outline", "project_code_outline",
]


def main() -> None:
    task = get_task()
    h = Harness()

    # kick off the run. run_agent returns immediately — the worker thread
    # starts when we begin iterating the Result below.
    r = h.run_agent(
        tools=READ_TOOLS,
        system_prompt=(
            "You are a senior engineer exploring a codebase. Use tools to read "
            "what you need, then explain your findings clearly."
        ),
        prompt=task,
    )

    # iterate the Result to stream events as they happen. Each event has a
    # .type and the fields relevant to that type.
    for event in r:
        if event.type == "content":
            # assistant text: print chunks inline, no newline between them.
            print(event.text, end="", flush=True)

        elif event.type == "reasoning":
            # reasoning chunks from thinking models. prefix so they're visible.
            print(event.text, end="", flush=True)

        elif event.type == "tool_call":
            # fires before the tool runs. show name + arguments.
            print(f"\n\n→ tool_call: {event.tool_name}({event.tool_args})")

        elif event.type == "tool_result":
            # fires when the tool returns. truncate long results for readability.
            status = "error" if event.is_error else "ok"
            result_text = event.tool_result or ""
            preview = result_text[:200].replace("\n", " ")
            suffix = "..." if len(result_text) > 200 else ""
            print(f"← tool_result [{status}]: {preview}{suffix}\n")

    # the iterator is fully drained at this point. final-state fields are
    # populated and safe to read.
    print(f"\n\n--- finished ({r.finish_reason}) ---")
    if r.error:
        print(f"error: {r.error}")


if __name__ == "__main__":
    main()
