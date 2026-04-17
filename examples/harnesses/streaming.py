"""Streaming example: watch the agent work in real time.

Demonstrates iterating over a Result to stream Event objects as the agent
runs. The SDK yields four event types:

    content       — assistant text chunk (stream token-by-token)
    reasoning     — thinking-model reasoning chunk (if enabled)
    tool_call     — a tool invocation is starting
    tool_result   — a tool invocation has returned

This example also demonstrates the hierarchical JSONL log: every Harness +
agent + enrich call writes structured records to the configured log
path. Tail the log in another terminal with ``cai --logger`` to watch the
session live, or open it with any JSONL viewer afterwards.

Usage:
    python examples/harnesses/streaming.py "what are the main classes in src/cai/?"
"""
from cai import Harness
from _helpers import get_task


READ_TOOLS = [
    "read_file",
    "list_files",
    "search",
]

# Write this run's structured log somewhere obvious so the example can
# print the path at the end. Any absolute path works; the directory is
# created on init.
LOG_PATH = "/tmp/cai/streaming-example.log"


def main() -> None:
    task = get_task()

    # Harness as a context manager scopes the log nesting: every record
    # produced inside the `with` block folds as a child of the HARNESS
    # record. log_path overrides the default /tmp/cai/cai.log.
    with Harness(name="streaming-example", log_path=LOG_PATH) as h:
        # kick off the run. agent returns immediately — the worker thread
        # starts when we begin iterating the Result below. Naming the block
        # (here, "explore") makes the BLOCK record self-describing in the log.
        r = h.agent(
                            # tools=READ_TOOLS,
                            skills=['files'],
                            system_prompt=(
                                "You are a senior engineer exploring a codebase. Use tools to read "
                                "what you need, then explain your findings clearly."
                            ),
                            prompt=task,
                            name="explore",
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

    # exiting the `with` emits HARNESS DONE and pops the nest.
    print(f"\nstructured log written to: {LOG_PATH}")
    print(f"view it with:  cai --logger --log-path {LOG_PATH}")


if __name__ == "__main__":
    main()
