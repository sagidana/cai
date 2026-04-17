"""Port of harnesses/multi-task.harness.cai.

decompose → for-each subtask do (context → execute) in an isolated harness →
aggregate.

Each subtask runs with its own fresh Harness so quality does not degrade with
breadth. Mirrors the isolated-context property of the for-each directive.

Usage:
    python examples/harnesses/multi_task.py "write tests for every module in src/cai/"
"""
from cai import Harness
from _helpers import get_task, prepend_task


READ_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "symbol_search", "file_code_outline", "project_code_outline",
]
EDIT_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "symbol_search", "create_file", "edit_file", "rename_file",
]


def main() -> None:
    task = get_task()

    # stage 1: decompose the task into atomic subtasks.
    h_decompose = Harness()
    r = h_decompose.run_agent(
        tools=["list_files", "read_lines"],
        system_prompt=(
            "Output a plain list of independent tasks — one per line, no bullets, "
            "no numbering, no extra text."
        ),
        prompt=(
            f"User task:\n{task}\n\n"
            "Break this into atomic, self-contained subtasks that can each be "
            "completed in isolation. Include all context each executor needs "
            "(file paths, specific goals, constraints). One per line."
        ),
    )
    r.wait()
    subtasks = [line.strip() for line in r.text.splitlines() if line.strip()]
    print(f"[decomposed into {len(subtasks)} subtasks]")

    # stage 2: run each subtask in an isolated harness (context → execute).
    # Fold each child's final messages into the parent so the aggregator sees them.
    parent = Harness()
    for i, subtask in enumerate(subtasks, 1):
        print(f"\n=== subtask {i}/{len(subtasks)}: {subtask}")
        child = Harness()

        # enrichment: gather context for this subtask.
        r = child.run_agent(
            tools=READ_TOOLS,
            system_prompt=(
                "You are a meticulous context-gathering agent. Your only job in this "
                "phase is to read and understand — never to execute or produce output. "
                "Be thorough: it is better to read one extra file than to miss a "
                "critical one. Follow every import, base class, and dependency you "
                "encounter until you have a complete picture."
            ),
            prompt=prepend_task(subtask, (
                "Gather every file and piece of context needed to complete the user's "
                "task to a high standard. Read broadly; follow imports, base classes, "
                "and callers until you have a complete, accurate picture. Produce no "
                "output. Only gather context."
            )),
        )
        r.wait()
        child.enrich(r.messages)

        # execute: produce the final result for this subtask.
        r = child.run_agent(
            tools=EDIT_TOOLS,
            system_prompt=(
                "You are an expert software engineer executing a task with full context "
                "already gathered. Produce a complete, correct, high-quality result "
                "that precisely matches the conventions and patterns of the existing "
                "codebase."
            ),
            prompt=prepend_task(subtask, (
                "You now have the full context required to complete the user's task. "
                "Follow existing conventions, make only the changes necessary, and "
                "produce the complete final result now."
            )),
        )
        r.wait()
        child.enrich(r.messages)
        print(r.text)

        # fold child transcript into parent so aggregate can reference it.
        parent.messages.extend(child.messages)

    # stage 3: aggregate all child transcripts into a final summary.
    r = parent.run_agent(
        system_prompt="You are a concise technical writer summarising completed work.",
        prompt=(
            "All subtasks have been completed (their results are in your context). "
            "Summarize what was accomplished, notable decisions/patterns, and "
            "anything the user should review."
        ),
    )
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
