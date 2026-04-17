"""Port of harnesses/context-and-execute.harness.cai.

Canonical harness: gather until sufficient, then execute. The verify gate
retries context-gathering (capped by `if-more-than 3 done`) before unblocking
the execute stage.

Usage:
    python examples/harnesses/context_and_execute.py "your task here"
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
    h = Harness()

    # stage 1: enrichment. loop up to 3 times until verify says 'ok'.
    for _ in range(3):
        r = h.agent(
            tools=READ_TOOLS,
            system_prompt=(
                "You are a meticulous context-gathering agent. Your only job in this "
                "phase is to read and understand — never to execute or produce output. "
                "Be thorough: it is better to read one extra file than to miss a "
                "critical one. Follow every import, base class, and dependency you "
                "encounter until you have a complete picture."
            ),
            prompt=prepend_task(task, (
                "Gather every file and piece of context needed to complete the user's "
                "task to a high standard. Read broadly; follow imports, base classes, "
                "and callers until you have a complete, accurate picture. Produce no "
                "output. Only gather context."
            )),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "retry"],
            "Is the gathered context sufficient to complete the task correctly "
            "and completely without guessing? Answer 'ok' or 'retry'.",
        )
        if verdict == "ok":
            break

    # stage 2: execute. produce the final result.
    r = h.agent(
        tools=EDIT_TOOLS,
        system_prompt=(
            "You are an expert software engineer executing a task with full context "
            "already gathered. Produce a complete, correct, high-quality result "
            "that precisely matches the conventions and patterns of the existing "
            "codebase."
        ),
        prompt=prepend_task(task, (
            "You now have the full context required to complete the user's task. "
            "Follow existing conventions, make only the changes necessary, and "
            "produce the complete final result now."
        )),
    )
    r.wait()
    h.enrich(r.messages)
    print(r.text)


if __name__ == "__main__":
    main()
