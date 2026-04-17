"""Port of harnesses/bug-fix.harness.cai.

Four stages: gather → verify → fix → review. Verify loops back to gather on
retry; review loops back to fix on needs-revision. Bounded by `if-more-than 3`.

Usage:
    python examples/harnesses/bug_fix.py "describe the bug here"
"""
from cai import Harness
from _helpers import get_task, prepend_task


READ_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "symbol_search", "file_code_outline", "git_log", "git_blame",
]
FIX_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "symbol_search", "edit_file", "create_file",
]


def main() -> None:
    task = get_task()
    h = Harness()

    # stage 1: gather. loop up to 3 times until verify says 'ok'.
    for _ in range(3):
        r = h.agent(
            tools=READ_TOOLS,
            system_prompt=(
                "You are an expert debugger in the context-gathering phase. Your only "
                "job is to read code — not to fix anything yet. Trace execution paths "
                "ruthlessly. Read every function, file, and dependency that touches "
                "the bug."
            ),
            prompt=prepend_task(task, (
                "Gather all context needed to understand and correctly fix the reported "
                "bug. Trace the full execution path; read every function, module, and "
                "test that participates. Stop only when you can explain why the bug "
                "occurs based solely on code you have read. Do not output anything. "
                "Do not attempt a fix."
            )),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "retry"],
            "Do you have enough context to diagnose the root cause and apply a "
            "correct fix? Answer 'ok' only if you've read every file in the "
            "execution path, pinpointed the buggy lines, and don't need to guess "
            "at any code. Otherwise 'retry'.",
        )
        if verdict == "ok":
            break

    # stage 2: fix. loop up to 3 times until review says 'ok'.
    for _ in range(3):
        r = h.agent(
            tools=FIX_TOOLS,
            system_prompt=(
                "You are an expert software engineer applying a surgical bug fix. You "
                "have full context. Fix only what is broken — do not refactor, do not "
                "clean up, do not add features. A minimal, correct fix is always "
                "better than a large one."
            ),
            prompt=(
                "Apply the fix for the reported bug. Fix the root cause, not the "
                "symptom. Make the smallest change that correctly fixes it. Preserve "
                "all existing behaviour that is not part of the bug. Show the complete "
                "corrected code for every file you change."
            ),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "needs-revision"],
            "Review the fix you just applied. Answer 'ok' only if it addresses "
            "the root cause, preserves existing behaviour, and handles all edge "
            "cases. Otherwise 'needs-revision'.",
            system_prompt=(
                "You are a senior engineer performing a critical code review of a bug "
                "fix. You are skeptical. Answer only 'ok' or 'needs-revision'."
            ),
        )
        if verdict == "ok":
            break

    # stage 3: done. write a concise summary.
    r = h.agent(
        system_prompt="You are a clear technical communicator. Write a concise, accurate summary.",
        prompt=(
            "Summarize the root cause, what was changed, any edge cases addressed, "
            "and anything the caller should be aware of. Concise and factual."
        ),
    )
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
