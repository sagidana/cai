"""Port of harnesses/refactor.harness.cai.

Six stages: gather → verify → plan → validate-plan → execute → sanity → done.

Usage:
    python examples/harnesses/refactor.py "extract the retry logic in api.py into a standalone utility"
"""
from cai import Harness
from _helpers import get_task, prepend_task


READ_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "symbol_search", "file_code_outline", "project_code_outline",
]
EXEC_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search", "symbol_search",
    "edit_file", "create_file", "rename_file", "remove_file",
]


def main() -> None:
    task = get_task()
    h = Harness()

    # stage 1: gather. loop up to 3 times until verify says 'ok'.
    for _ in range(3):
        r = h.run_agent(
            tools=READ_TOOLS,
            system_prompt=(
                "You are an expert refactoring engineer in the context-gathering phase. "
                "Refactoring without full context causes regressions. Read every file "
                "the refactor touches, every callsite, every test."
            ),
            prompt=prepend_task(task, (
                "Gather the complete context needed to refactor safely: target files, "
                "dependencies, all callsites, tests, and existing utilities. Do not "
                "begin planning."
            )),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "retry"],
            "Is your context complete enough to plan a safe refactor? 'ok' or 'retry'.",
        )
        if verdict == "ok":
            break

    # stage 2: plan. loop up to 3 times until validate_plan says 'ok'.
    for _ in range(3):
        r = h.run_agent(
            system_prompt=(
                "You are a senior engineer designing a refactor plan. A good plan is "
                "a sequence of small, safe, independently-verifiable steps."
            ),
            prompt=(
                "Write a detailed step-by-step plan: GOAL, STEPS, PUBLIC API CHANGES, "
                "CALLSITE UPDATES, RISK AREAS. Each step must leave the code in a "
                "working state."
            ),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "revise"],
            "Review the plan. 'ok' only if behaviour preserved, every callsite "
            "listed, steps safely ordered, every changed file named, tests "
            "accounted for, steps specific, no missing steps.",
            system_prompt=(
                "You are a principal engineer reviewing a refactor plan. You are "
                "skeptical. Answer only 'ok' or 'revise'."
            ),
        )
        if verdict == "ok":
            break

    # stage 3: execute the plan. loop up to 3 times until sanity check is 'clean'.
    for _ in range(3):
        r = h.run_agent(
            tools=EXEC_TOOLS,
            system_prompt=(
                "You are an expert engineer executing an approved refactor plan. "
                "Follow precisely. Do not improvise. Do not add unrequested changes."
            ),
            prompt=(
                "Execute the refactor plan exactly as written. Make only the "
                "specified changes. Update every listed callsite. Show complete "
                "code for every changed file."
            ),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["clean", "issues"],
            "Review the completed refactor. 'clean' only if every callsite was "
            "updated, public APIs preserved, tests still pass, no dead code "
            "left, all plan steps fully executed.",
            system_prompt="You are a senior engineer performing a final regression check. Answer only 'clean' or 'issues'.",
        )
        if verdict == "clean":
            break

    # stage 4: done. final summary.
    r = h.run_agent(
        system_prompt="You are a clear technical communicator. Write a concise, accurate refactor summary.",
        prompt=(
            "Summarize: what was refactored, which files changed, which callsites "
            "updated, which tests updated, any deviations from the plan."
        ),
    )
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
