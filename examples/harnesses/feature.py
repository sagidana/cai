"""Port of harnesses/feature.harness.cai.

Six stages: gather → verify → design → validate → implement → write-tests.

Usage:
    python examples/harnesses/feature.py "add rate limiting to the API endpoints"
"""
from cai import Harness
from _helpers import get_task, prepend_task


READ_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "symbol_search", "file_code_outline", "project_code_outline",
]
IMPL_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "symbol_search", "create_file", "edit_file",
]


def main() -> None:
    task = get_task()
    h = Harness()

    # stage 1: gather. loop up to 3 times until verify says 'ok'.
    for _ in range(3):
        r = h.agent(
            tools=READ_TOOLS,
            system_prompt=(
                "You are a senior engineer in the context-gathering phase before "
                "designing a new feature. Your only job is to understand the codebase "
                "deeply enough to design something that fits naturally. Do not design "
                "or implement anything yet."
            ),
            prompt=prepend_task(task, (
                "Gather everything needed to design and implement this feature well. "
                "Read the areas the feature will touch, relevant patterns and "
                "utilities, entry points, tests, and configuration. Only gather context."
            )),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "retry"],
            "Do you have enough context to design this feature well? 'ok' or 'retry'.",
        )
        if verdict == "ok":
            break

    # stage 2: design. loop up to 3 times until validate_design says 'ok'.
    for _ in range(3):
        r = h.agent(
            system_prompt=(
                "You are a senior engineer designing a new feature. Design for the "
                "codebase as it is. Reuse what exists. Follow established patterns. "
                "Keep the design minimal."
            ),
            prompt=(
                "Produce a complete design document: PUBLIC API, INTERNAL DESIGN, "
                "DATA & STATE, ERROR HANDLING, REUSE, TESTING SURFACE, "
                "IMPLEMENTATION STEPS. Be specific."
            ),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "revise"],
            "Review the design. 'ok' only if consistent, reuses utilities, all "
            "failure modes handled, steps safely ordered, complete, minimal. "
            "Otherwise 'revise'.",
            system_prompt=(
                "You are a principal engineer reviewing a feature design before "
                "implementation. You are thorough and skeptical. Answer only 'ok' or 'revise'."
            ),
        )
        if verdict == "ok":
            break

    # stage 3: implement the feature per the approved design.
    r = h.agent(
        tools=IMPL_TOOLS,
        system_prompt=(
            "You are a senior engineer implementing an approved feature design. "
            "Follow the design precisely. Do not improve or refactor adjacent "
            "code. A minimal, correct implementation is the goal."
        ),
        prompt=(
            "Implement the feature exactly as designed. Match project conventions. "
            "Do not write tests in this phase. Show the complete code for every "
            "new or modified file."
        ),
    )
    r.wait()
    h.enrich(r.messages)

    # stage 4: write tests for the implementation.
    r = h.agent(
        tools=IMPL_TOOLS,
        system_prompt=(
            "You are a test engineer writing tests for a newly implemented feature. "
            "Tests must be thorough and match project conventions exactly."
        ),
        prompt=(
            "Write comprehensive tests covering every item in the testing surface "
            "section of the design. Happy path, edge cases, errors, and failures. "
            "Write complete test files."
        ),
    )
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
