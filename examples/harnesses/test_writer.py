"""Port of harnesses/test-writer.harness.cai.

Five stages: gather-source → gather-tests → plan → validate-plan → write.

Usage:
    python examples/harnesses/test_writer.py "write tests for src/auth/token.py"
"""
from cai import Harness
from _helpers import get_task, prepend_task


READ_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "symbol_search", "file_code_outline",
]
CONV_TOOLS = ["read", "read_lines", "list_files", "pattern_search"]
WRITE_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "create_file", "edit_file",
]


def main() -> None:
    task = get_task()
    h = Harness()

    # stage 1: gather the source code to test.
    r = h.agent(
        tools=READ_TOOLS,
        system_prompt=(
            "You are a test engineer in the context-gathering phase. Deeply "
            "understand the code you will test — intended behaviour, edge cases, "
            "failure modes, dependencies. Do not write tests yet."
        ),
        prompt=prepend_task(task, (
            "Read the target file(s) completely. Identify happy path, edge cases, "
            "error cases, and pre/postconditions for every public function. Read "
            "imported dependencies and any existing tests."
        )),
    )
    r.wait()
    h.enrich(r.messages)

    # stage 2: gather the project's testing conventions.
    r = h.agent(
        tools=CONV_TOOLS,
        system_prompt=(
            "You are a test engineer studying the project's testing conventions. "
            "Goal: write tests indistinguishable from existing ones."
        ),
        prompt=(
            "Read 3–5 representative existing test files. Note framework, "
            "naming, fixtures/mocks, assertions, grouping, test-data layout. Do "
            "not write tests yet."
        ),
    )
    r.wait()
    h.enrich(r.messages)

    # stage 3: plan. loop up to 3 times until validate_plan says 'ok'.
    for _ in range(3):
        r = h.agent(
            system_prompt=(
                "You are a senior test engineer planning a comprehensive test suite. "
                "A plan is only good if it covers every meaningful case — not just "
                "the happy path."
            ),
            prompt=(
                "Enumerate every test case. For each: name, what's tested, input, "
                "expected output, mocks required. Group by happy path, edge cases, "
                "error cases, failure cases, state/side-effect cases."
            ),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "insufficient"],
            "Is the plan comprehensive? 'ok' only if every public item has a "
            "happy-path test, edge cases covered, errors covered, dependencies "
            "addressed, cases implementable unambiguously, no duplication.",
            system_prompt="You are a senior test engineer reviewing a test plan. Answer only 'ok' or 'insufficient'.",
        )
        if verdict == "ok":
            break

    # stage 4: write the tests per the approved plan.
    r = h.agent(
        tools=WRITE_TOOLS,
        system_prompt=(
            "You are a test engineer implementing a pre-approved plan. Write "
            "every case. Follow project conventions exactly."
        ),
        prompt=(
            "Implement every test case from the plan. Match framework, naming, "
            "fixtures, mocking style exactly. Each test independent and "
            "single-purpose. Show the complete test file(s)."
        ),
    )
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
