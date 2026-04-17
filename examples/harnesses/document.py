"""Port of harnesses/document.harness.cai.

Five stages: gather-code → gather-conventions → outline → validate-outline → write.

Usage:
    python examples/harnesses/document.py "document the public API of src/cai/harness.py"
"""
from cai import Harness
from _helpers import get_task, prepend_task


READ_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "symbol_search", "file_code_outline",
]
CONV_TOOLS = ["read", "read_lines", "list_files", "pattern_search"]
WRITE_TOOLS = ["read", "read_lines", "list_files", "edit_file", "create_file"]


def main() -> None:
    task = get_task()
    h = Harness()

    # stage 1: gather code being documented.
    r = h.agent(
        tools=READ_TOOLS,
        system_prompt=(
            "You are a technical writer in the context-gathering phase. Read the "
            "code being documented with the precision of someone who must describe "
            "it perfectly to an outsider."
        ),
        prompt=prepend_task(task, (
            "Gather a complete understanding: purpose, inputs, outputs, errors, "
            "side effects, pre/postconditions for every public item. Read "
            "callsites, tests, and any existing partial docs. Do not write yet."
        )),
    )
    r.wait()
    h.enrich(r.messages)

    # stage 2: gather doc conventions used in the project.
    r = h.agent(
        tools=CONV_TOOLS,
        system_prompt=(
            "You are a technical writer studying the project's documentation "
            "style. Your goal is to produce docs indistinguishable from existing "
            "ones in format, tone, and depth."
        ),
        prompt=(
            "Read 3–5 similar docstrings, the README, and any doc config. Note "
            "format (Google/NumPy/RST/JSDoc/prose), param docs, returns, errors, "
            "tone, examples, level of detail. Produce a short convention summary."
        ),
    )
    r.wait()
    h.enrich(r.messages)

    # stage 3: outline. loop up to 3 times until validate_outline says 'ok'.
    for _ in range(3):
        r = h.agent(
            system_prompt=(
                "You are a technical writer planning a documentation structure. A "
                "good outline ensures every public interface is covered and nothing "
                "is omitted."
            ),
            prompt=(
                "Produce a structured outline listing every item (module, classes, "
                "functions, constants, examples, warnings) with what each entry will "
                "cover. Be specific."
            ),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "incomplete"],
            "Is the outline complete? 'ok' only if every public item, parameter, "
            "return, exception, and caveat is planned, and it follows project "
            "conventions. Otherwise 'incomplete'.",
            system_prompt=(
                "You are a senior technical writer reviewing an outline. Answer only "
                "'ok' or 'incomplete'."
            ),
        )
        if verdict == "ok":
            break

    # stage 4: write final docs.
    r = h.agent(
        tools=WRITE_TOOLS,
        system_prompt=(
            "You are a technical writer producing final documentation. Write with "
            "precision, clarity, and consistency. Every statement must be accurate."
        ),
        prompt=(
            "Write the complete documentation following the approved outline and "
            "project conventions. Document every item. Include examples for "
            "non-trivial APIs. Show the full output."
        ),
    )
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
