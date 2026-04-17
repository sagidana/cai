"""Port of harnesses/explain.harness.cai.

Four stages: gather → verify → trace → explain.

Usage:
    python examples/harnesses/explain.py "explain how request authentication works end-to-end"
"""
from cai import Harness
from _helpers import get_task, prepend_task


READ_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "symbol_search", "file_code_outline", "project_code_outline",
]


def main() -> None:
    task = get_task()
    h = Harness()

    # stage 1: gather. loop up to 3 times until the verify gate says 'ok'.
    for _ in range(3):
        r = h.run_agent(
            tools=READ_TOOLS,
            system_prompt=(
                "You are an expert engineer in the context-gathering phase. Your goal "
                "is to read every piece of code involved in the topic being explained. "
                "Surface-level reading produces inaccurate explanations."
            ),
            prompt=prepend_task(task, (
                "Gather the full context needed to explain the topic accurately. Read "
                "every function, class, and module in the topic's path, plus relevant "
                "configuration and tests. Do not explain yet."
            )),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "retry"],
            "Can you explain the topic accurately from code you've actually read? 'ok' or 'retry'.",
        )
        if verdict == "ok":
            break

    # stage 2: trace. build a structural mental model of the system.
    r = h.run_agent(
        system_prompt=(
            "You are an expert engineer building a precise mental model of a "
            "system. Think in terms of execution paths, data transformations, and "
            "state changes. Build the model structurally before explaining."
        ),
        prompt=(
            "Produce a structural trace with: EXECUTION FLOW (numbered steps), "
            "DATA FLOW, STATE CHANGES, KEY DECISIONS, COMPONENT RELATIONSHIPS. "
            "Use actual names from the code. Be precise."
        ),
    )
    r.wait()
    h.enrich(r.messages)

    # stage 3: explain. write the final layered explanation.
    r = h.run_agent(
        system_prompt=(
            "You are a senior engineer who excels at explaining complex systems "
            "clearly. You are precise, layered, concrete, and use real names from "
            "the code."
        ),
        prompt=(
            "Write a layered explanation based on your trace: ## Overview, "
            "## How It Works, ## Data Flow, ## Error Handling, ## Key Design "
            "Decisions, ## Edge Cases, ## Where to Look."
        ),
    )
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
