"""Port of harnesses/code-review.harness.cai.

Five stages: gather → verify → analyze → classify → report. classify routes to
one of three report variants (blocking / non-blocking / clean).

Usage:
    python examples/harnesses/code_review.py "review the changes in src/auth/"
"""
from cai import Harness
from _helpers import get_task, prepend_task


READ_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search", "symbol_search",
    "file_code_outline", "git_diff", "git_log", "git_blame",
]


def main() -> None:
    task = get_task()
    h = Harness()

    # stage 1: gather. loop up to 3 times until verify says 'ok'.
    for _ in range(3):
        r = h.run_agent(
            tools=READ_TOOLS,
            system_prompt=(
                "You are an experienced code reviewer in the context-gathering phase. "
                "Before you can review code fairly, you must understand its full "
                "context. Gather all of that now."
            ),
            prompt=prepend_task(task, (
                "Gather all context needed for a meaningful, accurate code review: "
                "read every file under review in full, its imports and callsites, "
                "related tests, and enough surrounding code to judge project "
                "conventions. Do not write the review yet."
            )),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "retry"],
            "Do you have enough context to write a fair, thorough, and accurate "
            "review? Answer 'ok' only if yes. Otherwise 'retry'.",
        )
        if verdict == "ok":
            break

    # stage 2: analyze. produce structured findings without writing the report.
    r = h.run_agent(
        system_prompt=(
            "You are a senior engineer performing a thorough code review. You are "
            "thorough, fair, and constructive. Distinguish blocking issues from "
            "non-blocking suggestions. Be specific — cite exact lines."
        ),
        prompt=(
            "Perform a deep analysis across correctness, security, performance, "
            "maintainability, testing, and conventions. Label each finding "
            "blocking or non-blocking. Do not write the final report — just the "
            "analysis."
        ),
    )
    r.wait()
    h.enrich(r.messages)

    # stage 3: classify. pick one verdict.
    verdict = h.gate(
        ["blocking", "non-blocking", "clean"],
        "Classify the overall verdict: 'blocking' (issue must be fixed before "
        "merge), 'non-blocking' (merge OK but suggestions remain), or 'clean' "
        "(ready as-is).",
        system_prompt="You are a senior engineer making a merge decision. Answer with exactly one word.",
    )

    # stage 4: report. pick system/prompt per verdict and write final review.
    if verdict == "blocking":
        system = (
            "You are a senior engineer writing a code review report. Be direct "
            "and constructive. Blocking issues must be impossible to miss."
        )
        prompt = (
            "Write the final review with verdict CHANGES REQUIRED. List each "
            "blocking issue (location, problem, why it blocks, how to fix), then "
            "non-blocking suggestions, then positive notes."
        )
    elif verdict == "non-blocking":
        system = (
            "You are a senior engineer writing a code review report. Be "
            "constructive and specific. The code is mergeable but can improve."
        )
        prompt = (
            "Write the final review with verdict APPROVED WITH SUGGESTIONS. List "
            "each suggestion (location, opportunity, why it matters, recommendation), "
            "then positive notes, then a 1–2 sentence summary."
        )
    else:
        system = "You are a senior engineer writing a clean sign-off. Be concise and specific."
        prompt = (
            "Write the final review with verdict APPROVED. 2–4 sentence summary "
            "and 2–3 specific highlights. Keep it short."
        )
    r = h.run_agent(system_prompt=system, prompt=prompt)
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
