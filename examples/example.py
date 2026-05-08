"""
example.py — end-to-end SDK tour, woven into one cohesive use case: a quick
audit of a Python project.

Demonstrated features
- Custom system prompt at both the Harness layer and per ``agent()`` call.
- Custom tool registered via ``functions=`` (``line_count`` below).
- ``strict_format`` to coerce a clean bullet list of file paths.
- ``Agent(...)`` constructed directly with caller-owned messages, used to
  dispatch one sub-agent task per entry produced by the strict-format step.
- ``Harness.enrich(...)`` in both modes:
    * ``enrich(messages)`` — merge the *full* conversation that produced the
      file picks (including tool calls/results) back into the parent.
    * ``enrich(text)``     — merge only a sub-agent's final reply, keeping the
      sub-agent's intermediate turns out of the parent's context.
- ``Harness.clone()`` for a speculative deep-dive that the parent decides
  whether to keep.
- ``Harness.gate(...)`` for two strict-format decisions: keep-the-deep-dive,
  and a final ship/fix-first verdict.
"""

from __future__ import annotations

from cai import Harness, Agent


def line_count(path: str) -> str:
    """Return the line count of ``path`` as a decimal string. Errors are
    returned as text so the agent can decide what to do in-band."""
    try:
        with open(path) as f:
            return str(sum(1 for _ in f))
    except OSError as e:
        return f"error: {e}"


with Harness(
    name="audit",
    system_prompt=(
        "You are a senior engineer auditing a small Python project. "
        "Be terse and concrete. Prefer paths, numbers, and one-sentence verdicts."
    ),
    skills=["files"],            # list_files / read_file / search
    functions=[line_count],      # custom tool exposed alongside the skill tools
    log_path="/tmp/cai/example.log",
) as h:

    # ── 1) strict_format → bullet list of files to inspect ────────────────────
    # Per-call system_prompt *appends* to the harness's, narrowing the persona
    # for this one turn without touching subsequent calls.
    listing = h.agent(
        system_prompt=(
            "For this turn, behave like a triage tool: scan, count, list. "
            "No prose, no preamble — just the bullets."
        ),
        prompt=(
            "Scan src/cai/ with list_files, then use line_count to identify "
            "up to 3 of the largest .py files. Output one '- <path>' per "
            "line, nothing else."
        ),
        strict_format=r"regex-each-line:^- .+$",
        name="pick-files",
    ).wait()
    paths = [ln[2:].strip() for ln in listing.text.splitlines() if ln.startswith("- ")]
    print(f"selected: {paths}")

    # Enrich the *full* conversation that produced the picks — including the
    # tool calls/results the agent used to scan the tree. Subsequent harness
    # turns can now reason over the same evidence, not just the final list.
    h.enrich(listing.messages)

    # ── 2) Per-entry sub-agents via Agent(...) with caller-owned messages ────
    # Each sub-agent gets its own one-shot context (no harness state), runs
    # independently, and only its final reply is folded back to the parent.
    for path in paths:
        n_lines = line_count(path)
        sub = Agent(
            messages=[{
                "role": "user",
                "content": (
                    f"{path} has {n_lines} lines. In one sentence, what is its "
                    f"role in this project? Don't speculate beyond the path."
                ),
            }],
            system_prompt="You are a precise code summariser. One sentence only.",
            tools=[],               # text-only sub-task; no tools needed
            model=h.model,
            config={},              # vestigial; Agent doesn't read it
            block_name=f"sub-{path}",
        ).wait()

        # Response-only enrichment: parent learns the conclusion, not the
        # sub-agent's internal turns.
        h.enrich(f"{path}: {sub.text}")

    # ── 3) Speculative deep-dive on a clone, consumed as a stream ────────────
    # The clone shares bootstrap state (config, tool registry, MCP servers)
    # but owns its own ``messages``, so anything the deep-dive does is
    # invisible to the parent until we explicitly enrich it back.
    #
    # Instead of ``.wait()`` we iterate the Agent directly: each Event arrives
    # as it's produced. The for-loop drains the iterator, after which final
    # fields like ``deep.text`` are populated — same as if we had waited.
    spec = h.clone(name="audit-deepdive")
    deep = spec.agent(
        prompt=(
            "Anything alarming about coupling between the files above? "
            "Be specific — name modules and the smell."
        ),
        name="deepdive",
    )
    print("\n--- deepdive (streaming) ---")
    for event in deep:
        if event.type == "content":
            print(event.text, end="", flush=True)
        elif event.type == "tool_call":
            print(f"\n→ {event.tool_name}({event.tool_args})")
        elif event.type == "tool_result":
            status = "error" if event.is_error else "ok"
            print(f"\n← [{status}] {(event.tool_result or '')[:120]}")
    print("\n--- end deepdive ---")
    spec.close()

    # gate: parent decides whether the deep-dive earns a place in the record.
    keep = h.gate(
        options=["keep", "drop"],
        prompt=f"Worth keeping in the audit?\n\n{deep.text}",
    )
    if keep == "keep":
        h.enrich(deep.text)         # response only — not the deep-dive's turns

    # ── 4) Final verdict via gate ─────────────────────────────────────────────
    verdict = h.gate(
        options=["ship", "fix-first"],
        prompt="Based on the audit above, ship or fix-first?",
    )
    print(f"\nverdict: {verdict}")
