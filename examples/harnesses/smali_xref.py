"""Port of harnesses/smali-xref.harness.cai.

Stages:
  1. classify   — resolve descriptor, classify query → emit JSON envelope
  2. verify     — quality gate (ok / error)
  3. bfs_start  — for-each descriptor from classify → expand
  4. bfs_loop   — bfs_continue (expand / done); next_frontier → for-each (layers 1–4)
  5. synthesize — assemble graph into the user's answer

Usage:
    python examples/harnesses/smali_xref.py "what calls the encrypt method?"
"""
from cai import Harness
from _helpers import get_task


MAX_DEPTH = 5
MAX_LAYERS = 5


def main() -> None:
    task = get_task()
    h = Harness()

    # stage 1: classify. resolve the descriptor and emit one JSON envelope line
    # (or ERROR:…). Strict format enforces the single-line output shape.
    r = h.agent(
        strict_format=r"regex:^(\{\"descriptor\":\"L.+\}|ERROR:.+)$",
        tools=["smali_resolve_descriptor", "list_files", "pattern_search"],
        system_prompt=(
            "You are a smali reverse-engineering specialist. Resolve the target "
            "to a precise descriptor. Output ONLY a single JSON envelope line or "
            "an ERROR: line — no prose, no explanation."
        ),
        prompt=(
            f"User question:\n{task}\n\n"
            "STEP 1 orient in the project (list .smali files).\n"
            "STEP 2 resolve target descriptor via smali_resolve_descriptor / "
            "pattern_search / file_code_outline.\n"
            "STEP 3 classify query: CALLERS, CALLEES, REACHABILITY, PATH, FULL-XREF. "
            "mode mapping → 'callers' for REACHABILITY/PATH/FULL-XREF; "
            "'callees' for CALLEES.\n"
            "STEP 4 output EXACTLY ONE LINE:\n"
            "  {\"descriptor\":\"<smali>\",\"mode\":\"<callers|callees>\","
            "\"depth\":0,\"visited\":\"\",\"hop_type\":\"direct\","
            "\"source\":\"<for REACHABILITY/PATH, else none>\"}\n"
            "Or on failure: ERROR: <concise reason>"
        ),
    )
    r.wait()
    h.enrich(r.messages)
    envelope = r.text.strip()

    # stage 2: verify the classify output was a valid envelope.
    verdict = h.gate(
        ["ok", "error"],
        "Review the classify block's output. 'ok' if JSON line starting "
        "with {\"descriptor\":\"L; 'error' if ERROR: line.",
    )
    if verdict != "ok":
        # error path: explain why we couldn't resolve the query.
        r = h.agent(
            system_prompt="You are a helpful assistant explaining why a smali XRef query could not be completed.",
            prompt=(
                f"User asked: {task}\nResolution failure: {envelope}\n\n"
                "Explain concisely what the user asked, why it couldn't be resolved, "
                "and how to specify the query more precisely. Show the descriptor "
                "format: Lcom/example/Class;->method(Args)Return"
            ),
        )
        r.wait()
        print(r.text)
        return

    # stage 3 + 4: BFS expansion. layer 0 starts from the classify envelope;
    # each layer expands every frontier node in its own fresh harness, then
    # bfs_continue / next_frontier decide whether to keep going.
    frontier = [envelope]
    for _layer in range(MAX_LAYERS):
        # expand each frontier node in isolation.
        for env in frontier:
            h_expand = Harness()
            r = h_expand.agent(
                tools=[
                    "smali_find_callers", "smali_find_callees",
                    "smali_find_implementations", "read_lines",
                ],
                system_prompt=(
                    "You are a smali reverse-engineering analyst performing one BFS "
                    "expansion step. Parse the JSON envelope, run the appropriate tools, "
                    "and output ONLY new descriptor strings — one per line, no "
                    "explanation, no JSON. Use read_lines for spot-reads only."
                ),
                prompt=(
                    f"JSON envelope:\n{env}\n\n"
                    "STEP 1 primary lookup: mode=callers → smali_find_callers; "
                    "mode=callees → smali_find_callees.\n"
                    "STEP 2 (callers only) interface/abstract expansion via "
                    "smali_find_implementations, then smali_find_callers on each concrete.\n"
                    "STEP 3 deferred-execution detection: Thread.start, Handler.post, "
                    "Executor.submit, AsyncTask.execute — emit scheduling site with "
                    "'DEFERRED:' prefix.\n"
                    "STEP 4 deduplicate against the 'visited' field.\n"
                    "STEP 5 output one descriptor per line (empty output is valid). "
                    "Nothing else."
                ),
            )
            r.wait()
            # fold this expansion's output back into the parent harness so
            # bfs_continue / next_frontier can see it.
            h.enrich(r.text or "")

        # decide whether to keep expanding.
        keep_going = h.gate(
            ["expand", "done"],
            "Review the recent expansion results. 'expand' if new descriptors "
            f"found AND depth < {MAX_DEPTH} AND (for REACHABILITY/PATH) source "
            "not yet in caller graph. Otherwise 'done'.",
            system_prompt="You are a BFS controller. Answer only 'expand' or 'done'.",
        )
        if keep_going == "done":
            break

        # compute the next frontier: one JSON envelope line per not-yet-visited descriptor.
        r = h.agent(
            system_prompt=(
                "You are a BFS frontier manager. Emit JSON envelope lines for the "
                "next expansion round — one per non-visited descriptor. No "
                "explanations. Output FRONTIER_EMPTY if nothing new."
            ),
            prompt=(
                "From the latest expansion, collect every descriptor. Build the "
                "visited set (union of all prior rounds). For each not-visited "
                "descriptor, emit:\n"
                "  {\"descriptor\":\"<desc>\",\"mode\":\"<same>\",\"depth\":<N+1>,"
                "\"visited\":\"<csv>\",\"hop_type\":\"<direct|deferred>\"}\n"
                "Strip the DEFERRED: prefix for the descriptor field; hop_type=deferred. "
                "Output FRONTIER_EMPTY if none."
            ),
        )
        r.wait()
        out = (r.text or "").strip()
        if not out or out == "FRONTIER_EMPTY":
            break
        frontier = [line for line in out.splitlines() if line.strip().startswith("{")]
        if not frontier:
            break

    # stage 5: synthesize the accumulated graph into the user-facing answer.
    r = h.agent(
        tools=[
            "smali_find_callers", "smali_find_callees",
            "smali_find_implementations", "read_lines",
        ],
        system_prompt=(
            "You are a reverse-engineering analyst specialising in Android smali. "
            "Produce a precise, structured cross-reference analysis. Use "
            "read_lines for spot-reads only."
        ),
        prompt=(
            f"User question:\n{task}\n\n"
            "Sections: 1) Query Summary, 2) Call Graph (tree for callers/"
            "callees; YES/NO + path for reachability; numbered path for PATH), "
            "3) Interface Dispatch and Inheritance, 4) Deferred Execution Hops, "
            "5) Coverage Gaps, 6) Summary paragraph answering the question."
        ),
    )
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
