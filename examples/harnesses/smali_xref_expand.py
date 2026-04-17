"""Port of harnesses/smali-xref-expand.harness.cai.

Sub-harness: one BFS frontier expansion step. Input is a single JSON envelope
line describing a descriptor to expand; output is one line per new frontier
node, with 'DEFERRED:' prefix for deferred-execution hops.

Usage (standalone):
    python examples/harnesses/smali_xref_expand.py \\
      '{"descriptor":"Lcom/example/Target;->method()V","mode":"callers","depth":0,"visited":"","hop_type":"direct"}'
"""
from cai import Harness
from _helpers import get_task


def main() -> None:
    envelope_json = get_task()
    h = Harness()

    r = h.run_agent(
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
            f"JSON envelope:\n{envelope_json}\n\n"
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
    print(r.text or "")


if __name__ == "__main__":
    main()
