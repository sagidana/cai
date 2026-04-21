import json

from cai import Harness, mask_hook, compact_hook
from cai import llm as cai_llm


# No-op hooks for every event the SDK exposes. Each receives a ctx dict with:
#   messages, model, usage, tool_call, content
# Event-specific return-value semantics:
#   before_tool_call  -> return False to veto the call; anything else = allow
#   on_final_response -> return a str to replace the assistant's final content
#   after_tool_call / after_turn -> return value is ignored
def noop_before_tool_call(ctx): pass   # return False here to veto the tool call
def noop_after_tool_call(ctx): pass
def noop_after_turn(ctx): pass
def noop_on_final_response(ctx): pass  # return a str here to rewrite the reply


# after_tool_call hook: right after a tool returns, LLM-compress the raw
# result down to only what's relevant to the original objective. Mutates the
# last message in place so the *agent loop itself* sees the compressed text
# on the next turn (not just downstream consumers).
def summarize_tool_result_hook(ctx):
    messages  = ctx["messages"]
    model     = ctx["model"]
    tool_call = ctx["tool_call"]   # {'name', 'arguments', 'id'}

    # handle_tool_calls appends the tool result as the last message before
    # firing after_tool_call — grab it from there.
    if not messages or messages[-1].get("role") != "tool":
        return
    tool_msg = messages[-1]
    raw = tool_msg.get("content") or ""
    if not raw or raw.startswith("Error:"):
        return  # keep errors verbatim; nothing to compress

    # Objective = the most recent user turn. Across multi-ask conversations
    # the latest user message is the live "why", so filter against that
    # rather than whatever kicked the session off.
    objective = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )

    summary = cai_llm.chat(
        [
            {"role": "system", "content": (
                "You compress tool-call results for an agent loop. Given an "
                "OBJECTIVE and a tool call's RAW RESULT, return ONLY the facts "
                "relevant to making progress on the objective. Drop boilerplate, "
                "pagination, unrelated fields, and verbose prose. Preserve "
                "concrete data: ids, names, numeric values, URLs, error reasons. "
                "Output plain text, no preamble."
            )},
            {"role": "user", "content": (
                f"OBJECTIVE:\n{objective}\n\n"
                f"TOOL CALLED: {tool_call['name']}({tool_call.get('arguments', '')})\n\n"
                f"RAW RESULT:\n{raw}"
            )},
        ],
        model=model,
    )

    if summary:
        tool_msg["content"] = summary


all_hooks = [
    ("before_tool_call",  noop_before_tool_call),
    ("after_tool_call",   noop_after_tool_call),
    ("after_tool_call",   summarize_tool_result_hook),
    ("after_turn",        noop_after_turn),
    ("on_final_response", noop_on_final_response),
    # Opt-in built-ins for context-budget management:
    # ("after_turn", mask_hook),     # mask old tool results near the budget
    # ("after_turn", compact_hook),  # summarise middle turns via an LLM call
]

# No hooks run by default. Pass hooks=all_hooks to register the no-ops above
# (plus the commented-out built-ins if you uncomment them).
harness = Harness(system_prompt="",
                  log_path="/tmp/cai/cai.log",
                  # hooks=all_hooks,
                  mcp_servers=["harness-benchmark mcp --username bob"],
                  tools =   [
                              "harness_benchmark__list_challenges",
                              "harness_benchmark__introspect_challenge",
                              "harness_benchmark__join_challenge",
                              "harness_benchmark__get_available_actions",
                              "harness_benchmark__get_objective",
                              "harness_benchmark__get_cost",
                              "harness_benchmark__perform_action",
                              "harness_benchmark__poll_events",
                              "harness_benchmark__leave_challenge",
                              "harness_benchmark__end_challenge"
                            ])

r = harness.agent(  system_prompt="",
                    prompt="try and solve the cipher decoder challenge at dificulty easy.")

# Stream events live instead of blocking on r.wait(). Event.type is one of
# "content" | "reasoning" | "tool_call" | "tool_result".
for ev in r:
    if ev.type == "reasoning":
        print(ev.text, end="", flush=True)
    elif ev.type == "content":
        print(ev.text, end="", flush=True)
    elif ev.type == "tool_call":
        print(f"\n-> {ev.tool_name}({json.dumps(ev.tool_args)})", flush=True)
    elif ev.type == "tool_result":
        marker = "x" if ev.is_error else "<-"
        print(f"   {marker} {ev.tool_name}: {ev.tool_result[:200]}", flush=True)
print()

# Local context-window estimate from the current messages[]. No tokenizer
# dependency, so use a ~4-chars-per-token heuristic (off by 10-30% for
# code/unicode — good enough for a progress indicator).
from cai.llm import get_model_profile
def _msg_text(m):
    c = m.get("content") or ""
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {})
        c += fn.get("name", "") + (fn.get("arguments") or "")
    return c
chars   = sum(len(_msg_text(m)) for m in r.messages)
est_tok = chars // 4
ctx_max = get_model_profile(harness.model).get("context", 0)
pct     = f"{est_tok / ctx_max:.1%}" if ctx_max else "?"
print(f"[context] ~{est_tok} tok / {ctx_max} ({pct})  "
      f"[{len(r.messages)} messages, {chars} chars]")

# cloned = harness.clone()
# for _ in range(3):
    # r = harness.agent(  system_prompt="you are an expert python source writer",
                        # prompt="return a list of all functions in this project",
                        # skills=['files'])
    # r.wait()
    # cloned.enrich(r.messages)

# r = cloned.agent(system_prompt="you are an expert python source writer",
                 # prompt="summerazie all found functions into one comprehensive list")
# r.wait()

# print(r.text)
# # harness.enrich(r.text)


# def add(a: int, b: int) -> int:
    # """
    # adding a to b, returning sum.
    # """
    # return a + b

# r = harness.agent(functions=[add], prompt="add 1 and 789")
# print(r.text)
