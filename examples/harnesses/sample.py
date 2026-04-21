import json

from cai import Harness, mask_hook, compact_hook


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

all_hooks = [
    ("before_tool_call",  noop_before_tool_call),
    ("after_tool_call",   noop_after_tool_call),
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
                    prompt="try and solve the cipher decoder challenge at dificulty easy.",
                    tools=["list_challenges", "introspect_challenge", "join_challenge", "get_available_actions", "get_objective", "get_cost", "perform_action", "poll_events", "leave_challenge", "end_challenge"])
r.wait()

print(r.text)

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
