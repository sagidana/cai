name: subagents
tools: launch_agent, wait_agent, kill_agent
---
# Skill: Sub-Agents

Delegate self-contained or parallelizable subtasks to background sub-agents,
then collect each result by name.

## Calling the tools

Pass arguments as named fields, exactly as shown:

- `launch_agent(prompt="<self-contained task>", name="audit-auth-flow")` starts
  ONE sub-agent and returns its `agent_id` (the name). Optional: `tools=[...]`,
  `skills=[...]`, `model="..."`, `system_prompt="..."`.
- `wait_agent(agent_id="audit-auth-flow")` blocks for ONE sub-agent's final
  answer.
- `kill_agent(agent_id="audit-auth-flow")` stops ONE runaway sub-agent.

`agent_id` is a single string — the exact name `launch_agent` returned — and you
pass ONE per call, never a list. To wait on several sub-agents, make several
`wait_agent` calls, one `agent_id` each. Always supply `agent_id`; `wait_agent`
and `kill_agent` do nothing useful without it.

## Rules

- Name every agent descriptively — lowercase words delimited with dashes:
  `audit-auth-flow`, `summarize-test-logs`.
- Prompts must be self-contained. The sub-agent shares your working directory
  but sees nothing of your conversation, so include paths, constraints, and the
  expected output format in the prompt.
- Grant only what the task needs via `tools`/`skills` — each a list of names,
  and only ever a subset of your own. Nothing is inherited by default: a
  sub-agent starts with no tools and no skills unless you pass them explicitly.
- `skills` takes registry names — lowercase dash-delimited file names like
  `fs-read-only`, never a skill's display title. When unsure, prefer `tools`
  with exact tool names.
- `launch_agent` returns immediately with the agent's name; the work runs in the
  background. Call `wait_agent(agent_id="<name>")` to block for its final answer.
  On timeout it keeps running — call `wait_agent` again to keep waiting.
- Collect every result before giving your final answer. If a sub-agent fails,
  retry with a sharper prompt or do the work yourself.
- Abandon a runaway sub-agent with `kill_agent(agent_id="<name>")` (returns at
  once), then `wait_agent(agent_id="<name>")` to collect any partial output.
- Don't delegate trivial single-tool steps — do those yourself.
