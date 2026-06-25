name: subagents
tools: cai__launch_agent, cai__wait_agent, cai__kill_agent
---
# Skill: Sub-Agents

Delegate self-contained or parallelizable subtasks to background sub-agents with
`cai__launch_agent`, then collect each result with `cai__wait_agent('<name>')`.

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
- `cai__launch_agent` returns immediately with the agent's name; the work runs
  in the background. Call `cai__wait_agent('<name>')` to block for its final
  answer. On timeout it keeps running — call `cai__wait_agent` again to keep
  waiting.
- Collect every result before giving your final answer. If a sub-agent fails,
  retry with a sharper prompt or do the work yourself.
- Abandon a runaway sub-agent with `cai__kill_agent('<name>')` (returns at
  once), then `cai__wait_agent('<name>')` to collect any partial output.
- Don't delegate trivial single-tool steps — do those yourself.
