name: subagents
tools: launch_agent, wait_agent, list_agents, kill_agent
---
# Skill: Sub-Agents

Delegate self-contained or parallelizable subtasks to background sub-agents, then collect each result by name. Don't delegate trivial single-tool steps — do those yourself.

The tools (pass arguments as named fields):
- `launch_agent(prompt="...", name="audit-auth-flow")` — start ONE sub-agent, returns its `agent_id` (the name). Optional: `tools=[...]`, `skills=[...]`, `model="..."`, `system_prompt="..."`.
- `wait_agent(agent_id="audit-auth-flow")` — block for ONE sub-agent's answer. A finished agent keeps its answer parked, so a late call still returns it.
- `list_agents()` — every sub-agent's status (running / finished / failed), without blocking.
- `kill_agent(agent_id="audit-auth-flow")` — stop ONE runaway sub-agent.

`agent_id` is a single string — the exact name `launch_agent` returned — ONE per call, never a list. To wait on several, make several `wait_agent` calls.

Rules:
- Name every agent descriptively, lowercase-dashed: `audit-auth-flow`, `summarize-test-logs`.
- Prompts must be self-contained — the sub-agent shares your working directory but sees nothing of your conversation. Include paths, constraints, and the expected output format.
- Grant only what the task needs via `tools`/`skills` (each a list, only a subset of your own). Nothing is inherited by default.
- `skills` takes registry names (`fs-read-only`), never display titles. When unsure, prefer `tools` with exact tool names.
- Collect every result before your final answer. If one fails, retry with a sharper prompt or do it yourself.
