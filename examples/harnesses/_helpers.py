"""Shared helpers for the SDK port of harnesses/*.harness.cai.

These files are 1:1 translations of each .harness.cai into plain Python driving
the cai SDK. Harness runtime concepts map to SDK idioms like this:

  --enrich full         →  h.enrich(result.messages)      # replace messages
  --enrich result-only  →  h.enrich(result.text)          # append assistant turn
  --enrich none         →  (skip — don't call enrich)
  --prepend-user-prompt →  prepend the user task to the block prompt
  --tools a, b, c       →  agent(tools=["a", "b", "c"])
  --mcp "cmd …"         →  Harness(mcp_servers=["cmd …"])
  --system-prompt "…"   →  agent(system_prompt="…")
  --strict-format regex →  agent(strict_format="regex:<pattern>")
  --strict-format "…" + 1-word gate →  h.gate(["ok", "retry"], "prompt")
  if x == y: goto L     →  ordinary Python if/elif → function call
  for-each x in B: …    →  ordinary for-loop over B.text.splitlines()

Things the v1 SDK deliberately does not surface (so the ports omit them):
  --max-turns            — let llm.py run to completion
  --force-tools          — rely on prompt wording
  --reasoning-effort     — not exposed in agent
  compact-if-more-than   — llm.py auto-compacts via config
"""
from __future__ import annotations

import sys


def get_task() -> str:
    """Grab the user's task from argv or print usage and exit."""
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} \"<task>\"", file=sys.stderr)
        sys.exit(2)
    return " ".join(sys.argv[1:])


def prepend_task(task: str, block_prompt: str) -> str:
    """Mimic --prepend-user-prompt: the user's task is stapled before the block."""
    return f"User task:\n{task}\n\n{block_prompt}"
