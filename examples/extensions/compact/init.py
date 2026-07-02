"""init.py - example extension: context compaction (a hook and a command).

Folds the middle of a long conversation into a single [memory] message - keeping
the opening turn and the most recent exchanges verbatim - by summarising the
folded span with one LLM call. Two entry points share the same _fold logic:

  - @cai.hook("after_turn") auto_compact - runs automatically once the model's
    context window passes COMPACT_AT.
  - @cai.command compact - runs on demand when you type :compact in the TUI.

Both register the moment Environment.load() imports this file. The hook and
command live in one file because each loaded file is its own package, so they
could not share a helper module across hooks/ and commands/ subdirs."""
import json

import cai
from cai.models import ModelsRegistry


COMPACT_AT = 0.75
KEEP_RECENT = 4
DEFAULT_CONTEXT = 1_000_000


@cai.hook("after_turn")
def auto_compact(ctx: cai.HookContext):
    """Summarise older turns once the context window fills up."""
    usage = ctx.usage or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    limit = ModelsRegistry().context_length(ctx.model) or DEFAULT_CONTEXT
    if not prompt_tokens or not limit:
        return
    if prompt_tokens / limit < COMPACT_AT:
        return

    folded = _fold(ctx.messages, ctx.model)
    if folded is None:
        return
    ctx.messages[:] = folded

    agent = (ctx.data or {}).get("agent")
    if agent is not None:
        agent.set_messages(ctx.messages)
    ctx.ui.status(f"auto-compacted to {len(folded)} messages")


@cai.command(help="compact older turns into one [memory] message")
def compact(ctx: cai.CommandContext):
    """Compact older turns into one [memory] message, keeping recent ones."""
    messages = ctx.client.get_messages()
    model = ctx.client.get_info().get("model") or None
    folded = _fold(messages, model)
    if folded is None:
        ctx.write("nothing to compact yet\n")
        return
    ctx.client.set_messages(folded)
    ctx.write(f"compacted {len(messages)} messages into {len(folded)}\n")


def _fold(messages, model):
    """fold the middle of `messages` into one [memory] message, keeping the
    opening turn and the last KEEP_RECENT verbatim. returns the new list, or None
    when there is nothing worth compacting.

    the boundary is walked back off any leading tool result so an
    assistant(tool_calls) and its tool replies are never split across the fold -
    providers reject an orphaned tool result."""
    start = 0
    if start < len(messages) and messages[start].get("role") == "user":
        start += 1
    end = max(start, len(messages) - KEEP_RECENT)
    while end > start and messages[end].get("role") == "tool":
        end -= 1
    middle = messages[start:end]
    if len(middle) < 2:
        return None

    summary = _summarize(middle, model)
    if not summary:
        return None
    folded = {}
    folded["role"] = "assistant"
    folded["content"] = "[memory from compacted turns]: " + summary
    return messages[:start] + [folded] + messages[end:]


def _summarize(messages, model):
    """run one throwaway agent to summarise `messages`, returning its text. the
    Run makes no tool calls, so an after_turn hook never fires on it - the
    summarisation never recurses back into auto_compact."""
    rendered = json.dumps(messages, indent=2, default=str)
    prompt = ("Summarize the following conversation turns into a concise memory "
              "entry. Preserve key facts, tool results, findings, and decisions. "
              "Output only the summary, no preamble.\n\n" + rendered)
    run = cai.Run(messages=[{"role": "user", "content": prompt}], model=model)
    try:
        run.wait()
        return (run.text or "").strip()
    finally:
        run.close()
