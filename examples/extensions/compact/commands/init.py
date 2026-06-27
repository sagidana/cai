"""commands/init.py - example: the :compact command.

Folds the whole conversation into a single [memory] message by summarising it
with one LLM call through the cai SDK. Type :compact in the TUI when the context
gets long; the next turn continues from the summary.

Written against the current command surface - a CommandContext exposes ctx.args
(text after the name), ctx.client (the agent client: get_messages / set_messages
/ get_info), and ctx.write (a transcript line). cai.command registers it into the
global CommandsRegistry the moment cai.userconfig.load() imports this file."""
import json

import cai


@cai.command(help="summarise the whole conversation into one [memory] message")
def compact(ctx: cai.CommandContext):
    """Summarise the whole conversation into one [memory] message."""
    messages = ctx.client.get_messages()
    if len(messages) < 2:
        ctx.write("nothing to compact\n")
        return

    ctx.write(f"compacting {len(messages)} messages\n")
    model = ctx.client.get_info().get("model") or None
    summary = _summarize(messages, model)
    if not summary:
        ctx.write("compaction produced nothing (check API key / model)\n")
        return

    folded = {}
    folded["role"] = "assistant"
    folded["content"] = "[memory]: " + summary
    ctx.client.set_messages([folded])
    ctx.write("compacted into 1 message\n")


def _summarize(messages, model):
    """run one throwaway agent to summarise `messages`, returning its text."""
    rendered = json.dumps(messages, indent=2, default=str)
    prompt = ("Summarize the following conversation into a concise memory entry. "
              "Preserve key facts, decisions, and tool results. Be specific. "
              "Output only the summary, no preamble.\n\n" + rendered)
    run = cai.Run(messages=[{"role": "user", "content": prompt}], model=model)
    try:
        run.wait()
        return (run.text or "").strip()
    finally:
        run.close()
