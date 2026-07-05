"""init.py - example extension: summarize and continue (:summarize).

Branches the session for continued research: checkpoints the current session
to its .flow file, then swaps the served agent for a fresh branch seeded with
a single summary of everything so far - one LLM call, one swap. You continue
in a near-empty context that still knows what happened; the full session
stays frozen in the checkpoint, back via :sessions (or :load <path>).

The same seam as examples/extensions/clone - save + clone - with the branch's
conversation specified in the clone's spec ({'messages': [seed]}): a key
present in the spec overrides, an absent one inherits. The summary is made
BEFORE the swap so a failed LLM call leaves the session untouched."""
import json

import cai
from cai.screen.ansi import SGR_AZURE_ON_DGRAY
from cai.screen.chip import Chip


@cai.command(help="branch to a fresh session seeded with a summary of this one")
def summarize(ctx: cai.CommandContext):
    """Checkpoint the session, then continue on a summary-seeded branch."""
    messages = ctx.client.get_messages()
    if not messages:
        ctx.write("nothing to summarize yet\n")
        return
    model = ctx.client.get_info().get("model") or None
    # a hover pill while the summarisation call runs - the command blocks the
    # input line for those seconds, so show why. add_chip repaints at once;
    # the finally guarantees the pill never outlives the work.
    ctx.screen.add_chip("summarize", Chip("summarizing…", sgr=SGR_AZURE_ON_DGRAY))
    try:
        summary = _summarize(messages, model)
    finally:
        ctx.screen.remove_chip("summarize")
    if not summary:
        ctx.write("summarize failed: no summary came back; session untouched\n")
        return
    saved = ctx.client.save(None)
    if not saved:
        ctx.write("checkpoint failed: could not save the session\n")
        return
    seed = {}
    seed["role"] = "user"
    seed["content"] = ("[summary of the previous session, for continued "
                       "research]: " + summary)
    info = ctx.client.clone({"messages": [seed]})
    if not info:
        ctx.write("clone failed: the agent was not swapped\n")
        return
    ctx.write(f"checkpoint {saved}\n"
              f"continuing as {info.get('name')} on a summary of "
              f"{len(messages)} messages\n")


def _summarize(messages, model):
    """run one throwaway agent to summarise `messages`, returning its text."""
    rendered = json.dumps(messages, indent=2, default=str)
    prompt = ("Summarize the following conversation into a briefing for "
              "continuing the work in a fresh session. Preserve key facts, "
              "tool results, findings, decisions, and open questions. "
              "Output only the briefing, no preamble.\n\n" + rendered)
    run = cai.Run(messages=[{"role": "user", "content": prompt}], model=model)
    try:
        run.wait()
        return (run.text or "").strip()
    finally:
        run.close()
