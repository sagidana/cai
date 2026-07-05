"""init.py - example extension: session checkpoints (:clone).

Saves the session to its .flow file - the checkpoint - then swaps the served
agent for a fresh branch of itself over the 'clone' control op. The socket,
the client, and the conversation all stay put; only the agent (and so its
name) changes, so autosave writes the branch to a NEW .flow from here on and
the checkpoint file stays frozen at this point. Return to the checkpoint with
:sessions (or :load <path>).

The same two calls are the seam for any branch-and-continue command: pass a
spec to clone to shape the branch - e.g.
ctx.client.clone({"messages": [seed]}) to continue research on a summarized
copy of the session (see examples/extensions/summarize)."""
import cai


@cai.command(help="checkpoint the session and continue on a fresh branch")
def clone(ctx: cai.CommandContext):
    """Checkpoint the session, then continue on a fresh branch of it."""
    saved = ctx.client.save(None)
    if not saved:
        ctx.write("checkpoint failed: could not save the session\n")
        return
    info = ctx.client.clone()
    if not info:
        ctx.write("clone failed: the agent was not swapped\n")
        return
    ctx.write(f"checkpoint {saved}\ncontinuing as {info.get('name')}\n")
