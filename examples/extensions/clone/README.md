# clone — an example cai extension

Session checkpoints: **`:clone`** saves the session to its `.flow` file (the
checkpoint), then swaps the served agent for a fresh branch of itself. You keep
typing into the same conversation, but under a new agent name — so autosave
writes the branch to a **new** `.flow` from here on and the checkpoint file
stays frozen at this point. Return to the checkpoint with `:sessions` (or
`:load <path>`).

## Layout

```
clone/
├── README.md
└── init.py      # @cai.command clone
```

A cai extension is a self-contained bundle. cai imports the bundle's `init.py`
at startup; the `@cai.command` decorator registers `:clone` onto the
Environment being loaded.

## Install

```sh
cp -r examples/extensions/clone ~/.config/cai/extensions/
```

Start cai: `:clone` tab-completes after `:` and is in the Ctrl-P palette.

## How it works

Two control ops over the wire client, nothing else:

- `ctx.client.save(None)` — persist the session to `<name>.flow` (the
  checkpoint).
- `ctx.client.clone()` — the `clone` op: the server swaps its agent for an
  `Agent.clone()` of it behind the same socket. The conversation, model,
  tools, and skills carry over; the retired agent is closed after handing its
  scratch directory to the branch. Nothing migrates on the client side.

The swap is a deferred op — it applies between turns, never racing a run.

## Notes

- **This is the seam for any branch-and-continue command.** `clone` takes an
  optional spec of the branch's nameable state (a key present overrides, an
  absent one inherits): `ctx.client.clone({"messages": [seed]})` is
  summarize-and-continue (see `examples/extensions/summarize`), and
  `ctx.client.clone({"messages": []})` is a fresh session that keeps the
  model/tools/skills.
- Sent mid-run, the op (like `:compact`'s `set_messages`) waits for the turn
  to finish; the input line blocks until then.
- The type is annotated (`ctx: cai.CommandContext`), so your editor can jump
  from `ctx.<field>` into its definition.
