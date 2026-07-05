# summarize — an example cai extension

Summarize and continue: **`:summarize`** checkpoints the session, swaps the
served agent for a fresh branch of itself, and replaces the branch's
conversation with a single summary of everything so far — one LLM call. You
keep working in a near-empty context that still knows what happened; the full
session stays frozen in the checkpoint, back via `:sessions` (or
`:load <path>`).

Where `:compact` folds the middle of a conversation in place and `:clone`
branches it whole, `:summarize` branches it *reduced* — the shape for long
research sessions: checkpoint the detail, carry only the briefing forward.

## Layout

```
summarize/
├── README.md
└── init.py      # @cai.command summarize + _summarize
```

## Install

```sh
cp -r examples/extensions/summarize ~/.config/cai/extensions/
```

Start cai: `:summarize` tab-completes after `:` and is in the Ctrl-P palette.

## How it works

The branch-and-continue seam from `examples/extensions/clone`, with the
branch shaped in the clone's spec:

- `_summarize(...)` — one throwaway `cai.Run` turns the conversation into a
  briefing. Runs FIRST, so a failed LLM call leaves the session untouched.
- `ctx.client.save(None)` — persist the full session to `<name>.flow` (the
  checkpoint).
- `ctx.client.clone({"messages": [seed]})` — the `clone` op: the server swaps
  its agent for a branch behind the same socket, its conversation replaced by
  the one seed message per the spec (a key present overrides, an absent one
  inherits — model, tools, skills, and the rest carry over). Autosave
  continues under the branch's new name, so the checkpoint file stays frozen.

## Notes

- **The command is TUI-only**, runs on the main thread, and blocks the input
  line for the few seconds the summarisation takes (same as `:compact`). A
  `summarizing…` hover pill (`ctx.screen.add_chip` with a `screen.chip.Chip`)
  shows over the conversation for that stretch, removed in a `finally` so it
  never outlives the work.
- The scrollback already printed isn't rewritten; the conversation *state* is
  what's replaced (visible in `:messages` and used for the next turn).
- The type is annotated (`ctx: cai.CommandContext`), so your editor can jump
  from `ctx.<field>` into its definition.
