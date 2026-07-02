# compact — an example cai extension

Context compaction in two flavours, sharing one folding routine:

- **`:compact`** — a command. Type it in the TUI to fold older turns into a
  single `[memory]` message on demand.
- **`auto_compact`** — an `after_turn` hook. Does the same automatically once the
  model's context window crosses `COMPACT_AT` (75%).

Both keep the opening turn and the most recent `KEEP_RECENT` (4) messages
verbatim, summarise everything in between with one LLM call, and splice the
summary back in its place.

## Layout

```
compact/
├── README.md
└── init.py      # @cai.hook auto_compact + @cai.command compact + shared _fold
```

A cai extension is a self-contained bundle. cai imports the bundle's `init.py`
(also `hooks/init.py` and `commands/init.py`) at startup; the `@cai.hook` and
`@cai.command` decorators register onto the Environment being loaded. The hook
and command live in one `init.py` here because each loaded file is its own
package — they could not share `_fold` across separate `hooks/` and `commands/`
dirs.

## Install

```sh
cp -r examples/extensions/compact ~/.config/cai/extensions/
```

Start cai: `:compact` tab-completes after `:` and is in the Ctrl-P palette; the
hook compacts on its own as the window fills.

## Verify it loaded (no API call)

```sh
python3 -c "
from cai.environment import Environment
env = Environment().load()
print('commands:', sorted(env.commands()))
print('hooks:', [(e, f.__name__) for e, f in env.hooks()])
"
```

Expect `['compact']` and `[('after_turn', 'auto_compact')]`.

## How it works

- `_fold(messages, model)` chooses the slice to compact (opening + last 4 kept),
  walks the boundary back so an `assistant(tool_calls)` and its tool replies are
  never split, and returns the new message list (or `None` when there's nothing
  worth folding).
- `cai.Run(...)` makes the one summarisation call. It issues no tool calls, so an
  `after_turn` hook never fires on it — the summarisation can't recurse back into
  `auto_compact`.
- The command writes via `ctx.client.set_messages(...)`; the hook mutates
  `ctx.messages` in place, notifies the agent so autosave persists it, and posts
  progress with `ctx.ui.status(...)` (the TUI status line; a no-op headless).

## Notes

- **The command is TUI-only**, runs on the main thread, and blocks the input line
  for the few seconds the summarisation takes. The hook runs on the worker thread
  mid-run.
- **`after_turn` fires after tool-using rounds**, which is where context grows
  fastest. To also compact a pure-chat session (no tools), add `after_run`:
  `@cai.hook("after_run")` on the same function — the `_fold` "middle < 2" guard
  keeps the summarisation's own short run from compacting itself.
- The scrollback already printed isn't rewritten; the conversation *state* is
  what's compacted (visible in `:messages` and used for the next turn).
- These types are annotated (`ctx: cai.HookContext`, `ctx: cai.CommandContext`),
  so your editor can jump from `ctx.<field>` into its definition.
