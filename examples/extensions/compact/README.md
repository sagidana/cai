# compact — an example cai extension

Adds a `:compact` command that folds the whole conversation into a single
`[memory]` message by summarising it with one LLM call. Use it in the
interactive TUI when the context window is getting full.

## Layout

```
compact/
└── commands/
    └── init.py      # @cai.command def compact(ctx): ...
```

A cai extension is a self-contained bundle. `commands/init.py` is one of the
files cai imports at startup; the `@cai.command` decorator in it registers the
command into the global `CommandsRegistry`, so it shows up as `:compact`.

## Install

Copy the bundle into your extensions dir:

```sh
cp -r examples/extensions/compact ~/.config/cai/extensions/
```

Then start cai. `:compact` now tab-completes after `:` and appears in the
Ctrl-P palette.

## Verify it loaded (no API call)

```sh
python3 -c "
from cai import userconfig
from cai.commands import CommandsRegistry
userconfig.load()
print(sorted(CommandsRegistry.commands()))
"
```

`compact` should be in the list.

## How it works

- `cai.command` registers the function; the TUI dispatches `:compact` to it with
  a `CommandContext`.
- `ctx.client.get_messages()` reads the live conversation; `get_info()` gives the
  current model.
- `cai.Run(...)` makes one throwaway summarisation call through the SDK.
- `ctx.client.set_messages([...])` replaces the conversation with the summary, so
  the next turn continues from it (and autosave persists it).

## Notes

- **TUI only.** `:`-commands run in the interactive TUI; headless `cai -p "…"`
  doesn't have a `:`-dispatch (though it still loads extensions for hooks).
- **It blocks while summarising.** The command runs on the TUI's main thread, so
  the input line is frozen for the few seconds the summarisation call takes.
- **The scrollback isn't rewritten.** The conversation *state* is compacted
  (visible in `:messages` and used for the next turn); the text already printed
  above stays on screen.
- For *automatic* compaction once the window fills up, register an `after_turn`
  hook with `@cai.hook` instead — same idea, driven by context usage rather than
  on demand.
