# cai

A small LLM agent, built from scratch layer by layer. One package gives you a
CLI, a full-screen TUI, and a Python SDK — over any OpenAI-compatible endpoint.

```
Layer 0: cai.api    - the OpenAI-compatible HTTP client (the LLM call)
Layer 1: cai.llm    - the core agentic loop (call_llm)
Layer 2: cai.agent  - Agent (persistent conversation) + Run (one-shot execution)
Entry:   cai.cli    - the `cai` command; drops into the TUI (cai.tui) when interactive
```

## Install

```sh
make install        # pip install .
make dev            # pip install -e .[dev]
```

Requires Python >= 3.11.

## Configuration

Everything lives under `~/.config/cai/`. Two files are required:

```sh
cat > ~/.config/cai/config.json <<'EOF'
{"base_url": "https://openrouter.ai/api/v1", "model": "anthropic/claude-sonnet-4"}
EOF
echo "sk-..." > ~/.config/cai/api_key
```

Nothing is defaulted: a missing or incomplete config stops cai with a clear
message. If `~/.config/cai/SYSTEM.md` or `./SYSTEM.md` exist, they are appended
to the system prompt (then `--system-prompt`, when given).

### init.py

`~/.config/cai/init.py` is optional Python imported on every load, after the
extensions — so its registrations win. Everything the SDK registers works
here: MCP servers, settings, tools, hooks, commands.

```python
import cai

# MCP servers - local (spawned stdio subprocess) or remote (URL)
cai.mcp_server("github",
               command=["npx", "-y", "@modelcontextprotocol/server-github"],
               env={"GITHUB_TOKEN": "..."})
cai.mcp_server("linear",
               url="https://mcp.linear.app/mcp",
               headers={"Authorization": "Bearer ..."})

# settings - the same object the TUI's :config overlay edits
cai.settings.show_reasoning = False
cai.settings.tool_result_max_chars = 20_000
cai.settings.auto_save_sessions = True
cai.settings.skills.append("fs")                  # auto-activated on CLI runs
cai.settings.tools.append("github__search_issues")

# any config.json field can be shadowed from here: a non-None cai.settings
# attribute of the same name wins over the file (model, base_url, ssl_verify,
# default_context_size, python_base, python_sandbox, python_venv).
cai.settings.model = "anthropic/claude-sonnet-4"
cai.settings.python_venv = "~/.pyenv/versions/cai-tools"  # run the python tool
                                                          # under your own env

# tools / hooks / commands - same decorators as the SDK
@cai.tool
def shout(text: str) -> str:
    """upper-case text."""
    return text.upper()

@cai.hook("after_turn")
def log_turn(ctx):
    print(f"turn done, {len(ctx.messages)} messages")

@cai.command(name="clear", help="wipe the conversation")
def clear(ctx):
    ctx.client.set_messages([])
```

## CLI

```sh
cai -- explain this error            # prompt after '--'
git diff | cai -- write a commit message   # piped stdin becomes context
cai --file main.py -- find the bug
cai --skill fs -- rename foo to bar in src/
cai -t fs__read_file -- summarize /etc/hosts
cai --strict-format json -- list the planets as a JSON array
```

The prompt goes after `--` (or via `-p`) so `--skill`/`--tool` can take several
values without swallowing it. When stdout is piped, progress goes to stderr and
only the clean answer is printed. LLM knobs: `--model`, `--reasoning-effort`,
`--temperature`, `--max-steps`, `--non-streaming`, `--cwd`.

## TUI

`cai` with no prompt (and a terminal attached) opens the full-screen
interactive TUI; `-i` forces it. It is vim-modal, with `:`-commands for the
session: `:models`, `:messages`, `:history`, `:sessions`, `:save`, `:load`,
`:tools`, `:skills`, `:redraw` (repaint the view from the conversation, e.g.
after flipping *show reasoning* in `:config`).

```sh
cai                 # new interactive session
cai -c              # resume the most recent saved session
cai --sessions      # pick a saved session to resume
```

Sessions are saved as `.flow` files — a small JSON document holding the
conversation plus the settings needed to resume it.

## Skills, tools, sub-agents

- **Function tools** are plain Python callables registered with `@cai.tool`.
- **MCP tools** come from MCP servers, named `<server>__<tool>` so two
  servers can each expose a `search` without colliding. `fs` ships built in.
  A server is either a `mcps/*.py` FastMCP stdio script, or declared with
  `cai.mcp_server` (see [Configuration](#initpy)) — local or remote.
- **Skills** are markdown files: a small header (`tools:`, `skills:`) plus a
  prompt body. Activating one unions its tools into the registry and appends
  its body to the system prompt. Built in: `fs`, `fs-read-only`, `subagents`.
- The `subagents` skill gives the agent launch / wait / list / kill tools;
  each child runs on its own unix socket with a reduce-only subset of the
  parent's tools.
- The `python` skill gives the agent a `python(code, timeout=60)` tool that
  runs a snippet in a subprocess of a cai-managed virtualenv
  (`~/.config/cai/venv/`, created on first use, empty by default — stdlib only;
  manage its packages with `cai python install|uninstall|list-packages`).
  The snippet is jailed at the **kernel level**: it enters fresh user + mount +
  network namespaces and pivots onto a root containing only the working
  directory, the session scratch dir, the interpreter and the system library
  dirs its C extensions load from — no other path exists, and there is no
  network interface. The whole tree is mounted
  **read-only except the scratch dir**, the one writable island. On top of
  that, a `sys.addaudithook` jail enforces the same policy: it can read files
  and list directories inside the jail but create, modify or delete only under
  scratch, and subprocess/`ctypes`/`cffi` are blocked. The snippet also gets a
  `tool_call(name, **kwargs)` builtin that dispatches the agent's *own* selected
  tools in-process — through the same `before_tool_call` gates — so a script
  can read a large tool result, reduce it in Python, and return only the
  answer, the intermediate data never entering model context (and any file
  changes go through a write tool like `fs__create_file`, under its own gate).
  Point it at a different base interpreter with the optional `python_base` key
  in `config.json`, or run it under an existing virtualenv of your own (e.g. a
  `pyenv` one) with `python_venv` — cai never builds, rebuilds or deletes a
  user-supplied env. On hosts that forbid unprivileged user namespaces (e.g.
  default-hardened Docker) the tool fails closed — there the optional
  `python_sandbox` key set to `"hook"` runs the audit-hook jail only, the
  container itself being the boundary. Any of these keys can also be set from
  `init.py` (`cai.settings.python_venv = "…"`), which shadows `config.json`.

## Extensions

An extension is a self-contained bundle directory carrying any of
`skills/*.md`, `tools/*.py`, `mcps/*.py`, `init.py`, `hooks/init.py`,
`commands/init.py`, and a `README.md`. Installing drops it under
`~/.config/cai/extensions/<name>/`, where `Environment.load()` discovers it.

```sh
cai extend ./my-bundle              # install a folder, .zip, or http(s) URL
cai extend --list
cai extend --remove my-bundle
```

See `examples/extensions/`:

- `compact` — context compaction as a `:compact` command plus an `after_turn`
  auto-compact hook, in one `init.py`.
- `clone` — `:clone`, a session checkpoint: save the session, then swap the
  served agent for a fresh branch of itself and keep going.
- `summarize` — `:summarize`, branch and continue reduced: checkpoint, branch,
  and carry forward only a one-message summary of the session.

## SDK

```python
import cai

# one-shot
run = cai.Run(messages=[{"role": "user", "content": "hello"}])
print(run.wait().text)

# persistent conversation, streaming, with tools
agent = cai.Agent(skills=["fs"])
for event in agent.run("what's in ./src?"):
    if event.type == cai.EventType.CONTENT:
        print(event.text, end="", flush=True)
agent.save("session.flow")
```

Tools are explicit: `tools=` takes Python callables or MCP tool-name strings;
`skills=` takes skill names; `hooks=` takes `(event, fn)` pairs. Only what you
pass is sent to the model. An `Agent` is constructed against an `Environment`
(`env=`), so two agents in one process can see two different installs — and a
test builds a private empty one instead of resetting globals.

```python
@cai.tool
def word_count(text: str) -> int:
    """count the words in text."""
    return len(text.split())

@cai.hook("before_tool_call")
def veto(ctx):
    if ctx.tool_call.name == "fs__write_file": return False

@cai.command
def stats(ctx):
    ctx.write(f"{len(ctx.client.get_messages())} messages")
```

## Development

```sh
make test           # pytest -q
make clean
```

Logs go to `/tmp/cai.log`.
