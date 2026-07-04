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

## Configure

Two files under `~/.config/cai/`:

```sh
cat > ~/.config/cai/config.json <<'EOF'
{"base_url": "https://openrouter.ai/api/v1", "model": "anthropic/claude-sonnet-4"}
EOF
echo "sk-..." > ~/.config/cai/api_key
```

Nothing is defaulted: a missing or incomplete config stops cai with a clear
message. If `~/.config/cai/SYSTEM.md` or `./SYSTEM.md` exist, they are appended
to the system prompt (then `--system-prompt`, when given).

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
`:tools`, `:skills`.

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
  A server is either a `mcps/*.py` FastMCP stdio script, or declared from
  `init.py` with `cai.mcp_server` — local (spawned subprocess) or remote (URL):

  ```python
  cai.mcp_server("github",
                 command=["npx", "-y", "@modelcontextprotocol/server-github"],
                 env={"GITHUB_TOKEN": "..."})              # local stdio server

  cai.mcp_server("linear",
                 url="https://mcp.linear.app/mcp",
                 headers={"Authorization": "Bearer ..."})  # remote server
  ```
- **Skills** are markdown files: a small header (`tools:`, `skills:`) plus a
  prompt body. Activating one unions its tools into the registry and appends
  its body to the system prompt. Built in: `fs`, `fs-read-only`, `subagents`.
- The `subagents` skill gives the agent launch / wait / list / kill tools;
  each child runs on its own unix socket with a reduce-only subset of the
  parent's tools.

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

See `examples/extensions/compact` — context compaction as a `:compact` command
plus an `after_turn` auto-compact hook, in one `init.py`.

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
