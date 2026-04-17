# cai — Command-line AI

`cai` is a lightweight CLI tool and Python SDK that brings LLM intelligence into your terminal, editor, and scripts. It supports plain prompts, cursor-aware code generation, line-by-line batch processing, skills, MCP tool integration, a vim-modal interactive TUI, a hierarchical log viewer, and a programmatic `Harness` SDK — all powered by any OpenAI-compatible API (defaults to [OpenRouter](https://openrouter.ai)).

---

## Features

- **One-shot prompts** — `cai -p "..."`, with optional file, stdin, cursor location, and tool context.
- **Interactive vim-modal TUI** — `cai -i` opens a persistent chat session with NORMAL / INSERT / VISUAL / VISUAL_LINE / COMMAND / SEARCH modes and inline tool-call streaming.
- **Log viewer** — `cai --logger` tails the hierarchical JSONL log with folding, search, and yank.
- **Skills** — `--skill files web adb ...` activates curated tool + prompt bundles.
- **Task modes** — `--mode research|dev` tunes the system prompt toward investigation or implementation.
- **MCP tools** — add any external MCP server via `--mcp <command>`; tool names are namespaced as `{label}__{name}`.
- **Programmatic SDK** — `from cai import Harness, Result, Event` for building agents in Python.
- **Flow save/load** — `:save` / `:load` in the TUI, or `Harness.save()` / `Harness.load()` in code, persist conversation + settings as JSON.
- **Strict output formats** — force JSON, regex, or regex-per-line responses with automatic retry.
- **Model profiles** — built-in capability map (tier, context window, tool-calling) for 20+ models; user-overridable in `config.json`.
- **Context compaction** — older turns get summarised automatically when usage crosses the threshold.
- **Line-by-line batch mode** — per-line LLM calls with `--cores N` parallelism.
- **Tab completion** — auto-installs for bash, zsh, and fish on first run.

---

## Installation

```bash
git clone https://github.com/youruser/cai
cd cai
pip install -e .
```

### Configuration

Config is auto-created on first run under `~/.config/cai/`. Add your API key:

```bash
echo "sk-or-..." > ~/.config/cai/api_key
```

`~/.config/cai/config.json` defaults are created and backfilled automatically:

```json
{
  "base_url": "https://openrouter.ai/api/v1",
  "model": "arcee-ai/trinity-mini:free",
  "prompt_mode": "local",
  "observation_mask_pct": 0.60,
  "observation_mask_keep": 3,
  "context_budget_pct": 0.75,
  "tool_result_max_chars": 40000,
  "ssl_verify": true,
  "stuck_detection": false,
  "model_profiles": { "...": "..." }
}
```

Key settings:

- `prompt_mode` — `local` (default) or `sota`. Picks the base system prompt file from `src/cai/prompts/`.
- `context_budget_pct` — fraction of the context window before compaction kicks in.
- `tool_result_max_chars` — cap on a single tool-call result before dynamic trimming.
- `model_profiles` — override or add model capability entries.

User skills live at `~/.config/cai/skills/*.md` and override same-named built-ins.

---

## CLI Usage

### Basic prompt

```bash
cai -p "Explain what a closure is in Python"
# or trailing words after --:
cai -- Explain what a closure is in Python
```

### Pipe stdin as context

```bash
cat error.log | cai -p "What is causing this error?"
git diff | cai -p "Write a commit message for this diff"
```

### Include a file

```bash
cai --file ./src/cai/cli.py -p "What does this file do?"
```

### Cursor-aware generation

```bash
cai --cursor "./mymodule.py:42:4" -p "Implement this method"
```

The file is included with the cursor position marked — great for editor integrations.

### Interactive TUI

```bash
cai -i
cai -i --file ./src/cai/cli.py -p "Walk me through this file"
cai -i --skill files web --mode dev
```

### Skills

```bash
cai --skill files -p "Map the codebase"
cai --skill files web -p "Compare our implementation to the official docs"
```

Built-in skills: `adb`, `files`, `frida`, `smali`, `web`. Each declares a tool set and a prompt fragment (see `src/cai/skills/*.md`). Skills can also be layered at runtime via `:skill` inside the TUI.

### Task mode

```bash
cai --mode research -p "Why does the auth flow retry twice?"
cai --mode dev -p "Implement a rate limiter middleware" --file ./app.py
```

`--mode` appends a focus block to the system prompt (`research` or `dev`). Default: `research`.

### External MCP server

```bash
cai --mcp "python /path/to/my_mcp_server.py" -p "Use the custom tool"
cai --mcp "npx @modelcontextprotocol/server-filesystem /" -p "List project files"
cai --mcp "uvx mcp-server-git" -p "Summarise recent commits"
```

### Enable internal tools by name

```bash
cai -t cai__generic_linux_command -p "Check disk usage"
cai -t cai__fetch_codebase_metadata -p "Map the entire codebase"
```

Tool names are prefixed with the server label (`cai__`, `my-server__`, etc.).

### Batch mode (per line)

```bash
cat list_of_questions.txt | cai --line-by-line --cores 8 -p "Answer briefly"
```

Each line becomes a separate LLM call, parallelised with `--cores`.

### Strict output format

`--strict-format` retries (up to 4x) until the response matches, feeding the validation error back to the model.

```bash
cai --strict-format json -p "List 3 refactoring suggestions" --file ./cli.py
cai --strict-format "regex:^(yes|no)$" -p "Is this code thread-safe?" --file ./worker.py
cai --strict-format "regex-each-line:^\w+" -p "List function names" --file ./api.py
```

### Resume a saved session

```bash
cai --context ~/flows/session.flow -p "Continue where we left off"
```

Loads messages, skills, tools, and model from a `.flow` file written by `:save` or `Harness.save()`.

### Log viewer

```bash
cai --logger
cai --logger --log-path /tmp/cai/my-run.log
```

### Custom model + system prompt

```bash
cai --model "openai/gpt-4o" --system-prompt "You are a senior Go engineer." -p "Review this code" --file ./main.go
cai --system-prompt-file ./prompts/reviewer.md -p "Review this diff" --file ./diff.patch
cai --naked -p "raw prompt, no defaults"
```

### One-liner output

```bash
cai --oneline -p "Summarise this function in one sentence" --file ./cli.py
```

---

## CLI Flags

| Flag | Purpose |
|------|---------|
| `-p` / `--prompt` / `-- <words>` | The prompt |
| `-i` / `--interactive` | Vim-modal TUI chat session |
| `--logger` | Launch the hierarchical log viewer |
| `--log-path PATH` | Log file to write and to view (default `/tmp/cai/cai.log`) |
| `--file PATH` | Include a file in context |
| `--cursor file:line:col` | Cursor-aware code generation |
| `--mode research\|dev` | Task-focus hint prepended to the system prompt |
| `--skill NAME [NAME ...]` | Activate one or more skills |
| `--model` | Override model for this run |
| `--system-prompt` | Override the base system prompt |
| `--system-prompt-file PATH` | Load base system prompt from a file |
| `--naked` | No default system prompt (overridden by `--system-prompt*`) |
| `-t` / `--tools NAME [...]` | Enable specific tools (prefixed names, e.g. `cai__search`) |
| `--mcp CMD [CMD ...]` | Launch external MCP servers |
| `--force-tools` | Require at least one tool call (`tool_choice=required`) |
| `--max-turns N` | Cap tool-call turns (default: unlimited) |
| `--strict-format FMT` | `json`, `regex:<pat>`, or `regex-each-line:<pat>` |
| `--reasoning-effort high\|medium\|low` | Extended thinking via OpenRouter |
| `--temperature FLOAT` | Sampling temperature (not all models) |
| `--non-streaming` | Use blocking API instead of streaming |
| `--oneline` | Collapse output to a single line |
| `--line-by-line` | Process stdin/file one line at a time |
| `--cores N` | Parallel threads for batch mode |
| `--context PATH` | Resume from a `.flow` file |

---

## Interactive Mode

`cai -i` opens a persistent full-screen TUI implemented with raw ANSI (no external TUI dependencies). It uses a **vim-style modal editor** for the input line.

**Layout:**
- Scrollable conversation view (assistant output streams in).
- Status bar: `MODE | model | ctx XX% (used/total)`.
- Input area at the bottom; mode is shown in the status bar.

### Modes

| Mode | Purpose |
|------|---------|
| `INSERT` (default) | Free text entry. Backspace, Ctrl-W, Ctrl-U, etc. work as usual. |
| `NORMAL` | Vim-style motions (`h j k l w b e 0 $ gg G`), operators (`d c y p`), undo (`u`). |
| `VISUAL` | Character-wise selection. |
| `VISUAL_LINE` | Line-wise selection (`V`). |
| `COMMAND` | `:` commands (see below). |
| `SEARCH` | `/pattern` forward, `?pattern` backward; `n`/`N` cycle matches. |

`Esc` returns to NORMAL from any other mode.

### Input keybindings

| Key | Action |
|-----|--------|
| Arrow keys | Move cursor / browse history |
| Home / End, Ctrl-A / Ctrl-E | Line start / end |
| Ctrl-U / Ctrl-D | Scroll the conversation half a page |
| Page Up / Page Down | Scroll full page |
| Ctrl-K | Kill to end of line |
| Ctrl-W | Delete word before cursor |
| `\` + Enter, Alt-Enter | Insert newline (multi-line input) |
| Ctrl-V | Open current prompt in `nvim` |
| Enter | Submit |
| Ctrl-C | Interrupt stream / clear input / exit if empty |
| Ctrl-D | Exit |

Scrolling works during streaming without interrupting the LLM response.

### `:` commands

Type `:` on an empty INSERT-mode line (or from NORMAL) to enter COMMAND mode. Tab completion is available.

| Command | Action |
|---------|--------|
| `:compact` | Summarise older turns into a compact memory message |
| `:clear` | Clear history (keep the system prompt) |
| `:tools` | Open the tools toggle overlay (Space to toggle, `/` to search) |
| `:context` | Open the context-inspector overlay (browse / edit per-message tokens) |
| `:skill` | Show active + available skills |
| `:skill NAME [NAME ...]` | Activate skills (append) |
| `:skill off` | Deactivate all skills |
| `:model` | Open the live-model picker overlay |
| `:save PATH` | Write the current session to a `.flow` file |
| `:load PATH` | Load a `.flow` file into this session |

The `:tools` overlay lists every tool from every registered MCP server with its origin label (`[cai]`, `[myserver]`). Navigate with `j`/`k`, toggle with Space, search with `/` or `?`, Enter/Esc to close.

Interactive mode always uses agentic tool-calling. Tool calls and results are shown inline in dim gray.

---

## Python SDK

`cai` exposes a small programmatic surface for building agents in-process:

```python
from cai import Harness, Result, Event
```

### Minimal example

```python
from cai import Harness

with Harness(system_prompt="", log_path="/tmp/cai/cai.log") as h:
    r = h.agent(
        system_prompt="return a list of items only",
        prompt="list all functions in this project",
        strict_format=r"regex-each-line:^(-).*$",
        skills=["files"],
    )
    r.wait()

    for fn in r.text.splitlines():
        verdict = h.gate(
            options=["yes", "no"],
            skills=["files"],
            prompt=f"does '{fn}' touch the file system?",
        )
        print(f"{fn} -> {verdict}")
```

### Streaming example

```python
from cai import Harness

with Harness(name="explorer", log_path="/tmp/cai/explorer.log") as h:
    r = h.agent(
        skills=["files"],
        system_prompt="You are a senior engineer exploring a codebase.",
        prompt="what are the main classes in src/cai/?",
        name="explore",
    )
    for event in r:
        if event.type == "content":
            print(event.text, end="", flush=True)
        elif event.type == "reasoning":
            print(event.text, end="", flush=True)
        elif event.type == "tool_call":
            print(f"\n→ {event.tool_name}({event.tool_args})")
        elif event.type == "tool_result":
            status = "error" if event.is_error else "ok"
            print(f"← [{status}] {(event.tool_result or '')[:200]}")
    print(f"\n--- {r.finish_reason} ---")
```

### `Harness`

```python
Harness(
    system_prompt: str | None = None,
    skills: list[str] | None = None,
    tools: list[str] | None = None,
    functions: list[callable] | None = None,
    model: str | None = None,
    task_mode: str | None = None,     # 'research' | 'dev'
    mcp_servers: list[str] | None = None,
    name: str | None = None,
    log_path: str | None = None,
)
```

Methods:

| Method | Purpose |
|--------|---------|
| `agent(prompt=..., messages=..., system_prompt=..., skills=..., tools=..., functions=..., model=..., task_mode=..., strict_format=..., name=...)` | Run a multi-turn agentic call. Returns a `Result`. Per-call `system_prompt`/`skills`/`tools`/`functions` **append** to the harness; `model`/`task_mode` override. |
| `gate(options, prompt, *, system_prompt=..., tools=..., skills=...)` | Single-turn strict-format gate that returns exactly one of `options`. |
| `enrich(data)` | Append a user-visible message (text or full message list) to `self.messages`. |
| `compact(threshold_pct=...)` | Summarise older turns in-place. Returns True if compaction happened. |
| `save(path)` | Persist messages + settings to a v2 `.flow` file. |
| `Harness.load(path, *, functions=..., mcp_servers=..., name=..., log_path=...)` | Construct a new harness from a flow file. Local `functions=` must be re-supplied. |
| `clone(name=...)` | Independent copy; shares bootstrap state but owns its own `messages`. |
| `close()` / context-manager exit | Pop the log nesting and emit `HARNESS DONE`. |

Read-only properties: `messages` (mutable list), `tools`, `system_prompt`, `model`, `functions`.

### `Result`

Lazy, single-consumption handle:

- Iterate to stream `Event` objects.
- `r.wait()` — drain without iterating.
- `r.stop()` — abort the run.
- Final-state fields (read after draining): `r.text`, `r.reasoning`, `r.messages`, `r.tool_calls`, `r.finish_reason`, `r.usage`, `r.error`.

### `Event`

```python
Event.type: "content" | "reasoning" | "tool_call" | "tool_result"
```

Relevant fields per type:

| Type | Fields |
|------|--------|
| `content` | `.text` |
| `reasoning` | `.text` |
| `tool_call` | `.tool_name`, `.tool_args`, `.tool_call_id` |
| `tool_result` | `.tool_name`, `.tool_result`, `.tool_call_id`, `.is_error` |

More examples live in `examples/harnesses/` — streaming, bug-fix/review/refactor pipelines, web search, Smali/Android challenges, multi-task orchestration.

---

## Skills

A skill is a markdown file at `src/cai/skills/<name>.md` (or `~/.config/cai/skills/<name>.md` for user skills). Format:

```markdown
tools: search, read_file, list_files
---
## Skill: File System (Read-Only)

<prompt body merged into the system prompt>
```

Activating a skill:
- Unions its tools into the active toolset.
- Appends its prompt body to the system prompt.

Built-in skills:

| Name | Focus |
|------|-------|
| `adb` | Android Debug Bridge workflow (`adb_devices`, `adb_shell`, `adb_logcat`, ...) |
| `files` | Read-only filesystem exploration (`list_files`, `read_file`, `search`) |
| `frida` | Frida dynamic instrumentation |
| `smali` | Smali / dex static analysis |
| `web` | Web research with `fetch_url` + `search` |

User skills with the same name as a built-in take precedence.

---

## Agentic Behaviour

Both interactive mode and programmatic `Harness.agent()` run a multi-turn loop until the LLM answers without further tool calls (or hits `--max-turns`).

- A tier-appropriate (small/mid/large) base system prompt is injected unless overridden.
- **Stuck detection** (opt-in via `config.json`): a warning is injected if the same tool is called with the same args repeatedly.
- **Context compaction**: when prompt tokens exceed `context_budget_pct` of the model window, older turns are summarised in-place.
- **Tool-result trimming**: individual tool results are capped by `tool_result_max_chars` and shrunk dynamically as remaining context tightens.
- `--force-tools` sets `tool_choice=required` so at least one tool call is made.

---

## Log Viewer

Every `cai` invocation (CLI or SDK) writes structured JSONL records to `/tmp/cai/cai.log` (override with `--log-path` or `Harness(log_path=...)`).

```bash
cai --logger
```

Records nest hierarchically: `HARNESS` → `BLOCK` → tool call / result / content chunks → sub-calls. Each level is colour-coded.

**Navigation**

| Key | Action |
|-----|--------|
| `j` / `↓`, `k` / `↑` | Move cursor |
| `g` / `G` | Top / bottom |
| `Ctrl-d` / `Ctrl-u` | Half-page down / up |
| `F` | Toggle follow mode (auto-scroll) |

**Folding**

| Key | Action |
|-----|--------|
| `Tab` | Toggle fold on current entry |
| `zA` | Toggle fold recursively |
| `0` | Show only root level |
| `1`–`8` | Show 2–9 nesting levels |
| `9` | Unfold everything |

**Viewport**

| Key | Action |
|-----|--------|
| `zz` / `zt` / `zb` | Centre / top / bottom-align current entry |

**Search**

| Key | Action |
|-----|--------|
| `/pattern` / `?pattern` | Forward / backward regex search (case-insensitive) |
| `n` / `N` | Next / previous match |

Search crosses folded entries; ancestors unfold automatically to reveal a match.

**Other**

| Key | Action |
|-----|--------|
| `y` | Yank current entry to clipboard (`wl-copy`, `xclip`, `xsel`, `pbcopy`) |
| `q` / Ctrl-C | Quit |

---

## Model Profiles

Built-in capability map (prefix-matched against the model ID):

| Model prefix | Tier | Context | Tool calling |
|---|---|---|---|
| `arcee-ai/trinity-mini` | small | 8k | yes |
| `openai/gpt-4o-mini` | mid | 128k | yes |
| `openai/gpt-4o` | large | 128k | yes |
| `openai/o1` | large | 128k | yes |
| `openai/o3` | large | 200k | yes |
| `anthropic/claude-3-haiku` | small | 200k | yes |
| `anthropic/claude-3-5-haiku` | mid | 200k | yes |
| `anthropic/claude-3-5-sonnet` | large | 200k | yes |
| `anthropic/claude-3-7-sonnet` | large | 200k | yes |
| `anthropic/claude-opus-4` | large | 200k | yes |
| `anthropic/claude-sonnet-4` | large | 200k | yes |
| `google/gemini-2.0-flash` | mid | 1M | yes |
| `google/gemini-2.5-pro` | large | 1M | yes |
| `meta-llama/llama-3.3-70b` | mid | 128k | yes |
| `meta-llama/llama-3.1-8b` | small | 128k | yes |
| `mistralai/mistral-small` | small | 32k | yes |
| `mistralai/mistral-large` | large | 128k | yes |
| `deepseek/deepseek-r1` | large | 64k | no |
| `deepseek/deepseek-chat` | mid | 64k | yes |
| _(fallback)_ | mid | 128k | yes |

Profiles drive prompt verbosity, default max turns, and context thresholds. Override any entry under `"model_profiles"` in `config.json`.

---

## MCP Tools

All tools — built-in and external — share a single registry. The built-in server is always registered. Add more with `--mcp`:

```bash
cai --mcp "python server_a.py" "npx @modelcontextprotocol/server-filesystem /" \
    -p "Combine results from both tools"
```

Tool names are prefixed by a short server label to prevent collisions:

| Server command | Label | Example tool name |
|---|---|---|
| *(built-in)* | `cai` | `cai__generic_linux_command` |
| `python server.py` | `server` | `server__my_tool` |
| `npx @scope/my-server` | `my-server` | `my-server__fetch` |
| `uvx mcp-server-git` | `mcp-server-git` | `mcp-server-git__log` |

Enable specific tools with `-t` using the prefixed name. In the TUI, `:tools` toggles them interactively.

---

## Flow Files

A `.flow` file is a JSON snapshot of a session: composed system message (at index 0 of `messages`) plus a `settings` block with `system_prompt_base`, `task_mode`, `skills`, `selected_tools`, and `model`.

- CLI: `:save path` / `:load path` in interactive mode, or `cai --context path` to resume non-interactively.
- SDK: `harness.save(path)` / `Harness.load(path, functions=..., mcp_servers=...)`.

Local Python `functions=` registered via the SDK can't be serialised — pass them again on `Harness.load`.

---

## Shell Completion

Tab completion auto-installs on first run for bash, zsh, and fish. To set it up manually:

```bash
# bash
eval "$(register-python-argcomplete cai)"

# zsh
autoload -U bashcompinit && bashcompinit
eval "$(register-python-argcomplete cai)"

# fish
register-python-argcomplete --shell fish cai | source
```

---

## Project Structure

```
cai/
├── src/cai/
│   ├── __init__.py        # Re-exports Harness, Result, Event
│   ├── cli.py             # CLI entry point, agentic loop, interactive wiring
│   ├── sdk.py             # Programmatic SDK (Harness, Result, Event)
│   ├── core.py            # Bootstrap, config, skills, system-prompt assembly
│   ├── llm.py             # call_llm, MODEL_PROFILES, compaction, tool-result trimming
│   ├── api.py             # OpenAI-compatible clients (OpenAiApi, OpenRouterApi)
│   ├── tools.py           # MCP registry, internal + external servers, local function registration
│   ├── logger.py          # Hierarchical JSONL logger + log-viewer TUI
│   ├── screen/            # Vim-modal TUI (state, modes, input, layout, overlays)
│   ├── prompts/           # local.md, sota.md — base system prompts
│   ├── skills/            # adb.md, files.md, frida.md, smali.md, web.md
│   ├── adb_tools.py       # Built-in tool implementations
│   ├── files_tools.py
│   ├── frida_tools.py
│   ├── git_tools.py
│   ├── smali_tools.py
│   └── web_tools.py
├── examples/harnesses/    # Python SDK examples (sample, streaming, review, ...)
└── pyproject.toml         # Build config; entry point: cai = "cai.cli:main"
```

---

## License

See [LICENSE](./LICENSE).
