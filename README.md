# cai — Command-line AI

`cai` is a lightweight CLI tool that brings LLM intelligence into your terminal and editor workflow. It supports plain prompts, cursor-aware code generation, line-by-line batch processing, MCP tool integration, harness orchestration, and a full interactive TUI — all powered by any OpenAI-compatible API (defaults to [OpenRouter](https://openrouter.ai)).

---

## Features

- **`prompt`** — Send a prompt with optional file, stdin, cursor location, and tool context.
- **`--interactive`** — Full-screen TUI chat session with persistent history, streaming output, and vim-style commands.
- **`--harness`** — Run a `.harness.cai` orchestration file: multi-block agentic pipelines with control flow (labels, gotos, conditionals, loops, for-each sub-harnesses).
- **`--logger`** — Launch the interactive hierarchical log viewer TUI (tails `/tmp/cai/cai.log`).
- **`--force-tools`** — Require the LLM to use a tool at least once (sets `tool_choice=required`).
- **MCP tool support** — Add any MCP server via `--mcp <command>`; tools from all servers are unified under a prefixed namespace (e.g. `cai__generic_linux_command`, `myserver__scan_target`). The built-in server is always registered automatically.
- **Model profiles** — Built-in capability map (context window, tier, tool-calling) for 20+ models; user-overridable.
- **Context budget management** — Automatically compacts conversation history when the context window fills up.
- **Stuck detection** — Injects a warning if the LLM keeps calling the same tool with the same arguments.
- **Line-by-line batch mode** — Process stdin or a file one line at a time, calling the LLM per line (with optional parallelism via `--cores`).
- **Tab completion** — Auto-installs shell completion for bash, zsh, and fish on first run.
- Works with any OpenAI-compatible endpoint (OpenRouter, local Ollama, etc.).

---

## Installation

```bash
git clone https://github.com/youruser/cai
cd cai
pip install -e .
```

### Configuration

Config files are auto-created on first run under `~/.config/cai/`. You only need to add your API key:

```bash
echo "sk-or-..." > ~/.config/cai/api_key
```

`~/.config/cai/config.json` is created automatically with defaults. You can edit it to set a permanent base URL, default model, or tune agentic behavior:

```json
{
  "base_url": "https://openrouter.ai/api/v1",
  "model": "arcee-ai/trinity-mini:free",
  "context_budget_pct": 0.75,
  "tool_result_max_chars": 8000,
  "model_profiles": {
    "my-custom/model": {"tier": "large", "context": 128000, "tool_calling": true}
  }
}
```

---

## Usage

### Basic Prompt

```bash
cai -p "Explain what a closure is in Python"
# or using trailing words (no -p needed):
cai -- Explain what a closure is in Python
```

### Pipe stdin into context

```bash
cat error.log | cai -p "What is causing this error?"
git diff | cai -p "Write a commit message for this diff"
```

### Include a file in context

```bash
cai --file ./src/cai/cli.py -p "What does this file do?"
```

### Cursor-aware code generation

```bash
cai --cursor "./mymodule.py:42:4" -p "Implement this method"
```

The file is included with the cursor position marked — great for Vim/editor integrations.

### Interactive TUI session

```bash
cai --interactive
# with an initial prompt and a file pre-loaded:
cai --interactive --file ./src/cai/cli.py -p "Walk me through this file"
```

### Use an external MCP server

```bash
cai --mcp "python /path/to/my_mcp_server.py" -p "Use the custom tool to do X"
cai --mcp "npx @modelcontextprotocol/server-filesystem /" -p "List the project files"
cai --mcp "uvx mcp-server-git" -p "Summarise recent commits"
```

### Use internal named tools

```bash
cai -t cai__generic_linux_command -p "Check disk usage"
cai -t cai__fetch_codebase_metadata -p "Map the entire codebase"
```

### Process multiple lines in parallel (batch mode)

```bash
cat list_of_questions.txt | cai --line-by-line --cores 8 -p "Answer briefly"
```

Each line is sent as a separate LLM call, parallelized with `--cores`.

### Force strict output format

`--strict-format` retries the LLM until its response matches the required format, injecting a system prompt to guide it and feeding back failure messages between attempts (up to 4 by default).

**JSON** — response must be a valid JSON object:

```bash
cai --strict-format json -p "List 3 refactoring suggestions" --file ./cli.py
```

**Regex** — response must match a regular expression:

```bash
cai --strict-format "regex:^\d{4}-\d{2}-\d{2}$" -p "What is today's date?"
cai --strict-format "regex:^(yes|no)$" -p "Is this code thread-safe?" --file ./worker.py
```

**Regex-each-line** — every line of the response must match a pattern:

```bash
cai --strict-format "regex-each-line:^\w+" -p "List the function names" --file ./api.py
```

### Run a harness orchestration file

```bash
cai --harness harnesses/bug-fix.harness.cai -- "fix the crash in auth.py"
cai --harness harnesses/code-review.harness.cai -- "review src/payments/"
```

### Launch the log viewer

```bash
cai --logger
```

Opens the interactive hierarchical log viewer, tailing `/tmp/cai/cai.log` in real time.

### Custom model + system prompt

```bash
cai --model "openai/gpt-4o" --system-prompt "You are a senior Go engineer." -p "Review this code" --file ./main.go
```

### One-liner output (pipe-friendly)

```bash
cai --oneline -p "Summarize this function in one sentence" --file ./cli.py
```

---

## Key Flags

| Flag | Purpose |
|------|---------|
| `-p` / `--` | The prompt |
| `--interactive` / `-i` | Full-screen TUI chat session |
| `--harness` | Path to a `.harness.cai` orchestration file |
| `--logger` | Launch the interactive hierarchical log viewer |
| `--force-tools` | Require the LLM to call a tool at least once |
| `--max-turns N` | Max tool-call turns (default: tier-based 5/10/20) |
| `--file` | Include a file in context |
| `--cursor file:line:col` | Cursor-aware generation |
| `-t` | Internal tool names to enable (prefixed, e.g. `cai__generic_linux_command`) |
| `--mcp CMD` | Shell command to launch an external MCP server |
| `--line-by-line` | Process each line independently |
| `--cores N` | Parallel threads for batch processing |
| `--strict-format json\|regex:<pattern>\|regex-each-line:<pattern>` | Force output format |
| `--oneline` | Collapse response to single line |
| `--model` | Override model |
| `--system-prompt` | Set system prompt |
| `--non-streaming` | Use blocking API instead of streaming |
| `--progress` | Show progress bar (for `--line-by-line`) |

---

## Interactive Mode

`--interactive` opens a persistent full-screen TUI powered by raw ANSI escape codes. No external dependencies required.

```bash
cai --interactive
cai --interactive --model "openai/gpt-4o" -p "Let's review this PR" --file ./diff.patch
```

**Layout:**
- Scrollable conversation view (LLM responses stream in cyan)
- Status bar showing: `model | ctx XX% (used/total)`
- Input line at the very bottom

**Input editing:**

| Key | Action |
|-----|--------|
| Arrow up/down | History navigation (or move between lines in multi-line input) |
| Arrow left/right | Move cursor |
| Home / Ctrl-A | Beginning of current line / absolute beginning |
| End / Ctrl-E | End of current line / absolute end |
| Ctrl-U | Scroll half page up |
| Ctrl-D | Scroll half page down |
| Page Up / Page Down | Scroll full page |
| Ctrl-K | Kill to end of line |
| Ctrl-W | Delete word before cursor |
| Backspace | Delete character |
| Delete | Forward delete character |
| `\` + Enter | Insert newline (multi-line input) |
| Alt-Enter | Insert newline |
| Ctrl-V | Open current prompt in nvim |
| Enter | Submit |
| Ctrl-C | Clear input (or exit if buffer empty) |
| Ctrl-D | Exit |

**Scrolling during streaming:** While the LLM is responding, Page Up/Down and Ctrl-U/Ctrl-D scroll the conversation without interrupting the stream. Ctrl-C interrupts the current LLM call.

**Vim-style command mode:** Type `:` on an empty input line to enter command mode (shown in the status bar). Available commands:

| Command | Action |
|---------|--------|
| `:compact` | Summarize old conversation turns into a memory message |
| `:clear` | Clear conversation history (keep system prompt) |
| `:tools` | Open the interactive tools toggle overlay |

Tab completion is available in command mode.

**Tools overlay** (`:tools`): A floating panel listing all available tools from all registered MCP servers. Each row shows the tool name and its server origin (e.g. `[cai]`, `[python]`). Navigate with `j`/`k` or arrow keys, toggle with Space. Search forward with `/pattern` or backward with `?pattern`; use `n`/`N` to cycle matches. Press Enter or ESC to close.

**Interactive mode always uses agentic tool-calling.** Tool calls and results are shown inline in dim gray.

---

## Agentic Behavior

In interactive mode (and when tools are enabled in prompt mode), `cai` runs a multi-turn loop where the LLM calls tools, reads results, and continues until it can answer without further tool calls.

**How it works:**
1. Each turn: tool results are appended to messages and the LLM continues.
2. Interactive mode injects a tier-appropriate system prompt (small/mid/large) unless `--system-prompt` is set.
3. Default max turns: 5 (small models), 10 (mid), 20 (large). Override with `--max-turns`.
4. **Stuck detection:** if the LLM calls the same tool with identical arguments 2+ times, a warning is injected into the conversation.
5. **Context compaction:** when prompt token usage exceeds the configured threshold (default 75%), older conversation turns are summarized into a compact memory message to free up context space.

Use `--force-tools` to require a tool call at least once (`tool_choice=required`).

---

## Harness Orchestration

`.harness.cai` files are multi-block agentic programs — each block calls the LLM with its own prompt, tools, model, and format, and the blocks are connected by control flow instructions.

```bash
cai --harness harnesses/bug-fix.harness.cai -- "fix the NoneType crash in auth.py"
```

### Harness format

```
# Comments start with #

label:               # jump target (bare word followed by colon)

---                  # opens a block
    --name "x"                          # required: block identifier for branching
    --enrich full                       # required: none | result-only | full
    --prepend-user-prompt               # prepend the user's task to this block's prompt
    --tools cai__read_file, cai__list_files  # prefixed tool names (comma or space separated)
    --model gpt-4o                      # override model for this block
    --max-turns 100                     # override max tool-call turns
    --strict-format "regex:^(ok|retry)$"  # enforce output format
    --system-prompt "..."               # block-specific system prompt
    --force-tools                       # require at least one tool call
    '''
    Prompt text goes here.
    Multiple lines are fine.
    '''
---

if x == ok: goto label       # conditional jump (exact string match)
goto label                   # unconditional jump
exit                         # terminate harness
compact-if-more-than 30      # compact global context if usage exceeds 30% of window
if-more-than 5 done          # jump to 'done' if this point has been passed more than 5 times
for-each item in block: harness "path/to/sub.harness.cai"
```

### Context enrichment

`--enrich <mode>` is required on every block and controls what it contributes to `global_messages`:

- `--enrich full`: adds the user prompt, all tool calls/results, and the final assistant response. Use for context-gathering blocks.
- `--enrich result-only`: adds only the final assistant response. Use for classify/gate blocks with `--strict-format` where only the verdict matters downstream.
- `--enrich none`: nothing is added. Use for transient blocks whose deliberation is not needed downstream.

### `compact-if-more-than <percentage>`

Checks whether the accumulated `global_messages` exceed `<percentage>%` of the model's context window. If so, summarises the middle turns into a single `[memory]` entry. No-op if under the threshold. Useful in retry loops.

### `for-each <item> in <block>: harness "<path>"`

Runs a sub-harness once for each line of a named block's output. Each run starts with a fresh context and receives one line as its `user_prompt`. After all iterations complete, a structured summary is injected into the parent's global context so subsequent parent blocks can reference all results.

### Built-in harnesses

A collection of ready-to-use harnesses is included in `harnesses/`:

| Harness | Purpose |
|---------|---------|
| `context-and-execute` | Gather context → verify → execute. The canonical pattern for any code task. |
| `bug-fix` | Gather → verify → fix → self-review → summary |
| `code-review` | Gather → verify → deep analysis → severity → report |
| `test-writer` | Gather source → gather patterns → plan → validate → write |
| `refactor` | Gather → verify → plan → validate → execute → sanity check → summary |
| `feature` | Gather → verify → design → validate → implement → write tests |
| `security-audit` | Gather → verify → threat model → deep audit → severity → report |
| `migrate` | Gather → audit callsites → validate → plan → execute → verify → summary |
| `explain` | Gather → verify → trace execution/data flow → layered explanation |
| `document` | Gather code → gather doc conventions → outline → validate → write |
| `web-search` | Decompose into sub-questions → search each → synthesize |
| `multi-task` | Decompose task → run sub-harness for each part → aggregate |

```bash
cai --harness harnesses/refactor.harness.cai -- "extract retry logic in api.py into a reusable decorator"
cai --harness harnesses/web-search.harness.cai -- "trade-offs between PostgreSQL and MongoDB"
```

---

## Log Viewer

`--logger` opens a full-screen interactive TUI that tails `/tmp/cai/cai.log` in real time. The log is written automatically by every `cai` invocation in structured JSONL format.

```bash
cai --logger
```

Entries are displayed with **hierarchical nesting** — tool calls and their results nest under the turn that invoked them, harness blocks nest under the harness, etc. Each nesting level is color-coded.

**Navigation:**

| Key | Action |
|-----|--------|
| `j` / `↓` | Move cursor down one entry |
| `k` / `↑` | Move cursor up one entry |
| `g` | Jump to top |
| `G` | Jump to bottom |
| `Ctrl-d` | Half-page down |
| `Ctrl-u` | Half-page up |
| `F` | Toggle follow mode (auto-scroll to new entries) |

**Folding:**

| Key | Action |
|-----|--------|
| `Tab` | Toggle fold on current entry (non-recursive) |
| `zA` | Toggle fold on current entry and all descendants |
| `0` | Show only the shallowest (root) level |
| `1`–`8` | Show 2–9 nesting levels from the root |
| `9` | Unfold everything |

**Viewport alignment:**

| Key | Action |
|-----|--------|
| `zz` | Centre current entry in viewport |
| `zt` | Align current entry to top of viewport |
| `zb` | Align current entry to bottom of viewport |

**Search:**

| Key | Action |
|-----|--------|
| `/pattern` | Search forward (regex, case-insensitive) |
| `?pattern` | Search backward |
| `n` | Jump to next match |
| `N` | Jump to previous match |

Search jumps work across the entire log (including folded entries); ancestors are automatically unfolded to reveal the match.

**Other:**

| Key | Action |
|-----|--------|
| `y` | Yank current entry text to clipboard (`wl-copy`, `xclip`, `xsel`, or `pbcopy`) |
| `q` / Ctrl-C | Quit |

---

## Model Profiles

`cai` ships with a built-in capability map for 20+ models:

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
| _(unknown)_ | mid | 16k | yes |

Profiles drive system prompt verbosity, default max turns, and context compaction thresholds. You can override any profile in `config.json` under `"model_profiles"`.

---

## Line-by-line Batch Mode

Process many inputs in a batch, calling the LLM once per line:

```bash
# Process each line of a file
cai --file ./lines.txt --line-by-line -p "Classify this line."

# Read from stdin line by line, 4 threads
cat items.txt | cai --line-by-line --cores 4 -p "Translate to French."
```

Ctrl-C during batch mode cancels all pending tasks gracefully.

---

## MCP Tools

All tools — built-in and external — are managed through a unified MCP server registry. The built-in server is always registered automatically. Add external servers with `--mcp`:

```bash
# Single external server
cai --mcp "python /path/to/mcp_server.py" -p "Use this tool"

# Multiple servers at once
cai --mcp "python server_a.py" "npx @modelcontextprotocol/server-filesystem /" \
    -p "Combine results from both tools"
```

Tool names are automatically prefixed with a short server label to prevent collisions:

| Server command | Label | Example tool name |
|---|---|---|
| *(built-in)* | `cai` | `cai__generic_linux_command` |
| `python server.py` | `server` | `server__my_tool` |
| `npx @scope/my-server` | `my-server` | `my-server__fetch` |
| `uvx mcp-server-git` | `mcp-server-git` | `mcp-server-git__log` |

Enable specific tools via `-t` using the prefixed name:

```bash
cai -t cai__generic_linux_command -p "Check disk usage"
```

The `:tools` overlay in interactive mode shows all tools from all servers, with origin labels.

---

## Shell Completion

Tab completion is installed automatically on first run for bash, zsh, and fish. To set it up manually:

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
│   ├── cli.py      # CLI entry point, argument parsing, agentic loop, model profiles
│   ├── api.py      # OpenAI-compatible API clients (OpenAiApi, OpenRouterApi)
│   ├── screen.py   # Terminal TUI for --interactive (raw ANSI, no dependencies)
│   ├── harness.py  # .harness.cai parser and executor (blocks, labels, gotos, for-each)
│   ├── logger.py   # Hierarchical JSONL logger + interactive log viewer TUI
│   └── tools.py    # MCP tool server + tool dispatch logic (tree-sitter codebase parsing)
├── harnesses/      # Ready-to-use .harness.cai orchestration files
└── pyproject.toml  # Build config; entry point: cai = "cai.cli:main"
```

---

## License

See [LICENSE](./LICENSE).
