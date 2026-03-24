# cai — Command-line AI

`cai` is a lightweight CLI tool that brings LLM intelligence into your terminal and editor workflow. It supports plain prompts, cursor-aware code generation, line-by-line batch processing, MCP tool integration, multi-turn agentic loops, and a full interactive TUI — all powered by any OpenAI-compatible API (defaults to [OpenRouter](https://openrouter.ai)).

---

## Features

- **`prompt`** — Send a prompt with optional file, stdin, cursor location, and tool context.
- **`--interactive`** — Full-screen TUI chat session with persistent history, status bar, and streaming output.
- **`--agentic`** — Multi-turn tool-calling loop: the LLM keeps calling tools until it reaches a final answer.
- **MCP tool support** — Pass external MCP server scripts via `-t`; the LLM can call their tools automatically.
- **Model profiles** — Built-in capability map (context window, tier, tool-calling) for 20+ models; user-overridable.
- **Context budget management** — Automatically compacts conversation history when the context window fills up.
- **Stuck detection** — Injects a warning if the LLM keeps calling the same tool with the same arguments.
- **Line-by-line batch mode** — Process stdin or a file one line at a time, calling the LLM per line (with optional parallelism via `--cores`).
- **vimgrep integration** — Feed `vimgrep`-format output directly; `cai` loads each matched file's context automatically.
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

● Here are the most exciting and useful ways to run cai:

  ---
  Basic Prompt

  cai -p "Explain what a closure is in Python"
  # or using -- shorthand (no -p needed):
  cai -- Explain what a closure is in Python

  ---
  Pipe stdin into context

  cat error.log | cai -p "What is causing this error?"
  git diff | cai -p "Write a commit message for this diff"

  ---
  Include a file in context

  cai --file ./src/cai/cli.py -p "What does this file do?"

  ---
  Cursor-aware code generation (location)

  cai --location "./mymodule.py:42:4" -p "Implement this method"
  The file is included with the cursor position marked — great for Vim/editor integrations.

  ---
  Interactive TUI session

  cai --interactive
  # with an initial prompt and a file pre-loaded:
  cai --interactive --file ./src/cai/cli.py -p "Walk me through this file"

  Opens a full-screen chat interface. The status bar shows the model name, context usage,
  and current status. Supports multi-line input (\+Enter or Alt-Enter), history (arrow keys),
  and standard readline-style editing (Ctrl-A/E/U/K). Ctrl-C or Ctrl-D to exit.

  ---
  Agentic mode (multi-turn tool loop)

  cai --agentic -t fetch_codebase_metadata -p "What are all the public methods in api.py?"
  The LLM calls tools repeatedly until it has enough information to answer.

  ---
  Enable codebase introspection tool

  cai --codebase --file ./src/cai/api.py -p "Add a POST method"
  Unlocks the fetch_codebase_metadata tool (tree-sitter powered class/method map).

  ---
  Use a custom MCP tool server

  cai -t /path/to/my_mcp_server.py -p "Use the custom tool to do X"

  ---
  Use an internal named tool

  cai -t fetch_codebase_metadata -p "Map the entire codebase"

  ---
  Process multiple lines in parallel (batch mode)

  cat list_of_questions.txt | cai --line-by-line --cores 8 -p "Answer briefly"
  Each line is sent as a separate LLM call, parallelized with --cores.

  ---
  Vimgrep integration — analyze grep results with file context

  grep -rn "TODO" ./src | cai --vimgrep -p "Categorize each TODO by urgency"
  # or from Vim: :cai uses vimgrep format (file:line:col:text)
  Auto-loads the matched file into context for each result.

  ---
  Force JSON output

  cai --strict-format json -p "List 3 refactoring suggestions for this file" --file ./cli.py
  Retries the LLM until it returns valid JSON.

  ---
  Custom model + system prompt

  cai --model "openai/gpt-4o" --system-prompt "You are a senior Go engineer." -p "Review this code" --file ./main.go

  ---
  One-liner output (pipe-friendly)

  cai --oneline -p "Summarize this function in one sentence" --file ./cli.py

  ---
  Key flags at a glance:

  ┌──────────────────────────┬─────────────────────────────────────────────┐
  │           Flag           │                   Purpose                   │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ -p / --                  │ The prompt                                  │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --interactive            │ Full-screen TUI chat session                │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --agentic                │ Multi-turn tool-calling loop                │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --max-turns N            │ Max tool-call turns in agentic mode         │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --file                   │ Include a file in context                   │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --location file:line:col │ Cursor-aware generation                     │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --codebase               │ Enable tree-sitter codebase tool            │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ -t                       │ MCP tool server path or internal tool name  │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --line-by-line           │ Process each line independently             │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --vimgrep                │ Parse grep -n style input with file context │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --cores N                │ Parallel threads for batch processing       │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --strict-format json     │ Force JSON output                           │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --oneline                │ Collapse response to single line            │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --model                  │ Override model                              │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --system-prompt          │ Set system prompt                           │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ --non-streaming          │ Use blocking API instead of streaming       │
  └──────────────────────────┴─────────────────────────────────────────────┘

### Global Options

| Flag | Default | Description |
|------|---------|-------------|
| `-a`, `--action` | `prompt` | Action to perform (`prompt`) |
| `-p`, `--prompt` | — | The prompt text to send to the LLM |
| `--system-prompt` | — | Optional system prompt |
| `--file` | — | Path to a file to include in the LLM context (with line numbers) |
| `--location` | — | Cursor location in format `<file>:<line>:<col>` — includes that file with cursor position |
| `--model` | from config, else `arcee-ai/trinity-mini:free` | Model ID to use |
| `--cwd` | `.` | Working directory for tool execution |
| `-t`, `--tools` | — | One or more paths to external MCP server Python files, or internal tool names |
| `--non-streaming` | off | Use blocking (non-streaming) API call |
| `--strict-format` | — | Enforce LLM output format: `json` |
| `--include-reasoning` | off | Include reasoning in output |
| `--oneline` | off | Collapse response to a single line (vimgrep-friendly) |
| `--progress` | off | Show a progress bar (for `--line-by-line`) |
| `--line-by-line` | off | Process stdin or `--file` one line at a time, calling LLM per line |
| `--vimgrep` | off | Treat each input line as `file:line:col:text`; loads file context automatically (implies `--line-by-line`) |
| `--cores` | `1` | Number of parallel threads for `--line-by-line` processing |
| `--agentic` | off | Enable multi-turn agentic loop: LLM keeps calling tools until done |
| `--interactive` | off | Open a full-screen TUI session; implies `--agentic`. Ctrl-C or Ctrl-D to exit |
| `--max-turns` | tier-based (5/10/20) | Max tool-call turns in agentic mode |

---

## Actions

### `prompt` — LLM prompt with optional context

Send a prompt, optionally enriched with file content, stdin, or a cursor position.

```bash
# Basic prompt
cai -a prompt -p "What is a closure in Python?"

# With a file attached
cai -a prompt -p "Summarize what this file does." --file ./src/cai/cli.py

# Piped stdin
cat error.log | cai -a prompt -p "What is causing this error?"

# With a cursor location — includes the file and marks the position
cai -a prompt \
  --location "./mymodule.py:42:4" \
  --prompt "implement a method that returns the sorted list of keys"

# Use an external MCP tool server
cai -a prompt -p "Use this tool" -t /path/to/mcp_server.py

# Use a specific model
cai -a prompt -p "Explain recursion." --model "anthropic/claude-sonnet-4-6"
```

---

## Interactive Mode

`--interactive` opens a persistent full-screen TUI powered by raw ANSI escape codes. No external dependencies required.

```bash
cai --interactive
cai --interactive --model "openai/gpt-4o" -p "Let's review this PR" --file ./diff.patch
```

**Layout:**
- Scrollable output region at the top (LLM responses stream in here in cyan)
- Persistent status bar (reverse video) at the bottom showing: `model | ctx XX% (used/total) | status`
- Input line at the very bottom

**Input editing:**
| Key | Action |
|-----|--------|
| Arrow up/down | History navigation |
| Arrow left/right | Move cursor |
| Home / Ctrl-A | Beginning of line |
| End / Ctrl-E | End of line |
| Ctrl-U | Kill to beginning |
| Ctrl-K | Kill to end |
| Backspace / Ctrl-D | Delete character |
| `\` + Enter | Insert newline (multi-line input) |
| Alt-Enter | Insert newline |
| Enter | Submit |
| Ctrl-C / Ctrl-D | Exit |

**Interactive mode always uses agentic tool-calling.** Tool calls and results are shown inline in dim gray.

---

## Agentic Mode

`--agentic` enables a multi-turn loop where the LLM calls tools, reads results, and continues until it can answer without further tool calls.

```bash
cai --agentic -t fetch_codebase_metadata -p "Find all TODO comments in the codebase"
cai --agentic --max-turns 5 -t /path/to/server.py -p "Run the test suite and report failures"
```

**How it works:**
1. Turn 1 forces at least one tool call (`tool_choice=required`) so the model can't skip tools.
2. Each subsequent turn: tool results are appended to messages and the LLM continues.
3. Agentic mode applies a tier-appropriate system prompt (small/mid/large) unless `--system-prompt` is set.
4. Default max turns: 5 (small models), 10 (mid), 20 (large). Override with `--max-turns`.
5. **Stuck detection:** if the LLM calls the same tool with identical arguments 3+ times, a warning is injected into the conversation.
6. **Context compaction:** when prompt token usage exceeds the configured threshold (default 75%), older conversation turns are summarized into a compact memory message to free up context space.

---

## Model Profiles

`cai` ships with a built-in capability map for 20+ models:

| Model prefix | Tier | Context | Tool calling |
|---|---|---|---|
| `arcee-ai/trinity-mini` | small | 8k | yes |
| `openai/gpt-4o-mini` | mid | 128k | yes |
| `openai/gpt-4o` | large | 128k | yes |
| `anthropic/claude-3-5-sonnet` | large | 200k | yes |
| `anthropic/claude-3-7-sonnet` | large | 200k | yes |
| `google/gemini-2.5-pro` | large | 1M | yes |
| `meta-llama/llama-3.3-70b` | mid | 128k | yes |
| `deepseek/deepseek-r1` | large | 64k | no |
| _(unknown)_ | mid | 16k | yes |

Profiles drive agentic system prompt verbosity, default max turns, and context compaction thresholds. You can override any profile in `config.json` under `"model_profiles"`.

---

## Line-by-line & vimgrep Mode

Process many inputs in a batch, calling the LLM once per line:

```bash
# Process each line of a file
cai -a prompt --file ./lines.txt --line-by-line -p "Classify this line."

# Read from stdin line by line, 4 threads
cat items.txt | cai -a prompt --line-by-line --cores 4 -p "Translate to French."

# Feed vimgrep output — file context is loaded automatically per match
grep -rn "TODO" . --include="*.py" | \
  cai -a prompt --vimgrep --oneline -p "Suggest a fix for this TODO."
```

In `--vimgrep` mode, each input line must be in `file:line:col:text` format (standard `vimgrep` / `grep -n` output). The matched file's full content is included in the LLM context alongside the match location.

---

## MCP Tools

Pass one or more MCP server scripts with `-t`. The LLM can call their tools automatically during a session:

```bash
cai -a prompt -p "Use this tool" -t /path/to/mcp_server.py
```

Internal tool names (non-path values) are also accepted to enable built-in tools:

```bash
cai -a prompt -p "Inspect the project" -t fetch_codebase_metadata
```

Tab completion for `-t` completes both file paths and internal tool names.

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
│   ├── cli.py      # CLI entry point, action handlers, agentic loop, model profiles
│   ├── api.py      # OpenAI-compatible API clients (OpenAiApi, OpenRouterApi, AnthropicApi)
│   ├── screen.py   # Terminal TUI for --interactive (raw ANSI, no dependencies)
│   └── tools.py    # MCP tool server + tool dispatch logic (tree-sitter codebase parsing)
└── pyproject.toml  # Build config; entry point: cai = "cai.cli:main"
```

---

## License

See [LICENSE](./LICENSE).
