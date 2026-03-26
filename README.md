# cai — Command-line AI

`cai` is a lightweight CLI tool that brings LLM intelligence into your terminal and editor workflow. It supports plain prompts, cursor-aware code generation, line-by-line batch processing, MCP tool integration, and a full interactive TUI — all powered by any OpenAI-compatible API (defaults to [OpenRouter](https://openrouter.ai)).

---

## Features

- **`prompt`** — Send a prompt with optional file, stdin, cursor location, and tool context.
- **`--interactive`** — Full-screen TUI chat session with persistent history, streaming output, and vim-style commands.
- **`--force-tools`** — Require the LLM to use a tool every turn (sets `tool_choice=required`).
- **MCP tool support** — Pass external MCP server scripts via `-t`; the LLM can call their tools automatically.
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
cai --location "./mymodule.py:42:4" -p "Implement this method"
```

The file is included with the cursor position marked — great for Vim/editor integrations.

### Interactive TUI session

```bash
cai --interactive
# with an initial prompt and a file pre-loaded:
cai --interactive --file ./src/cai/cli.py -p "Walk me through this file"
```

### Use a custom MCP tool server

```bash
cai -t /path/to/my_mcp_server.py -p "Use the custom tool to do X"
```

### Use an internal named tool

```bash
cai -t fetch_codebase_metadata -p "Map the entire codebase"
```

### Process multiple lines in parallel (batch mode)

```bash
cat list_of_questions.txt | cai --line-by-line --cores 8 -p "Answer briefly"
```

Each line is sent as a separate LLM call, parallelized with `--cores`.

### Force JSON output

```bash
cai --strict-format json -p "List 3 refactoring suggestions" --file ./cli.py
```

Retries the LLM until it returns valid JSON.

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
| `--force-tools` | Require the LLM to call a tool every turn |
| `--max-turns N` | Max tool-call turns (default: tier-based 5/10/20) |
| `--file` | Include a file in context |
| `--location file:line:col` | Cursor-aware generation |
| `-t` | MCP tool server path or internal tool name |
| `--line-by-line` | Process each line independently |
| `--cores N` | Parallel threads for batch processing |
| `--strict-format json` | Force JSON output |
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

**Tools overlay** (`:tools`): A floating panel listing all available tools. Navigate with `j`/`k` or arrow keys, toggle with Space. Search forward with `/pattern` or backward with `?pattern`; use `n`/`N` to cycle matches. Press Enter or ESC to close.

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

Use `--force-tools` to require a tool call on every turn (`tool_choice=required`).

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

Pass one or more MCP server scripts with `-t`. The LLM can call their tools automatically during a session:

```bash
cai -p "Use this tool" -t /path/to/mcp_server.py
```

Internal tool names (non-path values) are also accepted to enable built-in tools:

```bash
cai -p "Inspect the project" -t fetch_codebase_metadata
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
│   ├── api.py      # OpenAI-compatible API clients (OpenAiApi, OpenRouterApi)
│   ├── screen.py   # Terminal TUI for --interactive (raw ANSI, no dependencies)
│   └── tools.py    # MCP tool server + tool dispatch logic (tree-sitter codebase parsing)
└── pyproject.toml  # Build config; entry point: cai = "cai.cli:main"
```

---

## License

See [LICENSE](./LICENSE).
