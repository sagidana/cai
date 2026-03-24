# cai — Command-line AI

`cai` is a lightweight CLI tool that brings LLM intelligence into your terminal and editor workflow. It supports plain prompts, cursor-aware code generation, line-by-line batch processing, and MCP tool integration — all powered by any OpenAI-compatible API (defaults to [OpenRouter](https://openrouter.ai)).

---

## Features

- **`prompt`** — Send a prompt with optional file, stdin, cursor location, and tool context.
- **MCP tool support** — Pass external MCP server scripts via `-t`; the LLM can call their tools automatically.
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

`~/.config/cai/config.json` is created automatically with defaults. You can edit it to set a permanent base URL or default model:

```json
{
  "base_url": "https://openrouter.ai/api/v1",
  "model": "arcee-ai/trinity-mini:free"
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
| `-a`, `--action` | `prompt` | Action to perform (currently only `prompt`) |
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
│   ├── cli.py      # CLI entry point and action handlers
│   ├── api.py      # OpenAI-compatible API clients (OpenAiApi, OpenRouterApi, AnthropicApi)
│   └── tools.py    # MCP tool server + tool dispatch logic (tree-sitter codebase parsing)
└── pyproject.toml  # Build config; entry point: cai = "cai.cli:main"
```

---

## License

See [LICENSE](./LICENSE).
