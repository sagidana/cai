# cai — Command-line AI

`cai` is a lightweight CLI tool that brings LLM intelligence into your terminal and editor workflow. It supports plain prompts, tool-assisted reasoning, and context-aware code generation — all powered by any OpenAI-compatible API (defaults to [OpenRouter](https://openrouter.ai)).

---

## Features

- **`prompt`** — Send a prompt with optional file or stdin context.
- **`knowit`** — Prompt the LLM with tool-calling support (MCP-based tools).
- **`impl`** — Generate code at a specific cursor location, with optional codebase introspection.
- **MCP tool server** — Built-in `fetch_codebase_infra` tool that parses Python files via tree-sitter and returns class/method structure.
- Works with any OpenAI-compatible endpoint (OpenRouter, local Ollama, etc.).

---

## Installation

```bash
git clone https://github.com/youruser/cai
cd cai
pip install -r requirements.txt
```

### Configuration

Create the config directory and add your API key and base URL:

```bash
mkdir -p ~/.config/cai

# API key (OpenRouter or any OpenAI-compatible provider)
echo "sk-or-..." > ~/.config/cai/api_key

# Base URL config
cat > ~/.config/cai/config.json <<EOF
{
    "base_url": "https://openrouter.ai/api/v1"
}
EOF
```

---

## Usage

```
python cai.py -a <action> [options]
```

### Global Options

| Flag | Description |
|------|-------------|
| `-a`, `--action` | Action to perform: `prompt`, `knowit`, `impl` |
| `-p`, `--prompt` | The prompt text to send to the LLM |
| `--system-prompt` | Optional system prompt |
| `--file` | Path to a file to include in the LLM context (with line numbers) |
| `--location` | Cursor location in format `<file>:<line>:<col>` (used by `impl`) |
| `--model` | Model ID to use (default: `arcee-ai/trinity-mini:free`) |
| `--output-language` | Programming language for `impl` output (default: `python`) |
| `--cwd` | Working directory for tools (default: `.`) |

---

## Actions

### `prompt` — Simple LLM prompt

Send a plain prompt, optionally with file content or stdin piped in.

```bash
# Basic prompt
python cai.py -a prompt -p "What is a closure in Python?"

# With a file attached
python cai.py -a prompt -p "Summarize what this file does." --file ./cai.py

# Piped stdin
cat error.log | python cai.py -a prompt -p "What is causing this error?"

# Use a specific model
python cai.py -a prompt -p "Explain recursion." --model "anthropic/claude-opus-4-6"
```

---

### `knowit` — Prompt with tool calling

Like `prompt`, but the LLM can invoke MCP tools to gather more information before answering.

```bash
# Ask a question where the LLM may need to inspect your codebase
python cai.py -a knowit -p "What classes are defined in this project?"

# With stdin
cat myfile.py | python cai.py -a knowit -p "Are there any issues with this code?"
```

The LLM has access to the `fetch_codebase_infra` tool, which parses all Python files in the current directory and returns a structured map of classes, methods, and top-level functions.

---

### `impl` — Code generation at cursor location

Generate code to insert at a specific location in a file. Provide the file content and cursor position so the LLM has full context. The LLM can also call `fetch_codebase_infra` to understand the rest of the project.

```bash
# Generate a method body at line 42, column 4
python cai.py -a impl \
  --file ./mymodule.py \
  --location "./mymodule.py:42:4" \
  --prompt "implement a method that returns the sorted list of keys"

# Generate in a different language
python cai.py -a impl \
  --file ./server.js \
  --location "./server.js:10:0" \
  --prompt "add an express route that returns all users" \
  --output-language javascript

# Pipe file content via stdin instead of --file
cat ./mymodule.py | python cai.py -a impl \
  --location "./mymodule.py:20:0" \
  --prompt "add input validation to this function"
```

The output is raw source code ready to be pasted or inserted at the cursor position.

---

## MCP Tool: `fetch_codebase_infra`

Available to `knowit` and `impl` actions. Walks the working directory, parses every `.py` file with tree-sitter, and returns a JSON structure:

```json
{
  "api.py": {
    "OpenAiApi": {
      "chat": "def chat(self, messages, model, system_prompt=None, tools=None)"
    },
    "OpenRouterApi": {
      "get_models": "def get_models(self)"
    }
  },
  "cai.py": {
    "main": "def main()"
  }
}
```

---

## Examples

### Explain a function

```bash
python cai.py -a prompt \
  --file ./tools.py \
  --prompt "Explain what the call_tool function does."
```

### Generate a new class method

```bash
python cai.py -a impl \
  --file ./api.py \
  --location "./api.py:30:4" \
  --prompt "add a post method that sends a JSON body to a given endpoint"
```

### Ask about your project structure

```bash
python cai.py -a knowit -p "List all classes and their methods in this project."
```

### Use a stronger model for complex tasks

```bash
python cai.py -a impl \
  --file ./cai.py \
  --location "./cai.py:65:0" \
  --prompt "implement the action_grep function to search files using ripgrep" \
  --model "anthropic/claude-sonnet-4-6"
```

---

## Project Structure

```
cai/
├── cai.py          # CLI entry point and action handlers
├── api.py          # OpenAI-compatible and Anthropic API clients
├── tools.py        # MCP tool server + tool dispatch logic
├── config.json     # Local config (copied to ~/.config/cai/)
└── requirements.txt
```

---

## License

See [LICENSE](./LICENSE).
