# cai — Command-line AI

`cai` is a lightweight CLI tool that brings LLM intelligence into your terminal and editor workflow. It supports plain prompts, cursor-aware code generation, and optional codebase introspection — all powered by any OpenAI-compatible API (defaults to [OpenRouter](https://openrouter.ai)).

---

## Features

- **`prompt`** — Send a prompt with optional file, stdin, cursor location, and codebase context.
- **MCP tool server** — Built-in `fetch_codebase_metadata` tool that parses source files via tree-sitter and returns class/method structure. Supports Python, Java, C, C++, and Smali; auto-detects language from file extension.
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
| `-a`, `--action` | Action to perform: `prompt` |
| `-p`, `--prompt` | The prompt text to send to the LLM |
| `--system-prompt` | Optional system prompt |
| `--file` | Path to a file to include in the LLM context (with line numbers) |
| `--location` | Cursor location in format `<file>:<line>:<col>` — includes that file with the cursor position highlighted |
| `--codebase` | Give the LLM access to the `fetch_codebase_metadata` tool so it can inspect project structure |
| `--model` | Model ID to use (default: `anthropic/claude-opus-4.6`) |
| `--cwd` | Working directory for tools (default: `.`) |

---

## Actions

### `prompt` — LLM prompt with optional context

Send a prompt, optionally enriched with file content, stdin, a cursor position, or full codebase introspection.

```bash
# Basic prompt
python cai.py -a prompt -p "What is a closure in Python?"

# With a file attached
python cai.py -a prompt -p "Summarize what this file does." --file ./cai.py

# Piped stdin
cat error.log | python cai.py -a prompt -p "What is causing this error?"

# With a cursor location — includes the file and marks the position
python cai.py -a prompt \
  --location "./mymodule.py:42:4" \
  --prompt "implement a method that returns the sorted list of keys"

# With codebase awareness — LLM can call fetch_codebase_metadata as needed
python cai.py -a prompt \
  --file ./api.py \
  --location "./api.py:30:4" \
  --codebase \
  --prompt "add a post method that sends a JSON body to a given endpoint"

# Use a specific model
python cai.py -a prompt -p "Explain recursion." --model "anthropic/claude-sonnet-4-6"
```

---

## MCP Tool: `fetch_codebase_metadata`

Available when `--codebase` is passed. Walks the working directory, auto-detects the language of each source file from its extension, parses it with tree-sitter, and returns a JSON structure:

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

**Supported languages and extensions:**

| Language | Extensions |
|----------|-----------|
| Python   | `.py` |
| Java     | `.java` |
| C        | `.c`, `.h` |
| C++      | `.cpp`, `.cc`, `.cxx`, `.c++`, `.hpp`, `.hh`, `.h++` |
| Smali    | `.smali` |

Mixed-language projects are handled transparently — each file is classified and parsed independently.

---

## Examples

### Explain a function

```bash
python cai.py -a prompt \
  --file ./tools.py \
  --prompt "Explain what the _parse_file function does."
```

### Generate code at a cursor position

```bash
python cai.py -a prompt \
  --location "./api.py:30:4" \
  --codebase \
  --prompt "add a post method that sends a JSON body to a given endpoint"
```

### Ask about your project structure

```bash
python cai.py -a prompt --codebase -p "List all classes and their methods in this project."
```

### Use stdin as context

```bash
cat myfile.py | python cai.py -a prompt -p "Are there any issues with this code?"
```

---

## Project Structure

```
cai/
├── cai.py          # CLI entry point and action handlers
├── api.py          # OpenAI-compatible and Anthropic API clients
├── tools.py        # MCP tool server + tool dispatch logic
├── config.json     # Local config template (copy to ~/.config/cai/)
└── requirements.txt
```

---

## License

See [LICENSE](./LICENSE).
