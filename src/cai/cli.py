import argparse
import argcomplete
import datetime
import json
import sys
import os
import re
import signal
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


global config
global available_tools
global api_key
global openai_api
global openrouter_api
global external_mcps
external_mcps = {}

logging.basicConfig(
    filename="/tmp/cai.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("cai")

# ─── Agent log (realtime file at /tmp/cai/agent_<datetime>.log) ──────────────
# One file per process invocation; shared across all call_llm calls in a session.
_AGENT_LOG_PATH = None
_AGENT_LOG_LOCK = threading.Lock()

def _get_agent_log_path():
    global _AGENT_LOG_PATH
    if _AGENT_LOG_PATH is None:
        os.makedirs("/tmp/cai", exist_ok=True)
        _AGENT_LOG_PATH = "/tmp/cai/agent.log"
    return _AGENT_LOG_PATH

def _alog(text):
    """Append a line to the realtime agent log file."""
    with _AGENT_LOG_LOCK:
        with open(_get_agent_log_path(), 'a', encoding='utf-8') as f:
            f.write(text + '\n')
            f.flush()

def _alog_messages_state(messages, label):
    """Dump the complete messages[] array to the agent log so the exact context is visible."""
    _CONTENT_LIMIT = 2000   # chars shown per message before truncation notice

    def _render_content(text):
        """Return lines for a content block, indented under the message header."""
        if not text:
            return ["│     (empty)"]
        lines = []
        if len(text) > _CONTENT_LIMIT:
            shown = text[:_CONTENT_LIMIT]
            omitted = len(text) - _CONTENT_LIMIT
            for line in shown.splitlines():
                lines.append("│     " + line)
            lines.append("│     ... [{:,} more chars truncated]".format(omitted))
        else:
            for line in text.splitlines():
                lines.append("│     " + line)
        return lines

    header = "┌─ MESSAGES[{}]  {} ".format(len(messages), label)
    header = header + "─" * max(0, 76 - len(header)) + "┐"
    _alog(header)

    for idx, msg in enumerate(messages):
        role = msg.get('role', 'unknown').upper()
        content = msg.get('content') or ''
        tool_calls = msg.get('tool_calls')
        tool_call_id = msg.get('tool_call_id')

        if tool_calls:
            parts = []
            for c in tool_calls:
                fn = c.get('function', {})
                name = fn.get('name', '?')
                raw_args = fn.get('arguments', '')
                call_id = c.get('id', '?')
                try:
                    parsed = json.loads(raw_args) if raw_args else {}
                    args_fmt = ', '.join(
                        "{}={}".format(k, json.dumps(v))
                        for k, v in parsed.items()
                    )
                except Exception:
                    args_fmt = raw_args
                parts.append("{}({})  id={}".format(name, args_fmt, call_id))
            _alog("│ [{:>2}] {}  [TOOL_CALL]".format(idx, role))
            for p in parts:
                _alog("│       call: {}".format(p))
            if content:
                _alog("│       content:")
                for line in _render_content(str(content)):
                    _alog(line)
        elif role == 'TOOL':
            text_content = str(content)
            _alog("│ [{:>2}] TOOL RESULT  call_id={}  ({:,} chars)".format(
                idx, tool_call_id, len(text_content)))
            for line in _render_content(text_content):
                _alog(line)
        else:
            text_content = str(content)
            _alog("│ [{:>2}] {}  ({:,} chars)".format(idx, role, len(text_content)))
            for line in _render_content(text_content):
                _alog(line)

        _alog("│")

    _alog("└" + "─" * 77 + "┘")
    _alog("")

# ─── Diagnostic output (stderr, TTY-only) ────────────────────────────────────
# Diagnostics are dim/faint so the final result is easy to spot.
# When stderr is not a TTY (e.g. piped) all diagnostic output is suppressed.
_STDERR_TTY = sys.stderr.isatty()
_DIM   = "\033[2m"   # faint — signals "not the result"
_RESET = "\033[0m"
# Start pessimistic: stdout may have written content without a trailing newline,
# so the first diagnostic needs to push itself onto a fresh line.
_at_bol = False  # is the terminal cursor currently at beginning-of-line?

def _diag(text, end='\n'):
    """Write a diagnostic line to stderr only when stderr is a TTY.

    Always starts on its own line: if the cursor is mid-line (e.g. after a
    streamed stdout chunk) a leading newline is emitted first.
    """
    global _at_bol
    if not _STDERR_TTY:
        return
    prefix = '' if _at_bol else '\n'
    sys.stderr.write(f"{prefix}{_DIM}{text}{_RESET}{end}")
    sys.stderr.flush()
    _at_bol = end.endswith('\n') if end else False

def _note_output(text):
    """Update BOL state after writing *text* to any terminal stream."""
    global _at_bol
    if text:
        _at_bol = text.endswith('\n')


# ---------------------------------------------------------------------------
# call_llm error hierarchy
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """Base class for errors raised by call_llm."""

class MaxTurnsReached(LLMError):
    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        super().__init__(f"Reached maximum turns ({max_turns}) without a final response.")

# Known model capability profiles. Matched by prefix against the model ID.
# tier:    "small" | "mid" | "large" — drives prompt verbosity, max_turns defaults
# context: context window in tokens
# tool_calling: whether the model reliably supports tool/function calling
MODEL_PROFILES = {
    "arcee-ai/trinity-mini":          {"tier": "small", "context": 8192,   "tool_calling": True},
    "openai/gpt-4o-mini":             {"tier": "mid",   "context": 128000, "tool_calling": True},
    "openai/gpt-4o":                  {"tier": "large", "context": 128000, "tool_calling": True},
    "openai/o1":                      {"tier": "large", "context": 128000, "tool_calling": True},
    "openai/o3":                      {"tier": "large", "context": 200000, "tool_calling": True},
    "anthropic/claude-3-haiku":       {"tier": "small", "context": 200000, "tool_calling": True},
    "anthropic/claude-3-5-haiku":     {"tier": "mid",   "context": 200000, "tool_calling": True},
    "anthropic/claude-3-5-sonnet":    {"tier": "large", "context": 200000, "tool_calling": True},
    "anthropic/claude-3-7-sonnet":    {"tier": "large", "context": 200000, "tool_calling": True},
    "anthropic/claude-opus-4":        {"tier": "large", "context": 200000, "tool_calling": True},
    "anthropic/claude-sonnet-4":      {"tier": "large", "context": 200000, "tool_calling": True},
    "google/gemini-2.0-flash":        {"tier": "mid",   "context": 1000000, "tool_calling": True},
    "google/gemini-2.5-pro":          {"tier": "large", "context": 1000000, "tool_calling": True},
    "meta-llama/llama-3.3-70b":       {"tier": "mid",   "context": 128000, "tool_calling": True},
    "meta-llama/llama-3.1-8b":        {"tier": "small", "context": 128000, "tool_calling": True},
    "mistralai/mistral-small":        {"tier": "small", "context": 32000,  "tool_calling": True},
    "mistralai/mistral-large":        {"tier": "large", "context": 128000, "tool_calling": True},
    "deepseek/deepseek-r1":           {"tier": "large", "context": 64000,  "tool_calling": False},
    "deepseek/deepseek-chat":         {"tier": "mid",   "context": 64000,  "tool_calling": True},
    # Conservative fallback for any unrecognised model
    "_default":                       {"tier": "mid",   "context": 16000,  "tool_calling": True},
}

AGENTIC_SYSTEM_PROMPTS = {
    'small': (
        "You are a CLI assistant with access to tools. "
        "You MUST use the available tools to answer questions — do not guess or make up information. "
        "Follow this process: 1) Call the appropriate tool. 2) Read the result carefully. "
        "3) Answer based only on what the tool returned. "
        "Do not ask clarifying questions. Output only the final answer, nothing else."
    ),
    'mid': (
        "You are a CLI assistant with access to tools. "
        "Use tools to gather information before answering. "
        "Do not ask clarifying questions — make reasonable assumptions. "
        "Be concise in your final response."
    ),
    'large': (
        "You are a CLI assistant. Use available tools to answer accurately. "
        "Do not ask clarifying questions. Be concise."
    ),
}

def get_model_profile(model_id):
    """Return capability profile for model_id.

    Tries exact match first, then prefix match (longest prefix wins),
    then applies any per-model overrides from config.json under 'model_profiles'.
    Falls back to '_default' if nothing matches.
    """
    # Exact match
    profile = MODEL_PROFILES.get(model_id)

    # Prefix match — pick the longest matching prefix
    if profile is None:
        best_prefix = ""
        for prefix, p in MODEL_PROFILES.items():
            if prefix == "_default":
                continue
            if model_id.startswith(prefix) and len(prefix) > len(best_prefix):
                best_prefix = prefix
                profile = p

    if profile is None:
        profile = MODEL_PROFILES["_default"]

    profile = dict(profile)  # shallow copy so overrides don't mutate the source

    # Apply user overrides from config.json: {"model_profiles": {"anthropic/claude-opus-4.6": {"context": 32000}}}
    user_overrides = config.get("model_profiles", {}) if 'config' in globals() and config else {}
    # Exact override
    if model_id in user_overrides:
        profile.update(user_overrides[model_id])
    else:
        # Prefix override
        for prefix, overrides in user_overrides.items():
            if model_id.startswith(prefix):
                profile.update(overrides)
                break

    return profile

def _tools_completer(prefix, **kwargs):
    """Completer for --tools: file paths for external MCPs, tool names for internal."""
    import glob as _glob
    import re as _re

    # If it looks like a path, complete as file
    if prefix.startswith('/') or prefix.startswith('./') or prefix.startswith('../') or os.sep in prefix:
        matches = _glob.glob(prefix + '*')
        return matches

    # Otherwise complete internal tool names from tools.py
    tools_file = os.path.join(os.path.dirname(__file__), 'tools.py')
    try:
        with open(tools_file) as f:
            content = f.read()
        names = _re.findall(r'@mcp\.tool\(\)\s+def\s+(\w+)', content)
        return [n for n in names if n.startswith(prefix)]
    except Exception:
        return []

def setup_shell_completion():
    config_dir = os.path.expanduser("~/.config/cai")
    flag = os.path.join(config_dir, "completion_setup")
    if os.path.exists(flag):
        return

    shell = os.path.basename(os.environ.get("SHELL", ""))
    eval_line = None
    rc_file = None

    if shell == "bash":
        eval_line = 'eval "$(register-python-argcomplete cai)"\n'
        rc_file = os.path.expanduser("~/.bashrc")
    elif shell == "zsh":
        eval_line = (
            'autoload -U bashcompinit && bashcompinit\n'
            'eval "$(register-python-argcomplete cai)"\n'
        )
        rc_file = os.path.expanduser("~/.zshrc")
    elif shell == "fish":
        eval_line = 'register-python-argcomplete --shell fish cai | source\n'
        rc_file = os.path.expanduser("~/.config/fish/config.fish")

    if rc_file and eval_line:
        try:
            with open(rc_file, "a") as f:
                f.write(f"\n# cai shell completion\n{eval_line}")
            open(flag, "w").close()
            print(f"[*] Shell completion added to {rc_file}. Run: source {rc_file}")
        except OSError:
            pass

def init():
    global config
    global available_tools
    global api_key
    global openai_api
    global openrouter_api
    global call_tool
    global call_external_tool
    global get_external_tools

    log.info("init: starting")

    import cai.api as _cai_api
    import cai.tools as _cai_tools
    OpenAiApi = _cai_api.OpenAiApi
    OpenRouterApi = _cai_api.OpenRouterApi
    get_tools = _cai_tools.get_tools
    call_tool = _cai_tools.call_tool
    get_external_tools = _cai_tools.get_external_tools
    call_external_tool = _cai_tools.call_external_tool

    config_dir = os.path.expanduser("~/.config/cai")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    config_path = os.path.join(config_dir, "config.json")
    if not os.path.exists(config_path):
        default_config = {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "arcee-ai/trinity-mini:free",
            "context_budget_pct": 0.75,
            "tool_result_max_chars": 8000,
            "ssl_verify": True,
            "model_profiles": {k: dict(v) for k, v in MODEL_PROFILES.items()}
        }
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        print(f"[*] Created default config at {config_path}")

    api_key_path = os.path.join(config_dir, "api_key")
    if not os.path.exists(api_key_path):
        with open(api_key_path, "w") as f:
            f.write("")
        print(f"[*] Created empty api_key file at {api_key_path}")

    config = json.loads(open(config_path).read())

    # Backfill keys added in newer versions so existing configs stay up-to-date.
    updated = False
    if 'ssl_verify' not in config:
        config['ssl_verify'] = True
        updated = True
    if updated:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    available_tools = get_tools()
    api_key = open(api_key_path).read().strip()
    openai_api = OpenAiApi(config.get('base_url'), api_key, ssl_verify=config.get('ssl_verify', True))
    openrouter_api = OpenRouterApi(api_key)
    log.info("init: done (base_url=%s, available_tools=%d)", config.get('base_url'), len(available_tools))

    # models = openrouter_api.get_models()
    # stats = openrouter_api.get_account_stats()
    # price_so_far = stats.get('data', {}).get('usage')

def get_model_context_length(model):
    global models
    for _model in models:
        if _model.get('id') == model:
            return _model.get('context_length')

def read_stdin_if_available():
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None

def _execute_tool(call_name, arguments, allowed_tool_names, usage=None, profile=None):
    """Validate and run a single tool call. Always returns a result string.

    allowed_tool_names is the set of tool names actually sent to the LLM for
    this session — validates against it so disabled tools cannot be executed
    even if the LLM hallucinates a call to one.
    """
    if call_name not in allowed_tool_names:
        log.warning("tool call: rejected tool '%s' (not in selected tools)", call_name)
        return f"Error: tool '{call_name}' is not enabled. Enabled tools: {sorted(allowed_tool_names)}"

    try:
        call_args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        log.warning("tool call: bad JSON args for '%s': %s", call_name, e)
        return f"Error: tool '{call_name}' received invalid JSON arguments: {e}. Raw arguments: {arguments!r}"

    try:
        for mcp_path in external_mcps:
            for tool in external_mcps[mcp_path]:
                if tool.get('function', {}).get('name') == call_name:
                    log.info("tool call: %s (external, mcp=%s) args=%s", call_name, mcp_path, call_args)
                    result = call_external_tool(mcp_path, call_name, call_args)
                    if result is None:
                        return f"Error: tool '{call_name}' returned no result"
                    log.info("tool call: %s -> result length=%d", call_name, len(result))
                    return trim_tool_result(result, usage=usage, profile=profile)
        log.info("tool call: %s (internal) args=%s", call_name, call_args)
        result = call_tool(call_name, call_args)
    except Exception as e:
        log.error("tool call: %s raised: %s", call_name, e)
        return f"Error: tool '{call_name}' raised an exception: {e}"

    if result is None:
        log.warning("tool call: %s returned None", call_name)
        return f"Error: tool '{call_name}' returned no result"
    log.info("tool call: %s -> result length=%d", call_name, len(result))
    return trim_tool_result(result, usage=usage, profile=profile)

def handle_tool_calls(tool_calls, messages, call_content, allowed_tool_names, tool_callback=None, usage=None, profile=None):
    log.info("handle_tool_calls: dispatching %d tool call(s)", len(tool_calls))
    for call in tool_calls:
        if call.get('type') != 'function':
            log.warning("tool call with invalid type: %s", call.get('type'))
            continue
        call_id = call.get('id')
        call_function = call.get('function', {})
        call_name = call_function.get('name')
        arguments = call_function.get('arguments') or ''

        # Log the dispatch before execution so failures are visible in order
        try:
            parsed_args = json.loads(arguments) if arguments else {}
            args_preview = ', '.join(
                "{}={}{}".format(k, json.dumps(v)[:80], '...' if len(json.dumps(v)) > 80 else '')
                for k, v in parsed_args.items()
            )
        except Exception:
            args_preview = arguments[:160]
        _alog("  >>> TOOL CALL: {}({})".format(call_name, args_preview))

        result = _execute_tool(call_name, arguments, allowed_tool_names, usage=usage, profile=profile)

        if result.startswith("Error:"):
            _alog("  <<< TOOL ERROR: {} — {}".format(call_name, result[:200]))
        else:
            preview = result[:200]
            ellipsis = '...' if len(result) > 200 else ''
            _alog("  <<< TOOL RESULT: {} — {:,} chars | {}{}".format(
                call_name, len(result), preview, ellipsis))

        if tool_callback:
            if result.startswith("Error:"):
                tool_callback(f"  \u2717 {call_name}: {result}\n", error=True)
            else:
                tool_callback(f"  <- {call_name}: {len(result)} chars\n")

        assistant_msg = {
            'role': 'assistant',
            'content': call_content or '',
            'tool_calls': [{'id': call_id, 'type': 'function',
                            'function': {'name': call_name, 'arguments': arguments}}],
        }
        tool_msg = {'role': 'tool', 'tool_call_id': call_id, 'content': result}
        messages.append(assistant_msg)
        messages.append(tool_msg)

def _retry_until_format(call_fn, system_prompt, check_fn, fail_msg_fn, format_label, messages, max_attempts):
    """Shared retry loop for format enforcement.

    call_fn      -- callable returning (content, reasoning, tool_calls, usage) or falsy
    system_prompt -- injected as a system message before the first attempt (if messages given)
    check_fn     -- callable(content) -> (ok: bool, normalised_content: str)
    fail_msg_fn  -- callable(attempt, max_attempts, content) -> str  (user feedback on failure)
    format_label -- short label used in stderr messages
    messages     -- conversation list mutated between retries; may be None
    max_attempts -- maximum number of LLM calls
    """
    if messages is not None:
        messages.insert(0, {'role': 'system', 'content': system_prompt})

    for attempt in range(1, max_attempts + 1):
        result = call_fn()
        if not result:
            return result
        orig_content, reasoning, tool_calls, usage = result

        # do not enforce format when tool calls are present
        if tool_calls:
            return orig_content, reasoning, tool_calls, usage

        ok, normalised = check_fn(orig_content)
        if ok:
            return normalised, reasoning, tool_calls, usage

        log.error(f"failed to get requested format from LLM: {format_label=} -> {orig_content=}, {reasoning=}, {tool_calls=}")
        _diag(f"[enforce_strict_format] attempt {attempt}/{max_attempts}: {fail_msg_fn(attempt, max_attempts, orig_content)}")
        if attempt == max_attempts:
            _diag(f"[enforce_strict_format] giving up after {max_attempts} attempts")
            return None
        if messages is not None:
            messages.append({'role': 'user', 'content': fail_msg_fn(attempt, max_attempts, orig_content)})

def _check_json(content):
    try:
        return True, json.dumps(json.loads(content))
    except Exception:
        return False, content

def _check_regex(pattern, content):
    import re
    return (True, content) if re.search(pattern, content) else (False, content)

def _check_regex_each_line(pattern, content):
    import re

    for line in content.splitlines():
        if not re.search(pattern, line):
            return (False, content)
    return (True, content)

def enforce_strict_format(call_fn, strict_format, messages=None, max_attempts=4):
    """Retry call_fn() until its content matches strict_format.
    call_fn must return (content, reasoning, tool_calls, usage) or None/falsy.
    messages, if provided, is mutated between retries to inject feedback.

    Supported values for strict_format:
      'json'                    -- response must be a valid JSON object
      'regex:<pattern>'         -- response must match the given regex pattern
      'regex-each-line:<pattern>' -- apply regex to each line; succeeds if any line matches
    """
    if strict_format == 'json':
        return _retry_until_format(
            call_fn,
            system_prompt='Respond only with a valid JSON object. Do not include markdown fences, explanations, or any text outside the JSON.',
            check_fn=_check_json,
            fail_msg_fn=lambda attempt, max_attempts, content: (
                f"Your previous response was not valid JSON (attempt {attempt}/{max_attempts}). "
                "Please respond with only a valid JSON object. "
                "Do not include markdown fences, explanations, or any text outside the JSON."
            ),
            format_label='json',
            messages=messages,
            max_attempts=max_attempts,
        )

    if strict_format and strict_format.startswith('regex-each-line:'):
        pattern = strict_format[len('regex-each-line:'):]
        return _retry_until_format(
            call_fn,
            system_prompt=f'At least one line of your response must match the regular expression pattern: {pattern}',
            check_fn=lambda content: _check_regex_each_line(pattern, content),
            fail_msg_fn=lambda attempt, max_attempts, content: (
                f"Your previous response did not have any line matching the required pattern (attempt {attempt}/{max_attempts}). "
                f"Please ensure at least one line matches the regular expression: {pattern}"
            ),
            format_label=f'regex-each-line:{pattern}',
            messages=messages,
            max_attempts=max_attempts,
        )

    if strict_format and strict_format.startswith('regex:'):
        pattern = strict_format[len('regex:'):]
        return _retry_until_format(
            call_fn,
            system_prompt=f'Your response must match the regular expression pattern: {pattern}',
            check_fn=lambda content: _check_regex(pattern, content),
            fail_msg_fn=lambda attempt, max_attempts, content: (
                f"Your previous response did not match the required pattern (attempt {attempt}/{max_attempts}). "
                f"Please ensure your response matches the regular expression: {pattern}"
            ),
            format_label=f'regex:{pattern}',
            messages=messages,
            max_attempts=max_attempts,
        )

    return call_fn()

def _run_nonstreaming_turn(messages, args, included_tools, stream_callback=None, tool_choice="auto", interrupt_event=None):
    """Single non-streaming LLM call. Returns (content, tool_calls, usage)."""
    result = enforce_strict_format(lambda: openai_api.chat( messages,
                                                            model=args.model,
                                                            tools=included_tools,
                                                            tool_choice=tool_choice),
                                    args.strict_format,
                                    messages=messages)

    if not result: return "", None, {}

    content, _reasoning, tool_calls, usage = result
    return content or "", tool_calls, usage

def _run_streaming_turn(messages, args, included_tools, stream_callback, tool_choice="auto", interrupt_event=None):
    """Single streaming LLM call. Returns (accumulated_text, tool_calls, usage)."""
    accumulated = []
    last_tool_calls = None
    usage = {}
    for chunk, tool_calls, usage in openai_api.chat_stream(messages,
                                                           model=args.model,
                                                           tools=included_tools,
                                                           tool_choice=tool_choice):
        if interrupt_event and interrupt_event.is_set():
            break
        if chunk:
            accumulated.append(chunk)
            if stream_callback:
                stream_callback(chunk)
        if tool_calls:
            last_tool_calls = tool_calls
    return "".join(accumulated), last_tool_calls, usage

CONTEXT_BUDGET_THRESHOLDS = {'small': 0.60, 'mid': 0.75, 'large': 0.80}

def _compact_messages(messages, model):
    """Summarize middle turns into a memory message to free up context space."""
    # Determine the slice to compact: after [system?][first_user], before last 4 messages
    start_idx = 0
    if messages and messages[0].get('role') == 'system':
        start_idx = 1
    if start_idx < len(messages) and messages[start_idx].get('role') == 'user':
        start_idx += 1

    end_idx = max(start_idx, len(messages) - 4)
    compactable = messages[start_idx:end_idx]
    if len(compactable) < 2:
        log.info("compaction: not enough messages to compact (%d)", len(compactable))
        _alog("  [TRIM] skipped — only {} messages in compactable slice (need ≥2)".format(len(compactable)))
        return

    _alog("  [TRIM] compacting {} messages (indices {}–{}) into a single memory entry...".format(
        len(compactable), start_idx, end_idx - 1))

    compaction_msgs = [{"role": "user", "content": (
        "Summarize the following conversation turns into a concise memory entry. "
        "Preserve all key facts, tool results, findings, and decisions. "
        "Be specific and concrete. Output only the summary, no preamble.\n\n"
        f"{json.dumps(compactable, indent=2)}"
    )}]

    result = openai_api.chat(compaction_msgs, model=model)
    if not result:
        log.warning("compaction: LLM call failed, skipping")
        _alog("  [TRIM] compaction LLM call failed — messages unchanged")
        return

    summary, _, _, _ = result
    memory = {"role": "system", "content": f"[memory from compacted turns]: {summary}"}
    log.info("compaction: replaced %d messages with memory (%d chars)", len(compactable), len(summary))
    messages[start_idx:end_idx] = [memory]
    _alog("  [TRIM] done — replaced {} messages with 1 [memory] entry ({} chars)".format(
        len(compactable), len(summary)))
    preview = summary[:200]
    ellipsis = '...' if len(summary) > 200 else ''
    _alog("  [TRIM]   memory preview: {}{}".format(preview, ellipsis))

def _check_context_budget(messages, usage, profile, args, status_callback=None):
    """Compact messages if prompt token usage exceeds the tier threshold."""
    prompt_tokens = usage.get('prompt_tokens', 0)
    context_limit = profile.get('context', 16000)
    if not prompt_tokens or not context_limit:
        return

    budget_pct = prompt_tokens / context_limit
    default_threshold = CONTEXT_BUDGET_THRESHOLDS.get(profile['tier'], 0.75)
    threshold = config.get('context_budget_pct', default_threshold) \
        if 'config' in globals() and config else default_threshold

    _alog("  [CONTEXT] {:.0%} used — {:,} / {:,} tokens  (threshold {:.0%})".format(
        budget_pct, prompt_tokens, context_limit, threshold))

    if budget_pct >= threshold:
        log.warning("context budget: %.0f%% used (%d/%d tokens), compacting",
                    budget_pct * 100, prompt_tokens, context_limit)
        msg = f"[context {budget_pct:.0%} >= {threshold:.0%}] compacting..."
        _alog("  [CONTEXT] budget exceeded — triggering trim/compaction")
        if status_callback:
            status_callback(msg)
        else:
            _diag(f"\n{msg}")
        _compact_messages(messages, args.model)

def _warn_if_stuck(tool_calls, call_history, messages):
    """Track repeated identical tool calls and inject a warning into messages when stuck."""
    for call in tool_calls:
        name = call.get('function', {}).get('name', '')
        args_str = call.get('function', {}).get('arguments', '')
        key = (name, args_str)
        call_history[key] = call_history.get(key, 0) + 1
        if call_history[key] >= 2:
            warning = (f"Warning: you have already called tool '{name}' with these exact arguments "
                       f"{call_history[key]} times and received the same result. "
                       f"Try a different approach or tool to make progress.")
            log.warning("stuck detection: '%s' called %d times with same args", name, call_history[key])
            _alog("  [STUCK] '{}' called {} times with identical args — injecting warning into context".format(
                name, call_history[key]))
            messages.append({"role": "user", "content": warning})

def _emit_status(text, status_callback):
    if status_callback:
        status_callback(text)

def call_llm(messages,
             args,
             available_tools,
             external_mcps,
             stream_callback=None,
             status_callback=None,
             tool_callback=None,
             ctx_callback=None,
             interrupt_event=None):
    # Build the tool list actually sent to the LLM:
    # only internal tools the user has selected, plus all external MCP tools.
    selected_tool_names = getattr(args, 'selected_tools', set())
    included_tools = [
        tool for tool in available_tools
        if tool.get('function', {}).get('name') in selected_tool_names
    ]
    for mcp_path in external_mcps:
        included_tools.extend(external_mcps[mcp_path])

    # Names the LLM was given — used to gate execution in _execute_tool.
    allowed_tool_names = {t.get('function', {}).get('name') for t in included_tools}

    profile = get_model_profile(args.model)

    tier_defaults = {'small': 5, 'mid': 10, 'large': 20}
    default_max_turns = tier_defaults.get(profile['tier'], 10)
    max_turns = getattr(args, 'max_turns', None) or default_max_turns

    log.info("call_llm: model=%s tier=%s context=%d messages=%d tools=%d streaming=%s strict_format=%s max_turns=%d",
             args.model,
             profile['tier'],
             profile['context'],
             len(messages),
             len(included_tools),
             stream_callback,
             args.strict_format or "none",
             max_turns)

    # ── Agent log: session header ─────────────────────────────────────────────
    _alog("=" * 80)
    _alog("CALL_LLM START  {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    _alog("  model:      {}".format(args.model))
    _alog("  tier/ctx:   {} / {:,} tokens".format(profile['tier'], profile['context']))
    _alog("  tools:      {}  ({} total)".format(
        ', '.join(sorted(allowed_tool_names)) if allowed_tool_names else '(none)',
        len(included_tools)))
    _alog("  max_turns:  {}".format(max_turns))
    _alog("  streaming:  {}".format('yes' if stream_callback else 'no'))
    _alog("  strict_fmt: {}".format(getattr(args, 'strict_format', None) or 'none'))
    _alog("  force_tools:{}".format(getattr(args, 'force_tools', False)))
    _alog("  messages:   {} (initial)".format(len(messages)))
    _alog("")
    _alog_messages_state(messages, "initial context")
    # ─────────────────────────────────────────────────────────────────────────

    call_history = {}  # (tool_name, args_str) -> call count, for stuck detection

    if stream_callback:
        run_turn = _run_streaming_turn
    else:
        run_turn = _run_nonstreaming_turn

    for turn in range(1, max_turns + 1):
        # Force at least one tool call on turn 1 in agentic mode so the model
        # doesn't skip tools and answer directly from training data.
        if args.force_tools and turn == 1:
            tool_choice = "required"
        else:
            tool_choice = "auto"

        _alog("=== TURN {}/{} ===  [{}]{}".format(
            turn, max_turns,
            datetime.datetime.now().strftime("%H:%M:%S"),
            "  tool_choice=required (force_tools)" if tool_choice == "required" else ""))
        _alog("")

        # Show exact context the LLM is about to receive
        _alog_messages_state(messages, "turn {} — LLM input".format(turn))

        content, tool_calls, usage = run_turn(messages,
                                              args,
                                              included_tools,
                                              stream_callback,
                                              tool_choice=tool_choice,
                                              interrupt_event=interrupt_event)
        log.info("call_llm: turn=%d tokens prompt=%s completion=%s total=%s",
                 turn,
                 usage.get('prompt_tokens'),
                 usage.get('completion_tokens'),
                 usage.get('total_tokens'))

        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        pct = f"{prompt_tokens / profile['context']:.0%}" if profile['context'] else "?"
        ctx_str = f"ctx {pct} ({prompt_tokens}/{profile['context']})"
        if ctx_callback:
            ctx_callback(ctx_str)

        _alog("  tokens: prompt={:,}  completion={:,}  ({} of context window)".format(
            prompt_tokens, completion_tokens, pct))
        _alog("")

        if not tool_calls:
            log.info("call_llm: done turn=%d length=%d", turn, len(content))

            # Show the final state: existing messages + the assistant's response
            final_state = messages + [{"role": "assistant", "content": content}]
            _alog_messages_state(final_state, "turn {} — FINAL STATE (response appended)".format(turn))

            _alog("CALL_LLM DONE  turns={}  response_len={:,}  {}".format(
                turn, len(content), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            _alog("=" * 80)
            _alog("")

            _emit_status("ready", status_callback)
            return content

        if not getattr(args, 'oneline', False):
            def _fmt_call(c):
                name = c.get('function', {}).get('name', '?')
                raw_args = c.get('function', {}).get('arguments', '')
                try:
                    parsed = json.loads(raw_args) if raw_args else {}
                    args_str = ", ".join(
                        f"{k}={json.dumps(v)[:80]}{'...' if len(json.dumps(v)) > 80 else ''}"
                        for k, v in parsed.items()
                    )
                except Exception:
                    args_str = raw_args[:160] + ("..." if len(raw_args) > 160 else "")
                return f"{name}({args_str})"

            tool_calls_fmt = [_fmt_call(c) for c in tool_calls]
            status = f"[turn {turn}/{max_turns}] {', '.join(tool_calls_fmt)}"
            _emit_status(status, status_callback)
            if tool_callback:
                for fmt in tool_calls_fmt:
                    tool_callback(f"-> {fmt}\n")

        # handle_tool_calls appends assistant + tool messages; it logs each
        # dispatch/result event.  We snapshot the array after.
        handle_tool_calls(tool_calls, messages, content, allowed_tool_names, tool_callback=tool_callback, usage=usage, profile=profile)
        if tool_callback:
            tool_callback("\n")
        _alog_messages_state(messages, "turn {} — after tool execution".format(turn))

        # _warn_if_stuck may append a [USER] warning into messages
        n_before_stuck = len(messages)
        _warn_if_stuck(tool_calls, call_history, messages)
        if len(messages) > n_before_stuck:
            _alog_messages_state(messages, "turn {} — after stuck-warning injection".format(turn))

        # _check_context_budget may compact (replace) messages
        n_before_compact = len(messages)
        _check_context_budget(messages, usage, profile, args, status_callback)
        if len(messages) != n_before_compact:
            _alog_messages_state(messages, "turn {} — after compaction".format(turn))

        _alog("")

    log.warning("call_llm: reached max_turns=%d", max_turns)
    _alog("MAX TURNS REACHED ({})  {}".format(max_turns, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    _alog_messages_state(messages, "max-turns final state")
    _alog("=" * 80)
    _alog("")
    _emit_status(f"[!] reached max turns ({max_turns})", status_callback)
    raise MaxTurnsReached(max_turns)

def prompt_line_by_line(args, messages, available_tools, external_mcps):
    if not sys.stdin.isatty():
        log.info("prompt_line_by_line: mode=streaming_stdin")
        streaming_stdin = True
        lines = None
    elif args.file:
        streaming_stdin = False
        with open(args.file) as f:
            lines = [l.rstrip('\n') for l in f if l.strip()]
        log.info("prompt_line_by_line: mode=file file=%s lines=%d", args.file, len(lines))
    else:
        print("--line-by-line requires piped stdin or --file.")
        return

    if lines is not None and not lines:
        print("[!] no lines to process.")
        return

    total = None if streaming_stdin else len(lines)
    completed_count = [0]
    lock = threading.Lock()

    def update_progress(completed):
        if args.progress:
            if total is not None:
                bar_len = 30
                filled = int(bar_len * completed / total)
                bar = '█' * filled + '░' * (bar_len - filled)
                _diag(f'\rProgress: [{bar}] {completed}/{total} ', end='')
                if completed == total:
                    _diag('', end='\n')

    def process_line(line):
        if shutdown_event.is_set():
            return
        local_messages = messages.copy()
        local_messages.append({"role": "user", "content": line})
        local_messages.append({"role": "user", "content": args.prompt})

        response = call_llm(local_messages, args, available_tools, external_mcps,
                            interrupt_event=shutdown_event)

        with lock:
            completed_count[0] += 1
            update_progress(completed_count[0])
            if args.oneline:
                oneline_response = response.replace('\n', ' ')
                print(f"{line}:{oneline_response}", flush=True)
            else:
                count_str = f"{completed_count[0]}/{total}" if total is not None else str(completed_count[0])
                print(f"\n{'─' * 80}")
                print(f"[{count_str}] {line}")
                print('─' * 80)
                if response:
                    print(response)

    shutdown_event = threading.Event()

    original_sigint = signal.getsignal(signal.SIGINT)

    executor = ThreadPoolExecutor(max_workers=args.cores)
    futures = []

    def _sigint_handler(signum, frame):
        shutdown_event.set()
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        if streaming_stdin:
            for raw_line in sys.stdin:
                if shutdown_event.is_set():
                    break
                line = raw_line.rstrip('\n')
                if line.strip():
                    futures.append(executor.submit(process_line, line))
        else:
            futures = [executor.submit(process_line, line) for line in lines]

        for future in as_completed(futures):
            if shutdown_event.is_set():
                break
            try:
                future.result()
            except Exception as e:
                log.error("thread error in prompt_line_by_line: %s", e)
                with lock:
                    print(f"[!] thread error: {e}")
    except KeyboardInterrupt:
        shutdown_event.set()
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown(wait=True)
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        if shutdown_event.is_set():
            _diag("\n[!] interrupted — all pending tasks cancelled.")

TOOL_RESULT_MAX_CHARS = 8000
_CHARS_PER_TOKEN = 4  # approximate for mixed code/text content

def _dynamic_max_chars(usage, profile):
    """Return a max_chars limit based on remaining context window, or None if no trimming needed.

    Strategy: when more than 50% of the context window remains, don't trim at all.
    As the window fills, progressively reserve a smaller fraction for each result so
    we don't blow the budget in a single call.
    """
    if not usage or not profile:
        return None
    prompt_tokens = usage.get('prompt_tokens', 0)
    context_limit = profile.get('context', 0)
    if not prompt_tokens or not context_limit:
        return None
    tokens_remaining = max(0, context_limit - prompt_tokens)
    remaining_fraction = tokens_remaining / context_limit
    if remaining_fraction > 0.5:
        return None  # plenty of headroom — skip trimming entirely
    # Shrink the per-result budget as context tightens:
    #   50% remaining → allocate 25% of remaining  (≈ large result still allowed)
    #   25% remaining → allocate 12.5% of remaining
    #   10% remaining → allocate 10% (floor)  — gets aggressive
    budget_fraction = max(0.10, remaining_fraction * 0.5)
    max_chars = int(tokens_remaining * budget_fraction * _CHARS_PER_TOKEN)
    return max(max_chars, 2000)  # always allow at least 2000 chars

def trim_tool_result(result, usage=None, profile=None, max_chars=None):
    if max_chars is None:
        dynamic = _dynamic_max_chars(usage, profile)
        if dynamic is not None:
            max_chars = dynamic
            log.info("trim_tool_result: dynamic limit=%d (prompt_tokens=%s, context=%s)",
                     max_chars,
                     usage.get('prompt_tokens') if usage else None,
                     profile.get('context') if profile else None)
        else:
            max_chars = config.get('tool_result_max_chars', TOOL_RESULT_MAX_CHARS) \
                if 'config' in globals() and config else TOOL_RESULT_MAX_CHARS
    if max_chars and len(result) > max_chars:
        omitted = len(result) - max_chars
        log.info("trim_tool_result: trimmed %d chars (limit=%d)", omitted, max_chars)
        return result[:max_chars] + f"\n[truncated: {omitted} chars omitted]"
    return result


def _read_file_numbered(path):
    """Return file contents as a numbered-line string."""
    with open(path) as f:
        return "".join(f"{i + 1}: {line}" for i, line in enumerate(f))


def _build_base_messages(args, stdin_content=None, always_agentic=False):
    """Build the initial messages list (system prompt, stdin, file, cursor)."""
    messages = []

    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    elif always_agentic or getattr(args, 'agentic', False):
        profile = get_model_profile(args.model)
        prompt_text = AGENTIC_SYSTEM_PROMPTS.get(profile['tier'], AGENTIC_SYSTEM_PROMPTS['mid'])
        messages.append({"role": "system", "content": prompt_text})
        log.info("_build_base_messages: injected agentic system prompt (tier=%s)", profile['tier'])

    line_by_line = getattr(args, 'line_by_line', False)
    if stdin_content is None and not line_by_line:
        stdin_content = read_stdin_if_available()
    if stdin_content:
        messages.append({"role": "user", "content": stdin_content})

    if args.file and (not line_by_line or not sys.stdin.isatty()):
        log.info("_build_base_messages: including file %s", args.file)
        messages.append({"role": "user",
                         "content": f"<file_content> {_read_file_numbered(args.file)} </file_content>"})

    if args.cursor:
        m = re.match(r"^(?P<file_path>.*):(?P<line_num>\d+):(?P<col_num>\d+)$", args.cursor)
        if m:
            fp, ln, cn = m.group('file_path'), m.group('line_num'), m.group('col_num')
            log.info("_build_base_messages: including cursor %s:%s:%s", fp, ln, cn)
            messages.append({"role": "user",
                             "content": f"<file_content> {_read_file_numbered(fp)} </file_content>"})
            messages.append({"role": "user",
                             "content": f"<cursor_location> line number: {ln}, column number: {cn} </cursor_location>"})

    return messages


ACTION_PROMPT = "prompt"
def action_prompt(args, available_tools, external_mcps):
    if not args.prompt:
        print("this action require --prompt to be provided.")
        return

    log.info("action_prompt: model=%s file=%s cursor=%s line_by_line=%s oneline=%s",
             args.model, args.file, args.cursor, args.line_by_line, args.oneline)

    messages = _build_base_messages(args)

    if args.line_by_line:
        return prompt_line_by_line(args, messages, available_tools, external_mcps)

    messages.append({"role": "user", "content": args.prompt})

    def _stderr_tool(chunk, error=False):
        if _STDERR_TTY:
            sys.stderr.write(_DIM + chunk + _RESET)
            sys.stderr.flush()
            _note_output(chunk)

    def _stderr_status(text):
        if text:
            _diag(f"[{text}]")

    def _stderr_ctx(ctx_str):
        _diag(f"[{ctx_str}]")

    if not args.non_streaming and not args.oneline and not args.strict_format:
        log.info("action_prompt: calling LLM (streaming)")
        try:
            def _stream_to_stdout(chunk):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                _note_output(chunk)

            call_llm(messages,
                     args,
                     available_tools,
                     external_mcps,
                     stream_callback=_stream_to_stdout,
                     tool_callback=_stderr_tool,
                     status_callback=_stderr_status,
                     ctx_callback=_stderr_ctx)
            print()
        except LLMError as e:
            print()
            _diag(f"[error] {e}")
    else:
        log.info("action_prompt: calling LLM (non-streaming)")
        try:
            content = call_llm(messages,
                               args,
                               available_tools,
                               external_mcps,
                               tool_callback=_stderr_tool,
                               status_callback=_stderr_status,
                               ctx_callback=_stderr_ctx)
            if args.oneline:
                content = content.replace('\n', ' ')
            print(content)
        except LLMError as e:
            _diag(f"[error] {e}")

def _handle_interactive_cmd(cmd, screen, messages, args, status_callback, last_ctx):
    """Execute a vim-style colon command from interactive mode."""
    if cmd == "compact":
        status_callback("compacting...")
        try:
            _compact_messages(messages, args.model)
        except LLMError as e:
            screen.write(f"\033[1;31m[compact error] {e}\033[m\n")
        last_ctx[0] = ""
        status_callback("ready")
    elif cmd == "clear":
        # Keep only the system prompt (first message if role==system)
        if messages and messages[0].get('role') == 'system':
            messages[1:] = []
        else:
            messages.clear()
        profile = get_model_profile(args.model)
        last_ctx[0] = f"ctx 0% (0/{profile['context']})"
        status_callback("ready")
    elif cmd == "tools":
        tool_names = [t.get('function', {}).get('name') for t in available_tools
                      if t.get('function', {}).get('name')]
        new_selected = screen.prompt_tools_overlay(tool_names, args.selected_tools)
        args.selected_tools = new_selected
        status_callback("ready")
    elif cmd == "":
        pass  # empty command, do nothing
    else:
        screen.write(f"\033[2;37m[unknown command: {cmd}]\033[m\n")


ACTION_INTERACTIVE = "interactive"
def action_interactive(args, available_tools, external_mcps):
    """Multi-turn TUI conversation loop using Screen for display."""
    from cai.screen import Screen

    if not sys.stdout.isatty():
        _diag("[!] --interactive requires a TTY stdout.")
        return

    # Pre-capture piped stdin BEFORE Screen (tty.setraw) takes over keyboard input
    stdin_content = read_stdin_if_available()

    messages = _build_base_messages(args, stdin_content=stdin_content, always_agentic=True)

    screen = Screen()
    screen.set_cmd_completions(["compact", "tools", "clear"])

    last_ctx = [""]

    def _status(text=None):
        ctx_part = f" | {last_ctx[0]}" if last_ctx[0] else ""
        screen.set_status(f"{args.model}{ctx_part}")

    def status_callback(text=None):
        _status()

    def ctx_cb(ctx_str):
        last_ctx[0] = ctx_str
        _status()

    _LLM_STYLE   = Screen._LLM_STYLE
    _META_STYLE  = Screen._META_STYLE
    _ERROR_STYLE = Screen._ERROR_STYLE
    _RESET       = Screen._RESET

    def stream_cb(chunk):
        screen.write(chunk)

    def tool_cb(line, error=False):
        style = _ERROR_STYLE if error else _META_STYLE
        screen.write(f"{style}{line}{_RESET}{_LLM_STYLE}")

    try:
        pending_input = args.prompt  # seed loop with initial prompt if provided

        # Main interactive loop
        _status("ready")
        while True:
            if pending_input is not None:
                user_input = pending_input
                pending_input = None
                screen.write(f"{Screen._USER_STYLE}> {user_input}{Screen._RESET}\n\n")
            else:
                user_input = screen.prompt("> ")
            if not user_input.strip():
                continue
            if user_input.startswith(":"):
                _handle_interactive_cmd(user_input[1:].strip(), screen, messages, args, status_callback, last_ctx)
                continue
            messages.append({"role": "user", "content": user_input})
            status_callback("thinking...")
            screen.show_prompt_placeholder("> ")
            screen.write(_LLM_STYLE)
            screen.start_input_listener()
            try:
                response = call_llm(messages, args, available_tools, external_mcps,
                                    stream_callback=stream_cb,
                                    status_callback=status_callback,
                                    tool_callback=tool_cb,
                                    ctx_callback=ctx_cb,
                                    interrupt_event=screen._interrupt_event)
                if screen._interrupt_event.is_set():
                    screen.write(f"\n{_RESET}{_META_STYLE}[interrupted]{_RESET}\n\n")
                    status_callback("interrupted")
                    continue
                screen.write(f"\n{_RESET}\n")
                messages.append({"role": "assistant", "content": response})
            except LLMError as e:
                screen.write(f"\n{_RESET}{_ERROR_STYLE}[error] {e}{_RESET}\n\n")
                status_callback("error")
            except BaseException:
                screen.write(_RESET)
                raise
            finally:
                screen.stop_input_listener()

    except (KeyboardInterrupt, EOFError):
        screen.write(f"{_RESET}\n[exiting]\n")
    finally:
        screen.close()


def main():
    global external_mcps
    global config

    parser = argparse.ArgumentParser(description="cai is a command line utility to make use of LLM intelegent in multiple cases.")

    parser.add_argument("-a", "--action",
                        choices=[
                            ACTION_PROMPT,
                            ],
                        default=ACTION_PROMPT,
                        help="the actiont to be performed.")
    parser.add_argument("-p", "--prompt",
                        help="the prompt to send to the LLM.")
    parser.add_argument("--system-prompt",
                        help="the system prompt to send to the LLM.")
    parser.add_argument("--file",
                        help="file path to include in the LLM context.")
    parser.add_argument("--cursor",
                        help="the cursor location in the codebase to be used by the action. in the format of => <file_path>:<line_num>:<col_num>")
    parser.add_argument("--model", default=None,
                        help="the model to be used by the LLM")
    parser.add_argument("--force-tools", action="store_true",
                        help="require from llm to make tool use")
    parser.add_argument("--progress", action="store_true",
                        help="show progess bar.")
    parser.add_argument("--oneline", action="store_true",
                        help="print results in a oneline all data.")
    parser.add_argument("--strict-format", default=None,
                        help="the expected format from the LLM response: 'json', 'regex:<pattern>', or 'regex-each-line:<pattern>'.")
    parser.add_argument("--include-reasoning", action="store_true",
                        help="let the action know whether or not to include reasoning in the output.")
    parser.add_argument("--non-streaming", action="store_true",
                        help="let the action know whether or not to use the non-streaming api.")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="open a persistent TUI session; implies --agentic. Ctrl-C or Ctrl-D to exit.")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="max tool-call turns in agentic mode (default: 5/10/20 by model tier).")
    tools_arg = parser.add_argument('-t',
                        '--tools',
                        nargs='+',
                        default=[],
                        help="list of mcp tools to give the LLM. the tools come in the form of abosult paths to the python files implementing the mcp server.")
    tools_arg.completer = _tools_completer
    parser.add_argument('--cores', type=int, default=1,
                        help="number of parallel threads for the grep action (default: 4).")
    parser.add_argument('--line-by-line', action='store_true', default=False,
                        help="process stdin (or --file) one line at a time, calling LLM per line.")
    parser.add_argument('prompt_words', nargs='*',
                        help="prompt words after -- (alternative to -p)")
    parser.add_argument('--harness', default=None,
                        help="path to a .harness.cai orchestration file.")
    parser.add_argument('--logger', action='store_true',
                        help="launch the interactive hierarchical log viewer (reads /tmp/cai/cai.log).")

    # Must be called before init() so tab completion exits immediately without
    # running any heavy initialization (API clients, tree-sitter, etc.).
    argcomplete.autocomplete(parser)

    init()
    setup_shell_completion()

    args = parser.parse_args()
    if args.prompt_words:
        if args.prompt:
            parser.error("cannot use both -p/--prompt and trailing words after --")
        args.prompt = " ".join(args.prompt_words)
    if args.model is None:
        args.model = config.get('model', "arcee-ai/trinity-mini:free")

    external_mcps = {}
    args.selected_tools = set()
    for entry in args.tools:
        entry = entry.strip().rstrip(',')
        if not entry:
            continue
        if os.path.isfile(entry) or entry.endswith('.py'):
            log.info("main: loading external MCP %s", entry)
            external_mcps[entry] = get_external_tools(entry)
        else:
            log.info("main: enabling internal tool %s", entry)
            args.selected_tools.add(entry)

    if args.interactive:
        if args.line_by_line or args.oneline:
            parser.error("--interactive is incompatible with --line-by-line / --oneline")

    log.info("main: action=%s model=%s selected_tools=%s external_mcps=%s interactive=%s",
             args.action, args.model, sorted(args.selected_tools), list(external_mcps.keys()),
             args.interactive)

    if args.logger:
        from cai.logger import launch_tui
        launch_tui()
        return

    if args.harness:
        from cai.harness import execute_harness, parse_harness_file
        instructions, label_map = parse_harness_file(args.harness)
        user_prompt = args.prompt or ""
        execute_harness(instructions, label_map, user_prompt, args, available_tools)
        return

    if args.interactive:
        action_interactive(args, available_tools, external_mcps)
        return

    if args.action == ACTION_PROMPT:
        action_prompt(args, available_tools, external_mcps)


if __name__ == "__main__":
    main()
