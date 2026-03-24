import argparse
import argcomplete
import json
import sys
import os
import re
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


global config
global tools
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
    global tools
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
    tools = get_tools()
    api_key = open(api_key_path).read().strip()
    openai_api = OpenAiApi(config.get('base_url'), api_key)
    openrouter_api = OpenRouterApi(api_key)
    log.info("init: done (base_url=%s, tools=%d)", config.get('base_url'), len(tools))

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

def _execute_tool(call_name, arguments):
    """Validate and run a single tool call. Always returns a result string."""
    known_tool_names = {t.get('function', {}).get('name') for t in tools}
    for mcp_tools in external_mcps.values():
        known_tool_names.update(t.get('function', {}).get('name') for t in mcp_tools)

    if call_name not in known_tool_names:
        log.warning("tool call: unknown tool '%s'", call_name)
        return f"Error: unknown tool '{call_name}'. Available tools: {sorted(known_tool_names)}"

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
                    return trim_tool_result(result)
        log.info("tool call: %s (internal) args=%s", call_name, call_args)
        result = call_tool(call_name, call_args)
    except Exception as e:
        log.error("tool call: %s raised: %s", call_name, e)
        return f"Error: tool '{call_name}' raised an exception: {e}"

    if result is None:
        log.warning("tool call: %s returned None", call_name)
        return f"Error: tool '{call_name}' returned no result"
    log.info("tool call: %s -> result length=%d", call_name, len(result))
    return trim_tool_result(result)


def handle_tool_calls(tool_calls, messages, call_content, tool_callback=None):
    log.info("handle_tool_calls: dispatching %d tool call(s)", len(tool_calls))
    for call in tool_calls:
        if call.get('type') != 'function':
            log.warning("tool call with invalid type: %s", call.get('type'))
            continue
        call_id = call.get('id')
        call_function = call.get('function', {})
        call_name = call_function.get('name')
        arguments = call_function.get('arguments') or ''
        result = _execute_tool(call_name, arguments)
        if tool_callback:
            tool_callback(f"  <- {call_name}: {len(result)} chars\n")
        messages.append({
            'role': 'assistant',
            'content': call_content or '',
            'tool_calls': [{'id': call_id, 'type': 'function',
                            'function': {'name': call_name, 'arguments': arguments}}],
        })
        messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': result})

def enforce_strict_format(call_fn, strict_format):
    """Retry call_fn() until its content matches strict_format.
    call_fn must return (content, reasoning, tool_calls) or None/falsy."""

    if strict_format == 'json':
        while True:
            result = call_fn()
            if not result: return result
            orig_content, reasoning, tool_calls, usage = result
            if tool_calls: # do not enforce format in case of tool calls
                return orig_content, reasoning, tool_calls, usage

            try:
                content = json.dumps(json.loads(orig_content))
                return content, reasoning, tool_calls, usage
            except Exception:
                log.error(f"failed to get requested format from LLM: {strict_format=} -> {orig_content=}, {reasoning=}, {tool_calls=}")
                continue
    return call_fn()

def _run_nonstreaming_turn(messages, args, included_tools, stream_callback=None, tool_choice="auto"):
    """Single non-streaming LLM call. Returns (content, tool_calls, usage)."""
    result = enforce_strict_format(
        lambda: openai_api.chat(messages, model=args.model, tools=included_tools,
                                tool_choice=tool_choice),
        args.strict_format,
    )
    if not result:
        return "", None, {}
    content, _reasoning, tool_calls, usage = result
    return content or "", tool_calls, usage

def _run_streaming_turn(messages, args, included_tools, stream_callback, tool_choice="auto"):
    """Single streaming LLM call. Returns (accumulated_text, tool_calls, usage)."""
    accumulated = []
    last_tool_calls = None
    usage = {}
    for chunk, tool_calls, usage in openai_api.chat_stream(
            messages, model=args.model, tools=included_tools, tool_choice=tool_choice):
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
        return

    compaction_msgs = [{"role": "user", "content": (
        "Summarize the following conversation turns into a concise memory entry. "
        "Preserve all key facts, tool results, findings, and decisions. "
        "Be specific and concrete. Output only the summary, no preamble.\n\n"
        f"{json.dumps(compactable, indent=2)}"
    )}]

    result = openai_api.chat(compaction_msgs, model=model)
    if not result:
        log.warning("compaction: LLM call failed, skipping")
        return

    summary, _, _, _ = result
    memory = {"role": "system", "content": f"[memory from compacted turns]: {summary}"}
    log.info("compaction: replaced %d messages with memory (%d chars)", len(compactable), len(summary))
    messages[start_idx:end_idx] = [memory]


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

    if budget_pct >= threshold:
        log.warning("context budget: %.0f%% used (%d/%d tokens), compacting",
                    budget_pct * 100, prompt_tokens, context_limit)
        msg = f"[context {budget_pct:.0%} >= {threshold:.0%}] compacting..."
        if status_callback:
            status_callback(msg)
        else:
            print(f"\n{msg}", file=sys.stderr, flush=True)
        _compact_messages(messages, args.model)


def _warn_if_stuck(tool_calls, call_history, messages):
    """Track repeated identical tool calls and inject a warning into messages when stuck."""
    for call in tool_calls:
        name = call.get('function', {}).get('name', '')
        args_str = call.get('function', {}).get('arguments', '')
        key = (name, args_str)
        call_history[key] = call_history.get(key, 0) + 1
        if call_history[key] >= 3:
            warning = (f"Warning: you have already called tool '{name}' with these exact arguments "
                       f"{call_history[key]} times and received the same result. "
                       f"Try a different approach or tool to make progress.")
            log.warning("stuck detection: '%s' called %d times with same args", name, call_history[key])
            messages.append({"role": "user", "content": warning})


def _emit_status(text, status_callback, stream_callback):
    if status_callback:
        status_callback(text)
    elif stream_callback:
        print(f"\n{text}", file=sys.stderr, flush=True)
    else:
        print(text, end="\r", file=sys.stderr, flush=True)


def call_llm(messages, args, stream_callback=None, status_callback=None, tool_callback=None, ctx_callback=None):
    global external_mcps

    # handling available tools for LLM.
    included_tools = []
    internal_tool_names = getattr(args, 'internal_tools', set())
    for tool in tools:
        tool_name = tool.get('function', {}).get('name')
        if tool_name in internal_tool_names:
            included_tools.append(tool)

    for mcp_path in external_mcps:
        included_tools.extend(external_mcps[mcp_path])

    # Streaming cannot enforce output format because the response is assembled
    # incrementally and validated only after completion. Fallback to non-streaming
    # when strict_format is requested so enforcement actually works.
    use_non_streaming = args.non_streaming or bool(args.strict_format)

    profile = get_model_profile(args.model)
    agentic = getattr(args, 'agentic', False)

    if agentic:
        tier_defaults = {'small': 5, 'mid': 10, 'large': 20}
        default_max_turns = tier_defaults.get(profile['tier'], 10)
        max_turns = getattr(args, 'max_turns', None) or default_max_turns
    else:
        max_turns = 1

    log.info("call_llm: model=%s tier=%s context=%d messages=%d tools=%d streaming=%s "
             "strict_format=%s agentic=%s max_turns=%d",
             args.model, profile['tier'], profile['context'], len(messages), len(included_tools),
             not use_non_streaming, args.strict_format or "none", agentic, max_turns)

    call_history = {}  # (tool_name, args_str) -> call count, for stuck detection
    run_turn = _run_nonstreaming_turn if use_non_streaming else _run_streaming_turn

    for turn in range(1, max_turns + 1):
        # Force at least one tool call on turn 1 in agentic mode so the model
        # doesn't skip tools and answer directly from training data.
        tool_choice = "required" if (agentic and turn == 1 and included_tools) else "auto"
        content, tool_calls, usage = run_turn(messages, args, included_tools, stream_callback,
                                              tool_choice=tool_choice)
        log.info("call_llm: turn=%d tokens prompt=%s completion=%s total=%s",
                 turn, usage.get('prompt_tokens'), usage.get('completion_tokens'), usage.get('total_tokens'))

        prompt_tokens = usage.get('prompt_tokens', 0)
        pct = f"{prompt_tokens / profile['context']:.0%}" if profile['context'] else "?"
        ctx_str = f"ctx {pct} ({prompt_tokens}/{profile['context']})"
        if ctx_callback:
            ctx_callback(ctx_str)

        if not tool_calls:
            log.info("call_llm: done turn=%d length=%d", turn, len(content))
            if agentic and not getattr(args, 'oneline', False):
                _emit_status("ready", status_callback, stream_callback)
            return content

        if agentic and not getattr(args, 'oneline', False):
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
            _emit_status(status, status_callback, stream_callback)
            if tool_callback:
                for fmt in tool_calls_fmt:
                    tool_callback(f"-> {fmt}\n")

        handle_tool_calls(tool_calls, messages, content, tool_callback=tool_callback)
        if tool_callback:
            tool_callback("\n")
        _warn_if_stuck(tool_calls, call_history, messages)

        if agentic:
            _check_context_budget(messages, usage, profile, args, status_callback)

        if not agentic:
            # Non-agentic: one follow-up call without tools, then done
            content, _, usage = run_turn(messages, args, [], stream_callback)
            log.info("call_llm: follow-up tokens prompt=%s completion=%s total=%s",
                     usage.get('prompt_tokens'), usage.get('completion_tokens'), usage.get('total_tokens'))
            pt = usage.get('prompt_tokens', 0)
            pct2 = f"{pt / profile['context']:.0%}" if profile['context'] else "?"
            if ctx_callback:
                ctx_callback(f"ctx {pct2} ({pt}/{profile['context']})")
            _emit_status("ready", status_callback, stream_callback)
            return content
        # agentic: loop to next turn

    log.warning("call_llm: reached max_turns=%d", max_turns)
    if agentic and not getattr(args, 'oneline', False):
        _emit_status(f"[!] reached max turns ({max_turns})", status_callback, stream_callback)
    return ""

def prompt_line_by_line(args, messages):
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
                print(f'\rProgress: [{bar}] {completed}/{total} ', end='', flush=True, file=sys.stderr)
                if completed == total:
                    print(file=sys.stderr)

    def process_line(line):
        local_messages = messages.copy()
        file_path = line_num = col_num = match_text = None

        if args.vimgrep:
            parts = line.split(':', 3)
            if len(parts) < 4:
                with lock:
                    print(f"[!] skipping malformed vimgrep line: {line}")
                return
            file_path, line_num, col_num, match_text = parts[0], parts[1], parts[2], parts[3]
            try:
                with open(file_path) as f:
                    numbered_lines = [f"{i + 1}: {l}" for i, l in enumerate(f.readlines())]
                messages.append({
                    "role": "user",
                    "content": f"<file_content>\n{''.join(numbered_lines)}</file_content>"
                })
            except (IOError, OSError) as e:
                log.error("could not read %s: %s", file_path, e)
                with lock:
                    print(f"[!] could not read {file_path}: {e}")
                return
            local_messages.append({
                "role": "user",
                "content": (
                    f"<match_location>\n"
                    f"  file: {file_path}\n"
                    f"  line: {line_num}\n"
                    f"  column: {col_num}\n"
                    f"  matched text: {match_text.strip()}\n"
                    f"</match_location>"
                )
            })
        else:
            local_messages.append({"role": "user", "content": line})

        local_messages.append({"role": "user", "content": args.prompt})

        response = call_llm(local_messages, args)

        with lock:
            completed_count[0] += 1
            update_progress(completed_count[0])
            if args.oneline:
                oneline_response = response.replace('\n', ' ')
                if args.vimgrep:
                    print(f"{file_path}:{line_num}:{col_num}:{oneline_response}", flush=True)
                else:
                    print(f"{line}:{oneline_response}", flush=True)
            else:
                count_str = f"{completed_count[0]}/{total}" if total is not None else str(completed_count[0])
                print(f"\n{'─' * 80}")
                if args.vimgrep:
                    print(f"[{count_str}] {file_path}:{line_num}:{col_num}  match: '{match_text.strip()}'")
                else:
                    print(f"[{count_str}] {line}")
                print('─' * 80)
                if response:
                    print(response)

    with ThreadPoolExecutor(max_workers=args.cores) as executor:
        if streaming_stdin:
            futures = []
            for raw_line in sys.stdin:
                line = raw_line.rstrip('\n')
                if line.strip():
                    futures.append(executor.submit(process_line, line))
        else:
            futures = [executor.submit(process_line, line) for line in lines]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log.error("thread error in prompt_line_by_line: %s", e)
                with lock:
                    print(f"[!] thread error: {e}")

TOOL_RESULT_MAX_CHARS = 8000

def trim_tool_result(result, max_chars=None):
    if max_chars is None:
        max_chars = config.get('tool_result_max_chars', TOOL_RESULT_MAX_CHARS)
    if max_chars and len(result) > max_chars:
        omitted = len(result) - max_chars
        log.info("trim_tool_result: trimmed %d chars (limit=%d)", omitted, max_chars)
        return result[:max_chars] + f"\n[truncated: {omitted} chars omitted]"
    return result


def _read_file_numbered(path):
    """Return file contents as a numbered-line string."""
    with open(path) as f:
        return "".join(f"{i + 1}: {line}" for i, line in enumerate(f))


ACTION_PROMPT = "prompt"
def action_prompt(args):
    if not args.prompt:
        print("this action require --prompt to be provided.")
        return

    log.info("action_prompt: model=%s file=%s location=%s line_by_line=%s vimgrep=%s oneline=%s",
             args.model, args.file, args.location, args.line_by_line, args.vimgrep, args.oneline)

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    elif getattr(args, 'agentic', False):
        profile = get_model_profile(args.model)
        prompt_text = AGENTIC_SYSTEM_PROMPTS.get(profile['tier'], AGENTIC_SYSTEM_PROMPTS['mid'])
        messages.append({"role": "system", "content": prompt_text})
        log.info("action_prompt: injected agentic system prompt (tier=%s)", profile['tier'])

    if not args.line_by_line:
        stdin = read_stdin_if_available()
        if stdin:
            log.info("action_prompt: including stdin (%d bytes)", len(stdin))
            messages.append({"role": "user", "content": stdin})

    if args.file and ((not args.line_by_line) or not sys.stdin.isatty()):
        log.info("action_prompt: including file %s", args.file)
        messages.append({"role": "user", "content": f"<file_content> {_read_file_numbered(args.file)} </file_content>"})

    if args.location:
        m = re.match(r"^(?P<file_path>.*):(?P<line_num>\d+):(?P<col_num>\d+)$", args.location)
        if m:
            fp, ln, cn = m.group('file_path'), m.group('line_num'), m.group('col_num')
            log.info("action_prompt: including location %s:%s:%s", fp, ln, cn)
            messages.append({"role": "user", "content": f"<file_content> {_read_file_numbered(fp)} </file_content>"})
            messages.append({"role": "user", "content": f"<cursor_location> line number: {ln}, column number: {cn} </cursor_location>"})

    if args.line_by_line:
        return prompt_line_by_line(args, messages)

    messages.append({"role": "user", "content": args.prompt})

    if not args.non_streaming and not args.oneline and not args.strict_format:
        log.info("action_prompt: calling LLM (streaming)")
        call_llm(messages, args, stream_callback=lambda chunk: (sys.stdout.write(chunk), sys.stdout.flush()))
        print()
    else:
        log.info("action_prompt: calling LLM (non-streaming)")
        content = call_llm(messages, args)
        if args.oneline:
            content = content.replace('\n', ' ')
        print(content)

def _handle_interactive_cmd(cmd, screen, messages, args, status_callback):
    """Execute a vim-style colon command from interactive mode."""
    if cmd == "compact":
        status_callback("compacting...")
        _compact_messages(messages, args.model)
        status_callback("ready")
    elif cmd == "tools":
        tool_names = [t.get('function', {}).get('name') for t in tools
                      if t.get('function', {}).get('name')]
        new_enabled = screen.prompt_tools_overlay(tool_names, args.internal_tools)
        args.internal_tools = new_enabled
        status_callback("ready")
    elif cmd == "":
        pass  # empty command, do nothing
    else:
        screen.write(f"\033[2;37m[unknown command: {cmd}]\033[m\n")


ACTION_INTERACTIVE = "interactive"
def action_interactive(args):
    """Multi-turn TUI conversation loop using Screen for display."""
    from cai.screen import Screen

    if not sys.stdout.isatty():
        print("[!] --interactive requires a TTY stdout.", file=sys.stderr)
        return

    # Pre-capture piped stdin BEFORE Screen (tty.setraw) takes over keyboard input
    stdin_content = read_stdin_if_available()

    # Build base messages — mirrors action_prompt's setup
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    else:
        profile = get_model_profile(args.model)
        prompt_text = AGENTIC_SYSTEM_PROMPTS.get(profile['tier'], AGENTIC_SYSTEM_PROMPTS['mid'])
        messages.append({"role": "system", "content": prompt_text})

    if stdin_content:
        log.info("action_interactive: including stdin (%d bytes)", len(stdin_content))
        messages.append({"role": "user", "content": stdin_content})

    if args.file:
        log.info("action_interactive: including file %s", args.file)
        messages.append({"role": "user",
                         "content": f"<file_content>{_read_file_numbered(args.file)}</file_content>"})

    if args.location:
        m = re.match(r"^(?P<file_path>.*):(?P<line_num>\d+):(?P<col_num>\d+)$", args.location)
        if m:
            fp, ln, cn = m.group('file_path'), m.group('line_num'), m.group('col_num')
            log.info("action_interactive: including location %s:%s:%s", fp, ln, cn)
            messages.append({"role": "user",
                             "content": f"<file_content>{_read_file_numbered(fp)}</file_content>"})
            messages.append({"role": "user",
                             "content": f"<cursor_location> line number: {ln}, column number: {cn} </cursor_location>"})

    screen = Screen()
    screen.set_cmd_completions(["compact", "tools"])

    last_ctx = [""]

    def _status(text):
        ctx_part = f"{last_ctx[0]} | " if last_ctx[0] else ""
        screen.set_status(f"{args.model} | {ctx_part}{text}")

    def status_callback(text):
        _status(text)

    def ctx_cb(ctx_str):
        last_ctx[0] = ctx_str
        screen._redraw_status_raw()
        sys.stdout.flush()

    _LLM_STYLE  = Screen._LLM_STYLE
    _META_STYLE = Screen._META_STYLE
    _RESET      = Screen._RESET

    def stream_cb(chunk):
        screen.write(chunk)

    def tool_cb(line):
        screen.write(f"{_META_STYLE}{line}{_RESET}")

    try:
        # If an initial prompt was given (-p or trailing words), run it first
        if args.prompt:
            messages.append({"role": "user", "content": args.prompt})
            screen.write(f"{Screen._USER_STYLE}> {args.prompt}{Screen._RESET}\n")
            screen.write("\n")
            status_callback("thinking...")
            screen.show_prompt_placeholder("> ")
            screen.write(_LLM_STYLE)
            response = call_llm(messages, args,
                                stream_callback=stream_cb,
                                status_callback=status_callback,
                                tool_callback=tool_cb,
                                ctx_callback=ctx_cb)
            screen.write(f"\n{_RESET}\n")
            if response:
                messages.append({"role": "assistant", "content": response})

        # Main interactive loop
        _status("ready")
        while True:
            user_input = screen.prompt("> ")
            if not user_input.strip():
                continue
            if user_input.startswith(":"):
                _handle_interactive_cmd(user_input[1:].strip(), screen, messages, args, status_callback)
                continue
            messages.append({"role": "user", "content": user_input})
            status_callback("thinking...")
            screen.show_prompt_placeholder("> ")
            screen.write(_LLM_STYLE)
            response = call_llm(messages, args,
                                stream_callback=stream_cb,
                                status_callback=status_callback,
                                tool_callback=tool_cb,
                                ctx_callback=ctx_cb)
            screen.write(f"\n{_RESET}\n")
            if response:
                messages.append({"role": "assistant", "content": response})

    except (KeyboardInterrupt, EOFError):
        screen.write("\n[exiting]\n")
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
    parser.add_argument("--cwd", default=".",
                        help="the current working for the script to operate at.")
    parser.add_argument("--index",
                        help="the index name for the index action to index into.")
    parser.add_argument("--file",
                        help="file path to include in the LLM context.")
    parser.add_argument("--location",
                        help="the location in the codebase to be used by the action. in the format of => <file_path>:<line_num>:<col_num>")
    parser.add_argument("--model", default=None,
                        help="the model to be used by the LLM")
    parser.add_argument("--progress", action="store_true",
                        help="show progess bar.")
    parser.add_argument("--oneline", action="store_true",
                        help="print results in a vimgrep style format, oneline all data.")
    parser.add_argument("--strict-format", default=None, choices=['json'],
                        help="the expected format provided from the LLM response.")
    parser.add_argument("--include-reasoning", action="store_true",
                        help="let the action know whether or not to include reasoning in the output.")
    parser.add_argument("--non-streaming", action="store_true",
                        help="let the action know whether or not to use the non-streaming api.")
    parser.add_argument("--agentic", action="store_true",
                        help="enable multi-turn agentic loop: LLM keeps calling tools until done.")
    parser.add_argument("--interactive", action="store_true",
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
    parser.add_argument('--vimgrep', action='store_true', default=False,
                        help="treat each input line as vimgrep format (file:line:col:text), load file context automatically. implies --line-by-line.")
    parser.add_argument('prompt_words', nargs='*',
                        help="prompt words after -- (alternative to -p)")

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
    args.internal_tools = set()
    for entry in args.tools:
        entry = entry.strip().rstrip(',')
        if not entry:
            continue
        if os.path.isfile(entry) or entry.endswith('.py'):
            log.info("main: loading external MCP %s", entry)
            external_mcps[entry] = get_external_tools(entry)
        else:
            log.info("main: enabling internal tool %s", entry)
            args.internal_tools.add(entry)

    if args.vimgrep:
        args.line_by_line = True

    if args.interactive:
        args.agentic = True
        if args.line_by_line or args.vimgrep or args.oneline:
            parser.error("--interactive is incompatible with --line-by-line / --vimgrep / --oneline")

    log.info("main: action=%s model=%s internal_tools=%s external_mcps=%s interactive=%s",
             args.action, args.model, sorted(args.internal_tools), list(external_mcps.keys()),
             args.interactive)

    if args.interactive:
        action_interactive(args)
        return

    if args.action == ACTION_PROMPT:
        action_prompt(args)


if __name__ == "__main__":
    main()
