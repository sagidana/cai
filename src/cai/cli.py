import argparse
import argcomplete
import json
import sys
import os
import re
import signal
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from cai import logger as _cai_logger
from cai.llm import (
    call_llm,
    LLMError,
    get_model_profile,
    MODEL_PROFILES,
    AGENTIC_SYSTEM_PROMPTS,
    _compact_messages,
)
import cai.llm as _llm


global config
global available_tools
global api_key
global openai_api
global openrouter_api

logging.basicConfig(
    filename="/tmp/cai.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("cai")

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_SKILLS_DIR  = os.path.join(os.path.dirname(__file__), "skills")
_USER_SKILLS_DIR = os.path.join(os.path.expanduser("~/.config/cai"), "skills")

# ─── Diagnostic output (stderr, TTY-only) ────────────────────────────────────
# Diagnostics are dim/faint so the final result is easy to spot.
# When stderr is not a TTY (e.g. piped) all diagnostic output is suppressed.
_STDERR_TTY = sys.stderr.isatty()
_DIM   = "\033[2m"   # faint — signals "not the result"
_RESET = "\033[0m"
# Start pessimistic: stdout may have written content without a trailing newline,
# so the first diagnostic needs to push itself onto a fresh line.
_at_bol = False  # is the terminal cursor currently at beginning-of-line?

def _diag(text, end='\n', ensure_newline=True):
    """Write a diagnostic line to stderr only when stderr is a TTY.

    Always starts on its own line: if the cursor is mid-line (e.g. after a
    streamed stdout chunk) a leading newline is emitted first.
    Pass ensure_newline=False to suppress the leading newline (e.g. for
    continuation chunks in a streaming sequence already on the same line).
    """
    global _at_bol
    if not _STDERR_TTY:
        return
    prefix = ('' if _at_bol else '\n') if ensure_newline else ''
    raw = f"{prefix}{_DIM}{text}{_RESET}{end}"
    # Normalise \n → \r\n so lines reset to column 1 in both cooked and raw
    # terminal modes.  The input listener puts the terminal in raw mode during
    # LLM streaming; without \r, bare \n is a pure line-feed that does not
    # return the cursor to column 1, causing each reasoning line to indent
    # further to the right.  An extra \r is harmless in cooked mode.
    raw = raw.replace('\r\n', '\n').replace('\n', '\r\n')
    sys.stderr.write(raw)
    sys.stderr.flush()
    _at_bol = end.endswith('\n') if end else False

def _note_output(text):
    """Update BOL state after writing *text* to any terminal stream."""
    global _at_bol
    if text:
        _at_bol = text.endswith('\n')

def _skills_completer(prefix, **kwargs):
    """Completer for --skill: returns skill names matching the prefix."""
    return [n for n in _list_skill_names() if n.startswith(prefix)]

def _tools_completer(prefix, **kwargs):
    """Completer for --tools: file paths for external MCPs, tool names for internal."""
    import glob as _glob
    import re as _re

    # If it looks like a path, complete as file
    if prefix.startswith('/') or prefix.startswith('./') or prefix.startswith('../') or os.sep in prefix:
        matches = _glob.glob(prefix + '*')
        return matches

    # Otherwise complete internal tool names from all *tools.py files
    tools_dir = os.path.dirname(__file__)
    names = []
    try:
        for tools_file in _glob.glob(os.path.join(tools_dir, '*tools.py')):
            with open(tools_file) as f:
                content = f.read()
            names.extend(_re.findall(r'@mcp\.tool\(\)\s+def\s+(\w+)', content))
    except Exception:
        pass
    return [n for n in names if n.startswith(prefix)]

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

    log.info("init: starting")

    import cai.api as _cai_api
    import cai.tools as _cai_tools
    OpenAiApi = _cai_api.OpenAiApi
    OpenRouterApi = _cai_api.OpenRouterApi
    _cai_tools.register_server(_cai_tools.INTERNAL_SERVER)

    config_dir = os.path.expanduser("~/.config/cai")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    config_path = os.path.join(config_dir, "config.json")
    if not os.path.exists(config_path):
        default_config = {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "arcee-ai/trinity-mini:free",
            "observation_mask_pct": 0.60,
            "observation_mask_keep": 3,
            "context_budget_pct": 0.75,
            "tool_result_max_chars": 40000,
            "ssl_verify": True,
            "stuck_detection": False,
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
    if 'prompt_mode' not in config:
        config['prompt_mode'] = 'local'
        updated = True
    if 'stuck_detection' not in config:
        config['stuck_detection'] = False
        updated = True
    if updated:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    available_tools = _cai_tools.get_all_tools()
    api_key = open(api_key_path).read().strip()
    openai_api = OpenAiApi(config.get('base_url'), api_key, ssl_verify=config.get('ssl_verify', True))
    openrouter_api = OpenRouterApi(api_key)
    log.info("init: done (base_url=%s, available_tools=%d)", config.get('base_url'), len(available_tools))

    _llm.setup(config, openai_api, _cai_tools.call_tool, diag_fn=_diag)

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

def prompt_line_by_line(args, messages, available_tools):
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

        response = call_llm(local_messages,
                            args,
                            available_tools,
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

def _read_file_numbered(path):
    """Return file contents as a numbered-line string."""
    with open(path) as f:
        return "".join(f"{i + 1}: {line}" for i, line in enumerate(f))

def _list_skill_names():
    """Return sorted list of available skill names (built-in + user)."""
    names = set()
    for d in (_SKILLS_DIR, _USER_SKILLS_DIR):
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith('.md'):
                    names.add(f[:-3])
    return sorted(names)

def _parse_skill_file(text):
    """Parse a skill markdown file into (tools: set[str], prompt: str).

    Format:
        tools: tool_a, tool_b
        ---
        ## Skill: Name
        ...prompt body...

    Everything before the first '---' line is metadata (key: value).
    Everything after is the prompt body. If no '---' is present the whole
    file is treated as the prompt body with no tool declarations.
    """
    tools: set[str] = set()
    if '\n---\n' in text:
        header, _, body = text.partition('\n---\n')
        for line in header.splitlines():
            if ':' not in line:
                continue
            key, _, val = line.partition(':')
            if key.strip().lower() == 'tools':
                tools = {t.strip() for t in val.split(',') if t.strip()}
        return tools, body.strip()
    return tools, text.strip()

def _load_skills(names):
    """Load and parse skill files by name.

    Search order per name: user skills dir first, then built-in package dir.
    Returns (all_tools: set[str], prompts: list[str]).
    Unknown skill names are logged as warnings and skipped.
    """
    all_tools: set[str] = set()
    prompts: list[str] = []
    for name in names:
        candidates = [
            os.path.join(_USER_SKILLS_DIR, f"{name}.md"),
            os.path.join(_SKILLS_DIR, f"{name}.md"),
        ]
        for path in candidates:
            try:
                with open(path) as f:
                    tools, prompt = _parse_skill_file(f.read())
                all_tools |= tools
                if prompt:
                    prompts.append(prompt)
                log.info("_load_skills: loaded %r from %s (tools=%s)", name, path, tools)
                break
            except OSError:
                continue
        else:
            log.warning("_load_skills: skill %r not found in %s or %s",
                        name, _USER_SKILLS_DIR, _SKILLS_DIR)
    return all_tools, prompts

_MODE_BLOCKS = {
    'research': (
        "## Current Task Mode: Research\n"
        "Your primary focus is investigation and analysis. Prioritize forming "
        "hypotheses, gathering evidence, and stating confidence levels. "
        "Treat development steps as secondary unless explicitly needed.\n"
    ),
    'dev': (
        "## Current Task Mode: Development\n"
        "Your primary focus is implementation and planning. Prioritize reading "
        "existing code, making targeted edits, and producing a clear plan before "
        "building. Treat research steps as secondary unless explicitly needed.\n"
    ),
}

def _load_cai_prompt():
    """Load the base cai system prompt from disk based on config prompt_mode.

    Reads either prompts/local.md or prompts/sota.md from the package directory.
    Falls back to the mid-tier agentic prompt if the file cannot be read.
    Mode and skill blocks are assembled by the caller so they appear after the
    base prompt (recency bias — more specific instructions carry more weight).
    """
    prompt_mode = config.get('prompt_mode', 'local')
    if prompt_mode not in ('local', 'sota'):
        log.warning("_load_cai_prompt: unknown prompt_mode %r, falling back to 'local'", prompt_mode)
        prompt_mode = 'local'
    prompt_path = os.path.join(_PROMPTS_DIR, f"{prompt_mode}.md")
    try:
        with open(prompt_path) as f:
            return f.read()
    except OSError as e:
        log.error("_load_cai_prompt: cannot read %s: %s", prompt_path, e)
        return AGENTIC_SYSTEM_PROMPTS.get('mid')

def _assemble_system_prompt(task_mode, skill_prompts):
    """Assemble the full system prompt in specificity order.

    Order: base → mode block → skill prompts
    Most specific instructions last so they take precedence via recency bias.
    """
    parts = [_load_cai_prompt()]
    if task_mode and task_mode in _MODE_BLOCKS:
        parts.append(_MODE_BLOCKS[task_mode])
    parts.extend(skill_prompts)
    return "\n\n".join(parts)

def _build_base_messages(args, stdin_content=None):
    """Build the initial messages list (system prompt, stdin, file, cursor)."""
    messages = []

    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    else:
        skill_names = getattr(args, 'skill', []) or []
        skill_tools, skill_prompts = _load_skills(skill_names)
        if skill_tools:
            args.selected_tools |= skill_tools

        task_mode = getattr(args, 'mode', 'research')
        messages.append({"role": "system", "content": _assemble_system_prompt(task_mode, skill_prompts)})
        log.info("_build_base_messages: system prompt assembled "
                 "(prompt_mode=%s, task_mode=%s, skills=%s)",
                 config.get('prompt_mode', 'local'), task_mode, skill_names)

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
def action_prompt(args, available_tools):
    if not args.prompt:
        print("this action require --prompt to be provided.")
        return

    log.info("action_prompt: model=%s file=%s cursor=%s line_by_line=%s oneline=%s",
             args.model, args.file, args.cursor, args.line_by_line, args.oneline)

    messages = _build_base_messages(args)

    if args.line_by_line:
        return prompt_line_by_line(args, messages, available_tools)

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

    _tools_str = ", ".join(sorted(getattr(args, 'selected_tools', set()) or [])) or "none"
    _flags = []
    if getattr(args, 'force_tools', False):
        _flags.append("force_tools")
    if getattr(args, 'strict_format', None):
        _flags.append(f"strict_format={args.strict_format}")
    if getattr(args, 'non_streaming', False):
        _flags.append("non_streaming")
    if getattr(args, 'oneline', False):
        _flags.append("oneline")
    _cai_logger.log(1, (
        f"PROMPT  model={args.model}  max_turns={getattr(args, 'max_turns', None)}  "
        f"tools=[{_tools_str}]  flags={_flags}\n{args.prompt}"
    ))

    if not args.non_streaming and not args.oneline and not args.strict_format:
        log.info("action_prompt: calling LLM (streaming)")
        try:
            def _stream_to_stdout(chunk):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                _note_output(chunk)

            _cai_logger.push_nest(1)
            call_llm(messages,
                     args,
                     available_tools,
                     stream_callback=_stream_to_stdout,
                     tool_callback=_stderr_tool,
                     status_callback=_stderr_status,
                     ctx_callback=_stderr_ctx)
            _cai_logger.pop_nest(1)
            print()
        except LLMError as e:
            _cai_logger.pop_nest(1)
            print()
            _diag(f"[error] {e}")
    else:
        log.info("action_prompt: calling LLM (non-streaming)")
        try:
            _cai_logger.push_nest(1)
            content = call_llm(messages,
                               args,
                               available_tools,
                               tool_callback=_stderr_tool,
                               status_callback=_stderr_status,
                               ctx_callback=_stderr_ctx)
            _cai_logger.pop_nest(1)
            if args.oneline:
                content = content.replace('\n', ' ')
            print(content)
        except LLMError as e:
            _cai_logger.pop_nest(1)
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
        import cai.tools as _cai_tools
        tool_entries = _cai_tools.get_tool_entries()
        new_selected = screen.prompt_tools_overlay(tool_entries, args.selected_tools)
        args.selected_tools = new_selected
        args._manual_selected_tools = set(new_selected)
        status_callback("ready")
    elif cmd == "context":
        # Extract the last known prompt_tokens and context_size from the
        # status-bar string (e.g. "ctx 45% (57600/128000)") so the overlay
        # can show live usage and scale the estimate as messages are edited.
        _ctx_m = re.search(r'\((\d+)/(\d+)\)', last_ctx[0])
        _prompt_tok  = int(_ctx_m.group(1)) if _ctx_m else 0
        _ctx_size    = int(_ctx_m.group(2)) if _ctx_m else 0
        _, _new_tok  = screen.prompt_context_overlay(
            messages, context_size=_ctx_size, prompt_tokens=_prompt_tok
        )
        if _new_tok and _ctx_size:
            _pct        = f"{_new_tok / _ctx_size:.0%}"
            last_ctx[0] = f"ctx {_pct} ({_new_tok}/{_ctx_size})"
        status_callback("ready")
    elif cmd == "" or cmd == "skill":
        # /skill with no args → show active skills
        active = getattr(args, 'skill', []) or []
        available = _list_skill_names()
        screen.write(f"\033[2;37m[active skills: {', '.join(active) or 'none'} | "
                     f"available: {', '.join(available)}]\033[m\n")
    elif cmd.startswith("skill "):
        skill_args = cmd[len("skill "):].split()
        if skill_args == ["off"]:
            args.skill = []
        else:
            args.skill = skill_args
        # Reset to manual /tools selection if the user made one, otherwise
        # fall back to the CLI snapshot — then layer skill tools on top.
        manual = getattr(args, '_manual_selected_tools', None)
        args.selected_tools = set(manual) if manual is not None else set(getattr(args, '_base_selected_tools', set()))
        skill_tools, skill_prompts = _load_skills(args.skill)
        args.selected_tools |= skill_tools
        # Rebuild system message in-place.
        task_mode = getattr(args, 'mode', 'research')
        new_system = _assemble_system_prompt(task_mode, skill_prompts)
        if messages and messages[0].get('role') == 'system':
            messages[0]['content'] = new_system
        else:
            messages.insert(0, {"role": "system", "content": new_system})
        active_str = ', '.join(args.skill) if args.skill else 'none'
        screen.write(f"\033[2;37m[skills set to: {active_str}]\033[m\n")
        status_callback("ready")
    elif cmd.startswith("save"):
        path = cmd[len("save"):].strip()
        if not path:
            screen.write(f"\033[2;37m[usage: /save <path>]\033[m\n")
        else:
            import json as _json
            payload = {
                "version": 1,
                "messages": messages,
                "settings": {
                    "selected_tools": sorted(getattr(args, 'selected_tools', set()) or []),
                    "skills": list(getattr(args, 'skill', []) or []),
                },
            }
            try:
                with open(path, 'w') as _f:
                    _json.dump(payload, _f, indent=2)
                screen.write(f"\033[2;37m[saved to {path}]\033[m\n")
            except OSError as _e:
                screen.write(f"\033[1;31m[save error] {_e}\033[m\n")
    elif cmd == "model":
        live_models = None
        if _llm.openai_api is not None:
            try:
                live_models = _llm.openai_api.get_models()
            except Exception:
                pass
        if not live_models:
            screen.write(f"\033[2;37m[current model: {args.model} | no models available]\033[m\n")
        else:
            picked = screen.prompt_model_overlay(live_models)
            if picked:
                args.model = picked
                screen.write(f"\033[2;37m[model set to: {args.model}]\033[m\n")
                status_callback("ready")
            else:
                screen.write(f"\033[2;37m[current model: {args.model}]\033[m\n")
    elif cmd.startswith("load"):
        path = cmd[len("load"):].strip()
        if not path:
            screen.write(f"\033[2;37m[usage: /load <path>]\033[m\n")
        else:
            import json as _json
            try:
                with open(path) as _f:
                    payload = _json.load(_f)
                # Restore messages
                messages[:] = payload.get("messages", [])
                settings = payload.get("settings", {})
                # Restore tools
                loaded_tools = set(settings.get("selected_tools", []))
                args.selected_tools = loaded_tools
                args._manual_selected_tools = set(loaded_tools)
                # Restore skills and rebuild system prompt
                loaded_skills = settings.get("skills", [])
                args.skill = loaded_skills
                skill_tools, skill_prompts = _load_skills(args.skill)
                args.selected_tools |= skill_tools
                task_mode = getattr(args, 'mode', 'research')
                new_system = _assemble_system_prompt(task_mode, skill_prompts)
                if messages and messages[0].get('role') == 'system':
                    messages[0]['content'] = new_system
                else:
                    messages.insert(0, {"role": "system", "content": new_system})
                screen.write(f"\033[2;37m[loaded from {path} — {len(messages)} messages, "
                             f"{len(args.selected_tools)} tools, "
                             f"skills: {', '.join(args.skill) or 'none'}]\033[m\n")
                _chars = sum(len(str(m.get('content', ''))) for m in messages)
                _rough_tok = _chars // 4
                _profile = get_model_profile(args.model)
                _ctx_limit = _profile.get('context', 0)
                if _ctx_limit:
                    last_ctx[0] = f"ctx {_rough_tok / _ctx_limit:.0%} (~{_rough_tok}/{_ctx_limit})"
                else:
                    last_ctx[0] = ""
                status_callback("ready")
            except (OSError, ValueError, KeyError) as _e:
                screen.write(f"\033[1;31m[load error] {_e}\033[m\n")
    else:
        screen.write(f"\033[2;37m[unknown command: {cmd}]\033[m\n")

ACTION_INTERACTIVE = "interactive"
def action_interactive(args, available_tools):
    """Multi-turn TUI conversation loop using Screen for display."""
    from cai.screen import Screen

    if not sys.stdout.isatty():
        _diag("[!] --interactive requires a TTY stdout.")
        return

    # Pre-capture piped stdin BEFORE Screen (tty.setraw) takes over keyboard input
    stdin_content = read_stdin_if_available()

    messages = _build_base_messages(args, stdin_content=stdin_content)

    screen = Screen()
    _skill_cmds = [f"skill {n}" for n in _list_skill_names()] + ["skill", "skill off"]
    _live_models = []
    if _llm.openai_api is not None:
        try:
            _live_models = _llm.openai_api.get_models() or []
        except Exception:
            pass
    screen.set_cmd_completions(["compact", "tools", "clear", "context", "save", "load", "model"] + _skill_cmds)

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

    _reasoning_active = [False]

    def reasoning_cb(chunk):
        if chunk is None:
            # End of reasoning — restore LLM style
            if _reasoning_active[0]:
                screen.write(f"\n{_RESET}{_LLM_STYLE}")
                _reasoning_active[0] = False
        else:
            if not _reasoning_active[0]:
                # First reasoning chunk — switch to meta (gray) style
                screen.write(f"\n{_META_STYLE}")
                _reasoning_active[0] = True
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
                if user_input.strip():
                    screen._history.insert(0, user_input)
                screen.write(f"{Screen._USER_STYLE}> {user_input}{Screen._RESET}\n\n")
            else:
                user_input = screen.prompt("> ")
            if not user_input.strip():
                continue
            if user_input.startswith("/"):
                _handle_interactive_cmd(user_input[1:].strip(), screen, messages, args, status_callback, last_ctx)
                continue
            messages.append({"role": "user", "content": user_input})
            status_callback("thinking...")
            screen.show_prompt_placeholder("> ")
            screen.write(_LLM_STYLE)
            screen.start_input_listener()
            _tools_str = ", ".join(sorted(getattr(args, 'selected_tools', set()) or [])) or "none"
            _cai_logger.log(1, (
                f"USER  model={args.model}  max_turns={getattr(args, 'max_turns', None)}  "
                f"tools=[{_tools_str}]\n{user_input}"
            ))
            try:
                _cai_logger.push_nest(1)
                response = call_llm(messages, args, available_tools,
                                    stream_callback=stream_cb,
                                    status_callback=status_callback,
                                    tool_callback=tool_cb,
                                    ctx_callback=ctx_cb,
                                    interrupt_event=screen._interrupt_event,
                                    reasoning_callback=reasoning_cb)
                _cai_logger.pop_nest(1)
                if screen._interrupt_event.is_set():
                    screen.write(f"\n{_RESET}{_META_STYLE}[interrupted]{_RESET}\n\n")
                    status_callback("interrupted")
                    continue
                screen.write(f"\n{_RESET}\n")
                messages.append({"role": "assistant", "content": response})
            except LLMError as e:
                _cai_logger.pop_nest(1)
                screen.write(f"\n{_RESET}{_ERROR_STYLE}[error] {e}{_RESET}\n\n")
                status_callback("error")
            except BaseException:
                _cai_logger.pop_nest(1)
                screen.write(_RESET)
                raise
            finally:
                screen.stop_input_listener()

    except (KeyboardInterrupt, EOFError):
        screen.write(f"{_RESET}\n[exiting]\n")
    finally:
        screen.close()


def main():
    global config
    global available_tools

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
    parser.add_argument("--system-prompt-file",
                        help="path to a file whose contents are used as the system prompt.")
    parser.add_argument("--mode", choices=list(_MODE_BLOCKS.keys()), default='research',
                        help="task-focus hint prepended to the system prompt (default: research).")
    skill_arg = parser.add_argument("--skill", nargs='+', default=[], metavar='SKILL',
                        help=f"activate one or more skills (available: {', '.join(_list_skill_names())}).")
    skill_arg.completer = _skills_completer
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
                        help="max tool-call turns in agentic mode (default: unlimited).")
    tools_arg = parser.add_argument('-t',
                        '--tools',
                        nargs='+',
                        default=[],
                        help="internal tool names to enable (e.g. generic_linux_command). use --mcp for external MCP servers.")
    tools_arg.completer = _tools_completer
    parser.add_argument('--mcp',
                        nargs='+',
                        default=[],
                        metavar='CMD',
                        help="shell commands to launch external MCP servers (e.g. 'python server.py' or 'npx @modelcontextprotocol/server-filesystem /').")
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

    _cai_logger.init()
    init()
    setup_shell_completion()

    args = parser.parse_args()
    if args.prompt_words:
        if args.prompt:
            parser.error("cannot use both -p/--prompt and trailing words after --")
        args.prompt = " ".join(args.prompt_words)
    if args.system_prompt_file:
        if args.system_prompt:
            parser.error("--system-prompt and --system-prompt-file are mutually exclusive.")
        try:
            with open(args.system_prompt_file) as f:
                args.system_prompt = f.read()
        except OSError as e:
            parser.error(f"cannot read --system-prompt-file: {e}")
    if args.model is None:
        args.model = config.get('model', "arcee-ai/trinity-mini:free")

    import cai.tools as _cai_tools
    _tools_before_mcp = {t['function']['name'] for t in available_tools}
    for cmd_str in args.mcp:
        cmd_str = cmd_str.strip().rstrip(',')
        if not cmd_str:
            continue
        log.info("main: registering MCP server %r", cmd_str)
        _cai_tools.register_server(cmd_str)
    if args.mcp:
        # Refresh available_tools now that additional servers are registered.
        available_tools = _cai_tools.get_all_tools()
        _mcp_tools = {t['function']['name'] for t in available_tools} - _tools_before_mcp

    args.selected_tools = set()
    for entry in args.tools:
        entry = entry.strip().rstrip(',')
        if not entry:
            continue
        log.info("main: enabling tool %s", entry)
        args.selected_tools.add(entry)
    # Auto-select tools from user-provided --mcp servers.
    if args.mcp:
        args.selected_tools |= _mcp_tools
        log.info("main: auto-selected %d MCP tools: %s", len(_mcp_tools), sorted(_mcp_tools))
    # Snapshot tools before skill injection so /skill can reset cleanly.
    args._base_selected_tools = frozenset(args.selected_tools)

    if args.interactive:
        if args.line_by_line or args.oneline:
            parser.error("--interactive is incompatible with --line-by-line / --oneline")

    log.info("main: action=%s model=%s selected_tools=%s interactive=%s",
             args.action, args.model, sorted(args.selected_tools), args.interactive)

    if args.logger:
        from cai.logger import launch_tui
        launch_tui()
        return

    if args.harness:
        from cai.harness import execute_harness, parse_harness_file
        instructions, label_map = parse_harness_file(args.harness)
        user_prompt = args.prompt or ""
        execute_harness(instructions, label_map, user_prompt, args, available_tools,
                        harness_path=args.harness)
        return

    if args.interactive:
        action_interactive(args, available_tools)
        return

    if args.action == ACTION_PROMPT:
        action_prompt(args, available_tools)


if __name__ == "__main__":
    main()
