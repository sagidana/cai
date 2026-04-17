import argparse
import argcomplete
import json
import queue as _queue
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
from cai.tools import select_tools
from cai import core


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
    return [n for n in core.list_skill_names() if n.startswith(prefix)]

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

    ctx = core.bootstrap(diag_fn=_diag)
    config = ctx.config
    api_key = ctx.api_key
    openai_api = ctx.openai_api
    openrouter_api = ctx.openrouter_api
    available_tools = ctx.available_tools

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

        tools = select_tools(available_tools, getattr(args, 'selected_tools', set()))
        response = call_llm(local_messages,
                            args.model,
                            tools,
                            strict_format=args.strict_format,
                            force_tools=args.force_tools,
                            max_turns=args.max_turns,
                            reasoning_effort=args.reasoning_effort,
                            temperature=args.temperature,
                            oneline=args.oneline,
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

def _build_base_messages(args, stdin_content=None):
    """Build the initial messages list (system prompt, stdin, file, cursor)."""
    messages = []

    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    elif getattr(args, 'naked', False):
        log.info("_build_base_messages: --naked mode, skipping default system prompt")
    else:
        skill_names = getattr(args, 'skill', []) or []
        skill_tools, skill_prompts = core.load_skills(skill_names)
        if skill_tools:
            args.selected_tools |= skill_tools

        task_mode = getattr(args, 'mode', 'research')
        messages.append({"role": "system", "content": core.assemble_system_prompt(config, task_mode, skill_prompts)})
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


def _load_context(path, args, merge_tools=False):
    """Load a .flow context file and apply settings to args.

    Returns the messages list from the flow.

    When *merge_tools* is False (default, used by /load), the flow's tools
    replace args.selected_tools.  When True (used by --context), CLI --tools
    already in args.selected_tools are kept and the flow's tools are added.

    Skills from the flow are always loaded and their tools/prompts injected.
    The system prompt in messages is rebuilt to reflect the loaded skills.
    """
    import json as _json
    with open(path) as f:
        payload = _json.load(f)

    messages = payload.get("messages", [])
    settings = payload.get("settings", {})

    flow_tools = set(settings.get("selected_tools", []))
    if merge_tools:
        args.selected_tools = flow_tools | args.selected_tools
    else:
        args.selected_tools = flow_tools
        args._manual_selected_tools = set(flow_tools)

    # Load skills from the flow and rebuild system prompt.
    flow_skills = settings.get("skills", [])
    existing_skills = set(getattr(args, 'skill', []) or [])
    args.skill = list(existing_skills | set(flow_skills))
    skill_tools, skill_prompts = core.load_skills(args.skill)
    args.selected_tools |= skill_tools

    task_mode = getattr(args, 'mode', 'research')
    new_system = core.assemble_system_prompt(config, task_mode, skill_prompts)
    if messages and messages[0].get('role') == 'system':
        messages[0]['content'] = new_system
    else:
        messages.insert(0, {"role": "system", "content": new_system})

    log.info("_load_context: loaded %s — %d messages, tools=%s, skills=%s",
             path, len(messages), sorted(args.selected_tools), args.skill)
    return messages


ACTION_PROMPT = "prompt"
def action_prompt(args, available_tools):
    if not args.prompt:
        print("this action require --prompt to be provided.")
        return

    log.info("action_prompt: model=%s file=%s cursor=%s line_by_line=%s oneline=%s",
             args.model, args.file, args.cursor, args.line_by_line, args.oneline)

    if getattr(args, 'context', None):
        messages = _load_context(args.context, args, merge_tools=True)
    else:
        messages = _build_base_messages(args)

    if args.line_by_line:
        return prompt_line_by_line(args, messages, available_tools)

    messages.append({"role": "user", "content": args.prompt})

    def _stderr_dim(chunk):
        if _STDERR_TTY:
            sys.stderr.write(_DIM + chunk + _RESET)
            sys.stderr.flush()
            _note_output(chunk)

    def _stderr_tool(chunk, error=False):
        _stderr_dim(chunk)

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

    streaming = not args.non_streaming and not args.oneline and not args.strict_format
    log.info("action_prompt: calling LLM (%s)", "streaming" if streaming else "non-streaming")

    tools = select_tools(available_tools, getattr(args, 'selected_tools', set()))

    openai_api.error_cb = lambda msg: _diag(f"[error] {msg}")
    try:
        _cai_logger.push_nest(1)
        content = call_llm(messages,
                           args.model,
                           tools,
                           strict_format=args.strict_format,
                           force_tools=args.force_tools,
                           max_turns=args.max_turns,
                           reasoning_effort=args.reasoning_effort,
                           temperature=args.temperature,
                           oneline=args.oneline,
                           stream_callback=_stderr_dim if streaming else None,
                           tool_callback=_stderr_tool)
        _cai_logger.pop_nest(1)
        if streaming and _STDERR_TTY:
            sys.stderr.write('\n')
            sys.stderr.flush()
        if args.oneline:
            content = (content or '').replace('\n', ' ')
        print(content)
    except KeyboardInterrupt:
        _cai_logger.pop_nest(1)
        _diag("[interrupted]")
        sys.exit(130)
    except LLMError as e:
        _cai_logger.pop_nest(1)
        _diag(f"[error] {e}")
    finally:
        openai_api.error_cb = None

def _handle_interactive_cmd(cmd, screen, messages, args, status_callback, last_ctx):
    """Execute a vim-style colon command from interactive mode."""
    if cmd == "compact":
        status_callback("compacting...")
        try:
            _compact_messages(messages, args.model)
        except LLMError as e:
            screen.write(f"[compact error] {e}\n", kind=screen.ERROR)
        last_ctx[0] = ""
        status_callback("ready")
    elif cmd == "clear":
        # Keep only the system prompt (first message if role==system)
        if messages and messages[0].get('role') == 'system':
            messages[1:] = []
        else:
            messages.clear()
        screen.clear_buffer()
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
        available = core.list_skill_names()
        screen.write(f"[active skills: {', '.join(active) or 'none'} | "
                     f"available: {', '.join(available)}]\n", kind=screen.META)
    elif cmd.startswith("skill "):
        skill_args = cmd[len("skill "):].split()
        if skill_args == ["off"]:
            args.skill = []
        else:
            # Append new skills, preserving existing ones (no duplicates).
            existing = getattr(args, 'skill', []) or []
            for s in skill_args:
                if s not in existing:
                    existing.append(s)
            args.skill = existing
        # Reset to manual /tools selection if the user made one, otherwise
        # fall back to the CLI snapshot — then layer skill tools on top.
        manual = getattr(args, '_manual_selected_tools', None)
        args.selected_tools = set(manual) if manual is not None else set(getattr(args, '_base_selected_tools', set()))
        skill_tools, skill_prompts = core.load_skills(args.skill)
        args.selected_tools |= skill_tools
        # Rebuild system message in-place.
        task_mode = getattr(args, 'mode', 'research')
        new_system = core.assemble_system_prompt(config, task_mode, skill_prompts)
        if messages and messages[0].get('role') == 'system':
            messages[0]['content'] = new_system
        else:
            messages.insert(0, {"role": "system", "content": new_system})
        active_str = ', '.join(args.skill) if args.skill else 'none'
        screen.write(f"[active skills: {active_str}]\n", kind=screen.META)
        status_callback("ready")
    elif cmd.startswith("save"):
        path = cmd[len("save"):].strip()
        if not path:
            screen.write("[usage: /save <path>]\n", kind=screen.META)
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
                screen.write(f"[saved to {path}]\n", kind=screen.META)
            except OSError as _e:
                screen.write(f"[save error] {_e}\n", kind=screen.ERROR)
    elif cmd == "model":
        live_models = None
        if _llm.openai_api is not None:
            try:
                live_models = _llm.openai_api.get_models()
            except Exception:
                pass
        if not live_models:
            screen.write(f"[current model: {args.model} | no models available]\n", kind=screen.META)
        else:
            picked = screen.prompt_model_overlay(live_models)
            if picked:
                args.model = picked
                screen.write(f"[model set to: {args.model}]\n", kind=screen.META)
                status_callback("ready")
            else:
                screen.write(f"[current model: {args.model}]\n", kind=screen.META)
    elif cmd.startswith("load"):
        path = cmd[len("load"):].strip()
        if not path:
            screen.write("[usage: /load <path>]\n", kind=screen.META)
        else:
            try:
                loaded = _load_context(path, args)
                messages[:] = loaded
                screen.write(f"[loaded from {path} — {len(messages)} messages, "
                             f"{len(args.selected_tools)} tools, "
                             f"skills: {', '.join(args.skill) or 'none'}]\n", kind=screen.META)
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
                screen.write(f"[load error] {_e}\n", kind=screen.ERROR)
    else:
        screen.write(f"[unknown command: {cmd}]\n", kind=screen.META)

ACTION_INTERACTIVE = "interactive"
def action_interactive(args, available_tools):
    """Multi-turn TUI conversation loop using Screen for display."""
    from cai.screen import Screen

    if not sys.stdout.isatty():
        _diag("[!] --interactive requires a TTY stdout.")
        return

    # Pre-capture piped stdin BEFORE Screen (tty.setraw) takes over keyboard input
    stdin_content = read_stdin_if_available()

    if getattr(args, 'context', None):
        messages = _load_context(args.context, args, merge_tools=True)
    else:
        messages = _build_base_messages(args, stdin_content=stdin_content)

    screen = Screen()
    _skill_cmds = [f"skill {n}" for n in core.list_skill_names()] + ["skill", "skill off"]
    _live_models = []
    if _llm.openai_api is not None:
        try:
            _live_models = _llm.openai_api.get_models() or []
        except Exception:
            pass
    _skill_names = core.list_skill_names()
    screen.set_cmd_completions({
        "compact": [],
        "clear": [],
        "tools": [],
        "context": [],
        "model": [],
        "save": [],
        "load": [],
        "skill": ["off"] + _skill_names,
    })

    last_ctx = [""]
    llm_state = {"phase": "ready"}
    message_queue: _queue.Queue = _queue.Queue()

    def _drain_queue():
        while True:
            try:
                message_queue.get_nowait()
                message_queue.task_done()
            except _queue.Empty:
                return

    def _refresh_status_line():
        parts = [args.model]
        if last_ctx[0]:
            parts.append(last_ctx[0])
        phase = llm_state.get("phase") or ""
        if phase:
            parts.append(phase)
        qsize = message_queue.qsize()
        if qsize:
            parts.append(f"queued:{qsize}")
        screen.set_status(" | ".join(parts))

    def status_callback(text=None):
        if text is not None:
            # llm.py emits "[turn N] foo(args), bar(args)" when tool calls
            # are dispatched. The detail is noisy on the status line —
            # collapse it to a short label. Details still show up in the
            # content buffer via tool_callback.
            if text.startswith("[turn "):
                text = "calling tools"
            llm_state["phase"] = text
        _refresh_status_line()

    def ctx_cb(ctx_str):
        last_ctx[0] = ctx_str
        _refresh_status_line()

    def _mark_thinking():
        # llm.py emits "calling tools" when dispatching tool calls but
        # never re-announces that it's back to generating a response,
        # so "calling tools" would otherwise stick on the status line
        # for the rest of the turn. As soon as we see content or
        # reasoning chunks streaming back, flip the phase.
        if llm_state["phase"] != "thinking...":
            llm_state["phase"] = "thinking..."
            _refresh_status_line()

    def stream_cb(chunk):
        _mark_thinking()
        screen.write(chunk, kind=Screen.LLM)

    _reasoning_buf = []

    def reasoning_cb(chunk):
        if chunk is None:
            return  # next write's kind handles the transition
        _mark_thinking()
        _reasoning_buf.append(chunk)
        screen.write(chunk, kind=Screen.REASONING)

    def tool_cb(line, error=False):
        screen.write(line, kind=Screen.ERROR if error else Screen.TOOL)

    def api_error_cb(msg):
        screen.write(f"\n{msg}\n", kind=Screen.ERROR)

    openai_api.error_cb = api_error_cb

    def _on_interrupt() -> bool:
        """Ctrl-C from the prompt — interrupt the in-flight LLM call and
        drop everything still queued. Returning True tells the TUI the
        event was consumed (skip its press-again-to-quit logic)."""
        if screen._busy or message_queue.qsize() > 0:
            _drain_queue()
            screen._interrupt_event.set()
            llm_state["phase"] = "interrupting..."
            _refresh_status_line()
            return True
        return False

    screen.set_interrupt_handler(_on_interrupt)

    def llm_worker():
        while True:
            user_input = message_queue.get()
            if user_input is None:
                message_queue.task_done()
                return
            if screen._interrupt_event.is_set():
                # Queue was drained by an interrupt; skip stragglers.
                message_queue.task_done()
                if message_queue.qsize() == 0:
                    screen.set_busy(False)
                    _refresh_status_line()
                continue
            screen._interrupt_event.clear()
            llm_state["phase"] = "thinking..."
            _refresh_status_line()
            # Echo the prompt into the content buffer now — at the moment
            # the LLM actually starts working on it — rather than at submit
            # time. For queued prompts this keeps the conversation log in
            # the same order the LLM sees them.
            screen.write(f"> {user_input}\n\n", kind=Screen.USER)
            _tools_str = ", ".join(sorted(getattr(args, 'selected_tools', set()) or [])) or "none"
            _cai_logger.log(1, (
                f"USER  model={args.model}  max_turns={getattr(args, 'max_turns', None)}  "
                f"tools=[{_tools_str}]\n{user_input}"
            ))
            try:
                messages.append({"role": "user", "content": user_input})
                _cai_logger.push_nest(1)
                _reasoning_buf.clear()
                try:
                    tools = select_tools(available_tools, getattr(args, 'selected_tools', set()))
                    response = call_llm(messages, args.model, tools,
                                        strict_format=args.strict_format,
                                        force_tools=args.force_tools,
                                        max_turns=args.max_turns,
                                        reasoning_effort=args.reasoning_effort,
                                        temperature=args.temperature,
                                        oneline=args.oneline,
                                        stream_callback=stream_cb,
                                        status_callback=status_callback,
                                        tool_callback=tool_cb,
                                        ctx_callback=ctx_cb,
                                        interrupt_event=screen._interrupt_event,
                                        reasoning_callback=reasoning_cb)
                finally:
                    _cai_logger.pop_nest(1)
                if screen._interrupt_event.is_set():
                    screen.write("\n[interrupted]\n\n", kind=Screen.META)
                    llm_state["phase"] = "interrupted"
                    _drain_queue()
                    screen._interrupt_event.clear()
                else:
                    screen.write("\n", kind=Screen.DEFAULT)
                    assistant_msg = {"role": "assistant", "content": response}
                    if _reasoning_buf:
                        assistant_msg['_reasoning'] = ''.join(_reasoning_buf)
                    messages.append(assistant_msg)
            except LLMError as e:
                screen.write(f"\n[error] {e}\n\n", kind=Screen.ERROR)
                llm_state["phase"] = "error"
            except Exception as e:
                log.exception("llm_worker: unhandled error")
                screen.write(f"\n[error] {e}\n\n", kind=Screen.ERROR)
                llm_state["phase"] = "error"
            finally:
                message_queue.task_done()
                if message_queue.qsize() == 0:
                    screen.set_busy(False)
                    if llm_state["phase"] not in ("interrupted", "error"):
                        llm_state["phase"] = "ready"
                _refresh_status_line()

    worker = threading.Thread(target=llm_worker, daemon=True, name='cai-llm-worker')
    worker.start()

    def _queue_prompt(user_input: str) -> None:
        if not user_input.strip():
            return
        # Set busy before enqueueing so Ctrl-C in the handoff window between
        # the worker dequeuing and entering call_llm still routes through
        # _on_interrupt instead of the press-again-to-quit path.
        screen.set_busy(True)
        message_queue.put(user_input)
        _refresh_status_line()

    try:
        _refresh_status_line()

        # Seed with initial --prompt if provided. The worker echoes it
        # into the content buffer when it actually starts processing.
        if args.prompt:
            seed = args.prompt
            if seed.strip():
                screen._history.insert(0, seed)
                _queue_prompt(seed)

        while True:
            user_input = screen.prompt("> ")
            if screen._command_result is not None:
                _handle_interactive_cmd(screen._command_result, screen, messages, args, status_callback, last_ctx)
                screen._command_result = None
                _refresh_status_line()
                continue
            # prompt() has already echoed the submission into the content
            # buffer; just queue it for the worker and keep looping so the
            # user can type the next prompt immediately.
            _queue_prompt(user_input)

    except (KeyboardInterrupt, EOFError):
        screen.write("\n[exiting]\n", kind=Screen.META)
    finally:
        screen._interrupt_event.set()
        _drain_queue()
        message_queue.put(None)
        openai_api.error_cb = None
        screen.close()
        worker.join(timeout=1.0)


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
    parser.add_argument("--mode", choices=list(core.MODE_BLOCKS.keys()), default='research',
                        help="task-focus hint prepended to the system prompt (default: research).")
    skill_arg = parser.add_argument("--skill", nargs='+', default=[], metavar='SKILL',
                        help=f"activate one or more skills (available: {', '.join(core.list_skill_names())}).")
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
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["high", "medium", "low"],
                        help="enable extended thinking via OpenRouter reasoning.effort")
    parser.add_argument("--temperature", type=float, default=None,
                        help="sampling temperature (0.0-2.0). not supported by all models.")
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
    parser.add_argument('--context', default=None, metavar='PATH',
                        help="path to a .flow file (from /save) to resume from. "
                             "Tools from --tools are appended; --model overrides.")
    parser.add_argument('--harness', default=None,
                        help="path to a .harness.cai orchestration file.")
    parser.add_argument('--naked', action='store_true',
                        help="do not prepend any default system prompts. overridden by --system-prompt / --system-prompt-file.")
    parser.add_argument('--logger', action='store_true',
                        help="launch the interactive hierarchical log viewer.")
    parser.add_argument('--log-path', default=_cai_logger.LOG_PATH, metavar='PATH',
                        help=f"path to the JSONL log file — used both for writing "
                             f"and for --logger viewing (default: {_cai_logger.LOG_PATH}).")

    # Must be called before init() so tab completion exits immediately without
    # running any heavy initialization (API clients, tree-sitter, etc.).
    argcomplete.autocomplete(parser)

    # Parse before init() so --log-path routes both the writer and viewer.
    _early_args, _ = parser.parse_known_args()
    _cai_logger.init(_early_args.log_path)
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
        launch_tui(args.log_path)
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
