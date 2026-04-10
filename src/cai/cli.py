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
    call_llm, LLMError, MaxTurnsReached,
    trim_tool_result, enforce_strict_format,
    get_model_profile, MODEL_PROFILES, AGENTIC_SYSTEM_PROMPTS,
    _compact_messages,
)
import cai.llm as _llm


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
            "observation_mask_pct": 0.60,
            "observation_mask_keep": 3,
            "context_budget_pct": 0.75,
            "tool_result_max_chars": 40000,
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

    _llm.setup(config, openai_api, call_tool, call_external_tool, diag_fn=_diag)

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
                     external_mcps,
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
                               external_mcps,
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
                response = call_llm(messages, args, available_tools, external_mcps,
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
    parser.add_argument("--system-prompt-file",
                        help="path to a file whose contents are used as the system prompt.")
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
        execute_harness(instructions, label_map, user_prompt, args, available_tools,
                        harness_path=args.harness)
        return

    if args.interactive:
        action_interactive(args, available_tools, external_mcps)
        return

    if args.action == ACTION_PROMPT:
        action_prompt(args, available_tools, external_mcps)


if __name__ == "__main__":
    main()
