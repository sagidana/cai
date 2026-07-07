"""cli: the command-line entry point.

A one-shot driver: build a standalone Run from the flags + config and stream its
answer. Tools/skills come from --tool/--skill; the LLM knobs (--model,
--system-prompt, --reasoning-effort, --temperature, --max-steps, --non-streaming)
are forwarded to the run. The base system prompt is composed by appending, in
order, whichever of ~/.config/cai/SYSTEM.md, ./SYSTEM.md and --system-prompt
exist. base_url/model/api_key come from cai.config. Tab
completion lists every available tool/skill via the registries themselves.

With no prompt to act on (no -p/'--', no --file, no piped stdin) and a terminal
attached, cai drops into the interactive full-screen TUI (cai.tui); -i forces it.

Two stream modes ride the same flags: --tail follows a live served agent's
conversation read-only over its unix socket (cai.tail), and --watch runs the
prompt as a one-shot agent each time piped stdin settles (cai.watch).

The prompt is given via -p/--prompt or after a '--' separator (so --skill/--tool
can each take several values without swallowing it): `cai --skill fs -- fix x`.

stdin/stdout behaviour:
- piped stdin (cai is in a pipeline) is read and prepended as context.
- when stdout is a TTY the answer streams there live; when stdout is piped, dim
  progress goes to stderr and the clean answer is printed to stdout once at the
  end - so `cai ... | tool` gets exactly the result."""
import os
import argparse
import sys

# optional: real tab completion when argcomplete is installed (pip install
# argcomplete). absent, the completer hooks below are simply never consulted.
try:
    import argcomplete
except ImportError:
    argcomplete = None


_STDOUT_TTY = sys.stdout.isatty()
_STDERR_TTY = sys.stderr.isatty()
_DIM = "\033[2m"
_RESET = "\033[0m"


def _diag(text):
    """progress/diagnostic line to stderr, dimmed, only when stderr is a TTY."""
    if not _STDERR_TTY:
        return
    sys.stderr.write(_DIM + text + _RESET + "\n")
    sys.stderr.flush()


def _stream_out(chunk):
    """write streamed model output: to stdout when a human is watching it there,
    else dimmed to stderr (the clean result is printed to stdout once at the
    end). when stdout is piped and stderr is not a TTY, progress is dropped."""
    if _STDOUT_TTY:
        sys.stdout.write(chunk)
        sys.stdout.flush()
    elif _STDERR_TTY:
        sys.stderr.write(_DIM + chunk + _RESET)
        sys.stderr.flush()


def _skill_completer(prefix, **kwargs):
    # the on-disk view without importing any extension Python - a completion
    # request should stay cheap.
    from cai.environment import Environment, list_extensions

    matching = []
    for name in Environment(list_extensions()).available_skills():
        if not name.startswith(prefix): continue
        matching.append(name)
    return matching


def _tool_completer(prefix, **kwargs):
    from cai.environment import Environment, list_extensions

    matching = []
    for name in Environment(list_extensions()).available_tools():
        if not name.startswith(prefix): continue
        matching.append(name)
    return matching


def _agent_completer(prefix, **kwargs):
    # the live served agents, straight off their sockets - a completion
    # request pays no config or LLM imports.
    from cai.tail import live_names

    matching = []
    for name in live_names():
        if not name.startswith(prefix): continue
        matching.append(name)
    return matching


def _path_completer(prefix, **kwargs):
    """filesystem-path completer for `cai extend <source>`."""
    directory = os.path.dirname(prefix) or "."
    matching = []
    try:
        entries = os.listdir(directory)
    except OSError:
        return matching
    for name in entries:
        full = os.path.join(directory, name)
        if not full.startswith(prefix): continue
        if os.path.isdir(full):
            full = full + os.sep
        matching.append(full)
    return matching


def _add_extend_subparser(sub):
    """`cai extend` - install / list / remove extension bundles. a first-class
    subcommand so it shows in `cai --help` and tab-completes; its handler lives
    in cai.extend, dispatched from main() before the heavy LLM imports."""
    from cai.extend import _extension_completer

    extend_parser = sub.add_parser(
        "extend",
        help="install, list, or remove extension bundles, then exit.",
        description="manage cai extension bundles under ~/.config/cai/extensions/. "
                    "install a folder, .zip file, or http(s) URL; or --list / "
                    "--remove the installed ones.")
    source_arg = extend_parser.add_argument(
        "source",
        nargs="?",
        metavar="PATH_OR_URL",
        help="folder, .zip file, or http(s) URL of the bundle to install.")
    source_arg.completer = _path_completer
    extend_parser.add_argument("--replace",
                               action="store_true",
                               help="overwrite the extension if it is already installed.")
    extend_parser.add_argument("--list",
                               action="store_true",
                               help="list the installed extensions and exit.")
    remove_arg = extend_parser.add_argument("--remove",
                                            default=None,
                                            metavar="NAME",
                                            help="uninstall the named extension and exit.")
    remove_arg.completer = _extension_completer


def _add_python_subparser(sub):
    """`cai python` - manage the managed virtualenv the python tool runs
    snippets in. only `install` for now: the sandbox has no network, so
    packages must be installed from out here."""
    python_parser = sub.add_parser(
        "python",
        help="manage the python tool's virtualenv, then exit.",
        description="manage the cai-managed virtualenv the python tool runs "
                    "snippets in (~/.config/cai/venv/). the sandbox has no "
                    "network, so packages are installed from here, outside it.")
    python_sub = python_parser.add_subparsers(dest="python_command", required=True)
    install_parser = python_sub.add_parser(
        "install",
        help="pip-install packages into the managed virtualenv.",
        description="pip-install packages into the managed virtualenv "
                    "(created first if needed).")
    install_parser.add_argument("packages",
                                nargs="+",
                                metavar="PACKAGE",
                                help="pip requirement specifiers "
                                     "(e.g. requests, 'numpy>=2').")
    uninstall_parser = python_sub.add_parser(
        "uninstall",
        help="pip-uninstall packages from the managed virtualenv.",
        description="pip-uninstall packages from the managed virtualenv "
                    "(no confirmation prompt).")
    uninstall_parser.add_argument("packages",
                                  nargs="+",
                                  metavar="PACKAGE",
                                  help="package names to remove.")
    python_sub.add_parser(
        "list-packages",
        help="list the packages installed in the managed virtualenv.",
        description="list the packages installed in the managed virtualenv "
                    "(created first if needed).")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cai",
        description="Send a prompt to an LLM and stream the answer.")
    parser.add_argument("-p", "--prompt",
                        default=None,
                        help="the prompt to send (or pass it after '--')")
    parser.add_argument("-i", "--interactive",
                        action="store_true",
                        help="launch the full-screen interactive TUI")
    parser.add_argument("-c", "--continue",
                        dest="continue_session",
                        action="store_true",
                        help="resume the most recent saved session (implies -i)")
    parser.add_argument("--sessions",
                        action="store_true",
                        help="pick a saved session to resume (implies -i)")
    parser.add_argument("--system-prompt",
                        default=None,
                        help="system prompt text (appended after "
                             "~/.config/cai/SYSTEM.md and ./SYSTEM.md, "
                             "when those exist)")
    skill_arg = parser.add_argument("--skill",
                                    nargs="+",
                                    default=[],
                                    metavar="SKILL",
                                    help="skills to activate (one or more)")
    skill_arg.completer = _skill_completer
    tool_arg = parser.add_argument("-t", "--tool",
                                   nargs="+",
                                   default=[],
                                   metavar="TOOL",
                                   help="tools to enable ('<server>__<tool>', one or more)")
    tool_arg.completer = _tool_completer
    parser.add_argument("--file",
                        default=None,
                        metavar="PATH",
                        help="include a file's contents as context")
    parser.add_argument("--cwd",
                        default=None,
                        metavar="DIR",
                        help="run from this directory, so --file, tools and the LLM "
                             "see paths relative to it")
    parser.add_argument("--model",
                        default=None,
                        help="model id (default: the `model` field in config.json)")
    parser.add_argument("--non-streaming",
                        action="store_true",
                        help="wait for the full response instead of streaming")
    parser.add_argument("--reasoning-effort",
                        default=None,
                        metavar="LEVEL",
                        help="reasoning effort (e.g. low/medium/high), provider-dependent")
    parser.add_argument("--temperature",
                        type=float,
                        default=None,
                        help="sampling temperature")
    parser.add_argument("--max-steps",
                        type=int,
                        default=None,
                        help="max agentic turns before giving up")
    parser.add_argument("--strict-format",
                        default=None,
                        metavar="FORMAT",
                        help="constrain the answer's shape: 'json', 'regex:<pat>' or "
                             "'regex-each-line:<pat>' (retries until it matches)")
    tail_arg = parser.add_argument("--tail",
                                   nargs="?",
                                   const="",
                                   default=None,
                                   metavar="AGENT",
                                   help="follow a live served agent's conversation "
                                        "read-only (bare --tail picks one with fzf), "
                                        "then exit")
    tail_arg.completer = _agent_completer
    parser.add_argument("--watch",
                        action="store_true",
                        help="watch piped stdin: each time the stream settles, run "
                             "the prompt as a one-shot agent over its tail; new "
                             "data kills an in-flight run")
    parser.add_argument("--watch-threshold",
                        type=float,
                        default=2.0,
                        metavar="SECONDS",
                        help="quiet time on stdin that counts as settled "
                             "(default 2)")
    parser.add_argument("--watch-window",
                        type=int,
                        default=65536,
                        metavar="BYTES",
                        help="sliding window: a triggered run sees the last BYTES "
                             "of the stream (default 64KiB)")

    # subcommands. the prompt is pulled out before argparse (see _split_dashdash)
    # so the top level keeps no free positional that would clash with these.
    sub = parser.add_subparsers(dest="command")
    _add_extend_subparser(sub)
    _add_python_subparser(sub)
    return parser


def _split_dashdash(argv):
    """split argv at the first '--': tokens after it are the prompt (joined),
    tokens before it go to argparse. the prompt lives after '--' (not as a bare
    positional) so --skill/--tool can take several values without swallowing it.
    returns (argv_before, prompt_after_or_None)."""
    if "--" not in argv:
        return argv, None
    index = argv.index("--")
    tail = argv[index + 1:]
    prompt = None
    if tail:
        prompt = " ".join(tail)
    return argv[:index], prompt


def _resolve_prompt(args, dashdash_prompt, parser):
    if args.prompt is not None and dashdash_prompt is not None:
        parser.error("provide the prompt via -p/--prompt OR after '--', not both")
    if args.prompt is not None:
        return args.prompt
    return dashdash_prompt


def _resolve_system_prompt(args, parser):
    """compose the base system prompt from, in order: ~/.config/cai/SYSTEM.md,
    ./SYSTEM.md, --system-prompt. each part that exists is appended; None when
    none do. a CLI-only convenience - SDK callers pass Agent(system_prompt=...)
    themselves and never read these files."""
    from cai import config
    parts = []
    for path in (os.path.join(config.config_dir(), "SYSTEM.md"), "SYSTEM.md"):
        if not os.path.isfile(path): continue
        try:
            with open(path) as f:
                content = f.read().strip()
        except OSError as e:
            parser.error(f"cannot read {path}: {e}")
        if content:
            parts.append(content)
    if args.system_prompt is not None:
        parts.append(args.system_prompt)
    if not parts:
        return None
    return "\n\n".join(parts)


def _build_messages(args, prompt, parser):
    """assemble the run's starting conversation: piped stdin, then the --file
    contents, then the prompt - each a user turn."""
    messages = []

    if not sys.stdin.isatty():
        stdin_content = sys.stdin.read()
        if stdin_content:
            messages.append({"role": "user", "content": stdin_content})

    if args.file:
        try:
            with open(args.file) as f:
                content = f.read()
        except OSError as e:
            parser.error(f"cannot read --file: {e}")
        block = f"<file_content path={args.file!r}>\n{content}\n</file_content>"
        messages.append({"role": "user", "content": block})

    if prompt:
        messages.append({"role": "user", "content": prompt})

    return messages


def _short_args(tool_args):
    if not tool_args:
        return ""
    parts = []
    for key in tool_args:
        text = str(tool_args[key])
        if len(text) > 40:
            text = text[:40] + "..."
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def _diag_tool_call(tool_name, tool_args):
    """the '-> tool(...)' diagnostic line; a python call is always a script,
    so its code argument prints as a syntax-colored block under the line
    instead of a truncated blob."""
    from cai.screen.render import python_code_arg, render_python_code

    code = python_code_arg(tool_name, tool_args)
    if code is None:
        _diag(f"  -> {tool_name}({_short_args(tool_args)})")
        return
    rest = dict(tool_args)
    del rest["code"]
    _diag(f"  -> {tool_name}({_short_args(rest)})")
    if not _STDERR_TTY:
        return
    sys.stderr.write(render_python_code(code))
    sys.stderr.flush()


def _drive(run, show_reasoning=True):
    """consume the run, routing output, and print the final answer. returns the
    process exit code. a run that fails for good (ApiError after the api layer's
    retries, or an LLMError like max_steps / strict-format exhaustion) prints
    one error line to stderr and exits 1 - never a silent empty answer."""
    from cai.api import ApiError
    from cai.events import EventType
    from cai.llm import LLMError

    try:
        for event in run:
            if event.type == EventType.CONTENT:
                _stream_out(event.text or "")
            elif event.type == EventType.REASONING:
                if show_reasoning:
                    _stream_out(event.text or "")
            elif event.type == EventType.TOOL_CALL:
                _diag_tool_call(event.tool_name, event.tool_args)
            elif event.type == EventType.TOOL_RESULT:
                _diag(f"  <- {event.tool_name}: {len(event.tool_result or '')} chars")
    except KeyboardInterrupt:
        run.interrupt.set()
        _diag("[interrupted]")
        return 130
    except (ApiError, LLMError) as e:
        print(f"[!] {e}", file=sys.stderr)
        return 1

    content = run.text
    if run.stream and _STDOUT_TTY:
        # the live stream already wrote the answer to stdout; just end the line.
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        if run.stream and _STDERR_TTY:
            sys.stderr.write("\n")
            sys.stderr.flush()
        print(content)
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv, dashdash_prompt = _split_dashdash(argv)

    parser = build_parser()
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)

    # subcommands run before the heavy LLM bootstrap (they need neither config
    # nor an API key) and after argcomplete so a completion request never runs
    # them.
    if args.command == "extend":
        from cai import extend
        return extend.run(args, parser)
    if args.command == "python":
        from cai import pytool
        if args.python_command == "install":
            return pytool.install(args.packages)
        if args.python_command == "uninstall":
            return pytool.uninstall(args.packages)
        return pytool.list_packages()
    if args.tail is not None:
        from cai import tail
        return tail.run(args.tail)

    # --watch preconditions, checked before any bootstrap so a bad invocation
    # fails fast: it is a headless stdin-driven mode, and the prompt IS the
    # task each settle triggers.
    if args.watch:
        if args.interactive or args.continue_session or args.sessions:
            parser.error("--watch cannot combine with -i/--continue/--sessions")
        if args.prompt is None and dashdash_prompt is None:
            parser.error("--watch needs a prompt (-p/--prompt or after '--'): "
                         "it is the task each settle triggers")
        if sys.stdin.isatty():
            parser.error("--watch reads piped stdin, but stdin is a terminal")

    # --cwd: move the whole process before any config/file/tool work, so --file,
    # the fs tool sandbox (which resolves against os.getcwd()) and every other
    # tool see paths relative to the requested directory.
    if args.cwd is not None:
        try:
            os.chdir(args.cwd)
        except OSError as e:
            print(f"cannot change to --cwd {args.cwd!r}: {e}", file=sys.stderr)
            return 1

    # heavy imports happen only here - after argcomplete has short-circuited, so
    # tab completion never pays for them.
    import threading

    from cai import config
    from cai.environment import Environment
    from cai.agent import Run
    from cai.api import OpenAiApi
    from cai.ui import TerminalUI

    try:
        cfg = config.load_config()
        api_key = config.load_api_key()
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    prompt = _resolve_prompt(args, dashdash_prompt, parser)
    system_prompt = _resolve_system_prompt(args, parser)

    # resume flags. --continue resolves the most recent saved session up front;
    # --sessions defers the choice to a picker shown once the TUI is up. both
    # imply interactive and resume the chosen session in place (autosave writes
    # back to it). they are mutually exclusive.
    if args.continue_session and args.sessions:
        parser.error("--continue and --sessions are mutually exclusive")
    resume_path = None
    pick_session = False
    if args.continue_session:
        from cai.session import SessionsRegistry
        saved = SessionsRegistry.list_sessions()
        if saved:
            resume_path = saved[0]
        else:
            _diag("[no saved sessions to continue — starting fresh]")
        args.interactive = True
    elif args.sessions:
        pick_session = True
        args.interactive = True

    # interactive TUI: explicit (-i / a resume flag), or the default when cai is
    # run on a terminal with no prompt to act on (no -p/'--', no --file, no
    # piped stdin).
    interactive = args.interactive
    if not interactive and prompt is None and not args.file and sys.stdin.isatty():
        interactive = True
    if interactive:
        from cai import tui
        return tui.run(model=args.model,
                       system_prompt=system_prompt,
                       tools=args.tool,
                       skills=args.skill,
                       reasoning_effort=args.reasoning_effort,
                       temperature=args.temperature,
                       max_steps=args.max_steps,
                       resume_path=resume_path,
                       pick_session=pick_session,
                       initial_prompt=prompt)

    model = args.model or cfg.model
    env = Environment.default().load()
    # the cai.settings skills / tools are auto-activated on every CLI run, merged
    # in on top of any --skill / --tool the user passed.
    tools = Environment.merge_activations(args.tool, env.settings.tools)
    skills = Environment.merge_activations(args.skill, env.settings.skills)
    api = OpenAiApi(cfg.base_url,
                    api_key,
                    ssl_verify=config.load_optional("ssl_verify", True))

    def _driver(run):
        # the settings flag the TUI honors gates the headless stream too.
        return _drive(run, show_reasoning=env.settings.show_reasoning)

    def _spawn(messages):
        return Run(messages,
                   model,
                   api,
                   env=env,
                   system_prompt=system_prompt,
                   tools=tools,
                   skills=skills,
                   ui=TerminalUI(),
                   interrupt=threading.Event(),
                   reasoning_effort=args.reasoning_effort,
                   temperature=args.temperature,
                   max_steps=args.max_steps,
                   tool_result_max_chars=env.settings.tool_result_max_chars,
                   stream=not args.non_streaming,
                   strict_format=args.strict_format)

    if args.watch:
        from cai import watch

        # the static turns every trigger shares - the --file block (read once,
        # up front, so a bad path fails now and not mid-watch) and the prompt.
        # per trigger the settled window text is prepended, mirroring the
        # stdin/file/prompt order of a normal piped run.
        base_messages = []
        if args.file:
            try:
                with open(args.file) as f:
                    file_content = f.read()
            except OSError as e:
                parser.error(f"cannot read --file: {e}")
            block = f"<file_content path={args.file!r}>\n{file_content}\n</file_content>"
            base_messages.append({"role": "user", "content": block})
        base_messages.append({"role": "user", "content": prompt})

        def _make_run(text):
            messages = []
            messages.append({"role": "user", "content": text})
            for message in base_messages:
                messages.append(message)
            return _spawn(messages)

        return watch.run(_make_run,
                         _driver,
                         threshold=args.watch_threshold,
                         window=args.watch_window)

    messages = _build_messages(args, prompt, parser)
    if not messages:
        parser.error("no prompt: pass -p/--prompt, a prompt after '--', --file, or piped stdin")

    return _driver(_spawn(messages))
