"""cli: the command-line entry point.

A one-shot driver: build a standalone Run from the flags + config and stream its
answer. Tools/skills come from --tool/--skill; the LLM knobs (--model,
--system-prompt, --reasoning-effort, --temperature, --max-steps, --non-streaming)
are forwarded to the run. base_url/model/api_key come from cai.config. Tab
completion lists every available tool/skill via the registries themselves.
Interactive (TUI) mode is a later layer.

The prompt is given via -p/--prompt or after a '--' separator (so --skill/--tool
can each take several values without swallowing it): `cai --skill fs -- fix x`.

stdin/stdout behaviour:
- piped stdin (cai is in a pipeline) is read and prepended as context.
- when stdout is a TTY the answer streams there live; when stdout is piped, dim
  progress goes to stderr and the clean answer is printed to stdout once at the
  end - so `cai ... | tool` gets exactly the result."""
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
    from cai.skills import SkillsRegistry

    matching = []
    for name in SkillsRegistry.available_skills():
        if not name.startswith(prefix): continue
        matching.append(name)
    return matching


def _tool_completer(prefix, **kwargs):
    from cai.tools import ToolRegistry

    matching = []
    for name in ToolRegistry.available_tools():
        if not name.startswith(prefix): continue
        matching.append(name)
    return matching


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cai",
        description="Send a prompt to an LLM and stream the answer.")
    parser.add_argument("-p", "--prompt",
                        default=None,
                        help="the prompt to send (or pass it after '--')")
    parser.add_argument("--system-prompt",
                        default=None,
                        help="system prompt text")
    parser.add_argument("--system-prompt-file",
                        default=None,
                        metavar="PATH",
                        help="read the system prompt from a file")
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
    if args.system_prompt and args.system_prompt_file:
        parser.error("--system-prompt and --system-prompt-file are mutually exclusive")
    if args.system_prompt_file:
        try:
            with open(args.system_prompt_file) as f:
                return f.read()
        except OSError as e:
            parser.error(f"cannot read --system-prompt-file: {e}")
    return args.system_prompt


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


def _drive(run):
    """consume the run, routing output, and print the final answer. returns the
    process exit code."""
    from cai.events import EventType

    try:
        for event in run:
            if event.type == EventType.CONTENT:
                _stream_out(event.text or "")
            elif event.type == EventType.REASONING:
                _stream_out(event.text or "")
            elif event.type == EventType.TOOL_CALL:
                _diag(f"  -> {event.tool_name}({_short_args(event.tool_args)})")
            elif event.type == EventType.TOOL_RESULT:
                _diag(f"  <- {event.tool_name}: {len(event.tool_result or '')} chars")
    except KeyboardInterrupt:
        run.interrupt.set()
        _diag("[interrupted]")
        return 130

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

    # heavy imports happen only here - after argcomplete has short-circuited, so
    # tab completion never pays for them.
    import threading

    from cai import config
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
    messages = _build_messages(args, prompt, parser)
    if not messages:
        parser.error("no prompt: pass -p/--prompt, a prompt after '--', --file, or piped stdin")

    model = args.model or cfg.model

    run = Run(messages,
              model,
              OpenAiApi(cfg.base_url, api_key),
              system_prompt=system_prompt,
              tools=args.tool,
              skills=args.skill,
              ui=TerminalUI(),
              interrupt=threading.Event(),
              reasoning_effort=args.reasoning_effort,
              temperature=args.temperature,
              max_steps=args.max_steps,
              stream=not args.non_streaming)

    return _drive(run)
