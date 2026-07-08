"""tail: follow a live served agent's conversation from the terminal.

`cai --tail <name>` attaches a read-only wire to the agent's unix socket
(~/.config/cai/agents/<name>.sock) and prints the conversation as it happens:
the stored messages first (the backlog), then the run's EVENT broadcast
streamed live. nothing is ever sent on the wire - no SUBMIT/STEER/INTERRUPT,
and PROMPT broadcasts are ignored (the owner answers them) - so the tailed
agent cannot be driven or disturbed from here; a UnixWiredAgent broadcasts to
every attached wire, and this one may come and go freely.

bare `--tail` offers the live agents in an fzf picker (the fzf binary must be
on PATH). the flag's tab completion lists the same live agents; a stale
socket file left by a crash is skipped by both - and reaped by the registry's
connect() the moment the probe finds it refusing connections.

kept import-light on purpose: the CLI dispatches --tail before the heavy LLM
bootstrap, so tailing (and completing) needs neither config nor an API key."""
from __future__ import annotations

import subprocess
import sys

from cai.agents_registry import AgentsRegistry
from cai.events import EventType
from cai.wire import Wire


_DIM = "\033[2m"
_RESET = "\033[0m"


def live_names():
    """the served agents whose socket accepts a connection - the registry's
    names minus stale files left by crashed agents."""
    names = []
    for name in AgentsRegistry.list_names():
        try:
            probe = AgentsRegistry.connect(name)
        except OSError:
            continue
        probe.close()
        names.append(name)
    return names


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


def _stored_call_args(function):
    """a stored tool call's arguments dict (they cross the API as JSON text)."""
    import json

    try:
        args = json.loads(function.get("arguments") or "{}")
    except ValueError:
        return {}
    if not isinstance(args, dict):
        return {}
    return args


class _Printer:
    """paints the conversation to stdout in the headless CLI's idiom: user
    turns as '> ...', streamed content raw, tool traffic and notes dimmed (on
    a TTY). tracks whether the cursor sits mid-stream so a line never glues
    onto a half-written content chunk. a stream chunk's trailing newlines are
    held back and only flushed when more text arrives - the model closes every
    reasoning/content block with '\\n\\n' before a tool call, and written
    eagerly those paint a stray blank line above each '->' line."""

    def __init__(self, out=None, show_reasoning=True):
        if out is None:
            out = sys.stdout
        self.out = out
        self.tty = out.isatty()
        self.show_reasoning = show_reasoning
        self.midline = False
        self.pending = 0        # held-back trailing newlines from stream()
        self.dirty = False      # anything painted since the last separator

    def stream(self, text, dim=False):
        if not text: return
        body = text.rstrip("\n")
        if body:
            if self.pending:
                self.out.write("\n" * self.pending)
                self.pending = 0
            if dim and self.tty:
                body = _DIM + body + _RESET
            self.out.write(body)
            self.out.flush()
            self.midline = True
            self.dirty = True
        self.pending += len(text) - len(body)

    def line(self, text, dim=True):
        # held newlines collapse into this line's own break: a block's
        # trailing '\n\n' must not blank-line the '->' that follows it.
        self.pending = 0
        if self.midline:
            self.out.write("\n")
            self.midline = False
        if dim and self.tty:
            text = _DIM + text + _RESET
        self.out.write(text + "\n")
        self.out.flush()
        self.dirty = True

    def separator(self):
        """the blank line between turns - skipped for a turn that painted
        nothing, so an invisible turn stays silent."""
        if not self.dirty: return
        self.line("", dim=False)
        self.dirty = False

    def tool_call(self, name, args):
        # a python call is always a script: its code argument prints as a
        # syntax-colored block under the line instead of a truncated blob.
        from cai.screen.render import python_code_arg, render_python_code

        code = python_code_arg(name, args)
        if code is None:
            self.line(f"  -> {name}({_short_args(args)})")
            return
        rest = dict(args)
        del rest["code"]
        self.line(f"  -> {name}({_short_args(rest)})")
        if not self.tty:
            return
        self.out.write(render_python_code(code))
        self.out.flush()

    def event(self, event):
        if event.type == EventType.USER:
            self.line(f"> {event.text or ''}", dim=False)
            return
        if event.type == EventType.CONTENT:
            self.stream(event.text or "")
            return
        if event.type == EventType.REASONING:
            if not self.show_reasoning: return
            self.stream(event.text or "", dim=True)
            return
        if event.type == EventType.TOOL_CALL:
            self.tool_call(event.tool_name, event.tool_args or {})
            return
        if event.type == EventType.TOOL_RESULT:
            self.line(f"  <- {event.tool_name}: {len(event.tool_result or '')} chars")
            return


def _replay(printer, messages):
    """print the stored conversation the way the TUI's replay paints it: user
    turns, assistant content, tool calls as '->' lines, each tool reply as
    '<- name: N chars' (the name mapped back through the call id). system
    turns and stored reasoning are skipped - a tail wants the visible
    conversation, not the scaffolding."""
    call_names = {}
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        if role == "user":
            printer.line(f"> {content}", dim=False)
            continue
        if role == "tool":
            name = call_names.get(message.get("tool_call_id"), "?")
            printer.line(f"  <- {name}: {len(content)} chars")
            continue
        if role != "assistant":
            continue
        if content.strip():
            printer.stream(content + "\n")
        for call in (message.get("tool_calls") or []):
            function = call.get("function") or {}
            name = function.get("name") or "?"
            call_names[call.get("id")] = name
            printer.tool_call(name, _stored_call_args(function))


def _show_reasoning():
    """the user's show_reasoning setting, the same one the TUI honors. loading
    the environment runs extension/init.py registrations - what every CLI run
    does anyway; still no config.json or API key needed. a broken init.py must
    not take a passive tail down with it, so any load failure means the
    default."""
    from cai.environment import Environment

    try:
        return Environment.default().load().settings.show_reasoning
    except Exception:
        return True


def _control(name, op):
    """run one read-only control op over a short-lived connection; None when
    the socket is gone or the op fails (the :agents view reads the same way)."""
    try:
        conn = AgentsRegistry.connect(name)
    except OSError:
        return None
    wire = Wire(conn)
    try:
        ok, value, _error = wire.control(op)
    except OSError:
        ok, value = False, None
    finally:
        conn.close()
    if not ok:
        return None
    return value


def _pick(names):
    """offer `names` in fzf and return the pick - None on cancel, or when the
    fzf binary is missing (the names are printed so the user can retry with
    `--tail <name>`)."""
    try:
        proc = subprocess.run(["fzf", "--prompt=agent> "],
                              input="\n".join(names) + "\n",
                              stdout=subprocess.PIPE,
                              text=True)
    except FileNotFoundError:
        print("fzf not found - live agents:", file=sys.stderr)
        for name in names:
            print(f"  {name}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        return None
    pick = proc.stdout.strip()
    if not pick:
        return None
    return pick


def _print_msg(printer, msg):
    kind = msg.get("type")
    if kind == Wire.EVENT:
        printer.event(Wire.event_from_dict(msg["event"]))
        return
    if kind == Wire.RESULT:
        # end of turn: a blank line keeps turns visually separate.
        printer.separator()
        return
    if kind == Wire.PROMPT:
        # the owner answers prompts; a tail just shows them. status updates
        # are transient noise, so only real prompts and notifies print.
        if msg.get("kind") == "status": return
        printer.line(f"[{msg.get('kind')}] {msg.get('message') or ''}")
        return


def _follow(printer, name, wire):
    """block on the broadcast and print it until the agent goes away."""
    while True:
        try:
            messages = wire.recv()
        except OSError:
            messages = None
        if messages is None:
            printer.line(f"[{name} finished]")
            return 0
        for msg in messages:
            _print_msg(printer, msg)


def run(name):
    """the `cai --tail` entry: tail `name`, or pick a live agent when name is
    empty. returns the process exit code."""
    if not name:
        names = live_names()
        if not names:
            print("no available running agent")
            return 1
        name = _pick(names)
        if name is None:
            return 1
    try:
        conn = AgentsRegistry.connect(name)
    except OSError:
        print(f"no running agent named {name!r}", file=sys.stderr)
        return 1
    # the stream attaches before the snapshot is read, so a message completing
    # in between renders twice at worst - never gets lost.
    wire = Wire(conn)
    printer = _Printer(show_reasoning=_show_reasoning())
    _replay(printer, _control(name, "get_messages") or [])
    try:
        return _follow(printer, name, wire)
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            conn.close()
        except OSError:
            pass
