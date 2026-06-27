"""ui: the human-interaction surface a run exposes to hooks.

A UI is the object a hook reaches through HookContext.ui to ask the human a
question mid-run: confirm a risky tool, pick from options, read a line of text,
push a notification, or post a transient status-line note. Every primitive
returns a safe default, so a hook can call ctx.ui.confirm(...) without ever
special-casing "no human is reachable".

This is the inbound prompt surface only. Outbound telemetry is cai.events (the
Event stream a Run yields), and serving a UI over a socket is a later layer.
Agent/Run take a ui= object and thread it down to call_llm, which stamps it on
every HookContext; with no ui in scope the loop falls back to NULL_UI."""
from __future__ import annotations

import getpass
import sys

from typing import Protocol, runtime_checkable


@runtime_checkable
class UI(Protocol):
    """the human-interaction surface exposed to hooks as HookContext.ui. every
    method returns its default when no human is reachable, so a hook never has
    to special-case the headless case."""

    def confirm(self, message, *, default=False, detail=""): ...
    def select(self, message, options, *, default=None, detail=""): ...
    def text(self, message, *, default="", secret=False): ...
    def notify(self, message, *, level="info"): ...
    def status(self, message): ...

    @property
    def interactive(self): ...


class BaseUI:
    """a UI whose every primitive returns a safe default. subclass and override
    only the prompts a real frontend can answer; the rest stay headless-safe."""

    interactive = False

    def confirm(self, message, *, default=False, detail=""):
        return default

    def select(self, message, options, *, default=None, detail=""):
        if isinstance(default, int) and 0 <= default < len(options):
            return options[default]
        return default

    def text(self, message, *, default="", secret=False):
        return None

    def notify(self, message, *, level="info"):
        pass

    def status(self, message):
        pass


class TerminalUI(BaseUI):
    """a UI that prompts the human over stdin/stdout - the surface the CLI hands
    to an Agent so a hook can ask y/n, pick an option, or read a line. when
    stdin is not a tty (a pipe, no human), every prompt falls back to the
    headless BaseUI default rather than blocking on a read that never answers."""

    def __init__(self):
        self.interactive = sys.stdin.isatty()

    def confirm(self, message, *, default=False, detail=""):
        if not self.interactive:
            return BaseUI.confirm(self, message, default=default, detail=detail)
        if default:
            suffix = "[Y/n]"
        else:
            suffix = "[y/N]"
        answer = self._ask(f"{message} {suffix} ", detail)
        if answer is None:
            return default
        answer = answer.strip().lower()
        if answer == "":
            return default
        return answer in ("y", "yes")

    def select(self, message, options, *, default=None, detail=""):
        options = list(options)
        if not self.interactive or not options:
            return BaseUI.select(self, message, options, default=default, detail=detail)
        if detail:
            sys.stderr.write(f"{detail}\n")
        sys.stderr.write(f"{message}\n")
        for i, option in enumerate(options):
            sys.stderr.write(f"  {i + 1}. {option}\n")
        sys.stderr.flush()
        answer = self._ask("> ", "")
        if answer is None:
            return BaseUI.select(self, message, options, default=default, detail=detail)
        answer = answer.strip()
        if answer == "":
            return BaseUI.select(self, message, options, default=default, detail=detail)
        if not answer.isdigit():
            return BaseUI.select(self, message, options, default=default, detail=detail)
        index = int(answer) - 1
        if 0 <= index < len(options):
            return options[index]
        return BaseUI.select(self, message, options, default=default, detail=detail)

    def text(self, message, *, default="", secret=False):
        if not self.interactive:
            return None
        prompt = f"{message} "
        try:
            if secret:
                answer = getpass.getpass(prompt)
            else:
                sys.stderr.write(prompt)
                sys.stderr.flush()
                answer = input()
        except (EOFError, KeyboardInterrupt):
            return None
        if answer == "":
            return default
        return answer

    def notify(self, message, *, level="info"):
        sys.stderr.write(f"[{level}] {message}\n")
        sys.stderr.flush()

    def status(self, message):
        sys.stderr.write(f"[status] {message}\n")
        sys.stderr.flush()

    def _ask(self, prompt, detail):
        """write an optional detail line, then a prompt, and read one line from
        stdin; None on EOF/Ctrl-C so the caller can apply its default."""
        if detail:
            sys.stderr.write(f"{detail}\n")
        sys.stderr.write(prompt)
        sys.stderr.flush()
        try:
            return input()
        except (EOFError, KeyboardInterrupt):
            return None


# the UI used when a hook fires with no frontend in scope (the headless path).
NULL_UI = BaseUI()
