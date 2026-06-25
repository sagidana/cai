"""ui: the human-interaction surface a run exposes to hooks.

A UI is the object a hook reaches through HookContext.ui to ask the human a
question mid-run: confirm a risky tool, pick from options, read a line of text,
or push a notification. Every primitive returns a safe default, so a hook can
call ctx.ui.confirm(...) without ever special-casing "no human is reachable".

This is the inbound prompt surface only. Outbound telemetry is cai.events (the
Event stream a Run yields), and serving a UI over a socket is a later layer.
Agent/Run take a ui= object and thread it down to call_llm, which stamps it on
every HookContext; with no ui in scope the loop falls back to NULL_UI."""
from __future__ import annotations

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


# the UI used when a hook fires with no frontend in scope (the headless path).
NULL_UI = BaseUI()
