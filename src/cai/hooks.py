"""hooks: the per-run registry plus the shapes passed to every hook.

A hook is an (event, fn) pair. HooksRegistry.fire(event, ctx) calls every fn
registered for that event, synchronously, in registration order, and returns
their responses. The caller decides whether to read the responses (a
before_tool_call hook returning False vetoes that tool; an on_final_response
hook returning a str replaces the answer) or ignore them. An exception in a
hook is logged and skipped, so one bad hook never breaks the turn.

HookContext is the one shape every hook receives; ToolCall is the slice it
carries for the *_tool_call events. ctx.ui is the live frontend during a run
(so a hook can prompt the human) and a no-op UI otherwise. Fields that don't
apply to the firing event are None."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

from cai.ui import UI, NULL_UI


log = logging.getLogger("cai")


class HookEvent(str, Enum):
    """Every hook event the loop fires, in one place. A str-Enum, so a user
    hook can register with the plain string ('before_tool_call') while core
    code uses the member (HookEvent.BEFORE_TOOL_CALL) - they compare equal."""
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    MESSAGES_MUTATED = "messages_mutated"
    MESSAGES_LOADED = "messages_loaded"
    AFTER_TURN = "after_turn"
    ON_FINAL_RESPONSE = "on_final_response"
    AFTER_RUN = "after_run"

    def __str__(self):
        return self.value


VALID_HOOK_EVENTS = tuple(e.value for e in HookEvent)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: str  # raw JSON string, as the model emitted it
    args: dict      # parsed (best-effort; {} on parse failure)
    id: str


@dataclass(frozen=True)
class HookContext:
    """Passed to every hook, one shape across all events. A before_tool_call
    hook returns False to skip that tool (the turn continues) - that is the
    only way to stop a tool."""
    event: str
    messages: list
    model: str
    config: Optional[Mapping] = None
    ui: UI = NULL_UI
    usage: Optional[dict] = None
    tool_call: Optional[ToolCall] = None
    content: Optional[str] = None
    data: Optional[dict] = None   # one unified bag: the caller's hooks_data
                                  # (identical across every context of a run)
                                  # with this event's own keys layered on top.


class HooksRegistry:
    # process-global hooks registered via cai.hook, as (event, fn, origin)
    # triples (origin is the file the hook was defined in). every HooksRegistry
    # built afterward bakes these in, so once extensions load every run fires
    # them. populated by register_global, cleared by reset_global.
    _registered = []

    def __init__(self):
        self._entries = []
        for event, fn, _origin in HooksRegistry._registered:
            self._entries.append((event, fn))

    @classmethod
    def register_global(cls, event, fn):
        """record a hook in the process-global store (cai.hook's backing). it is
        baked into every HooksRegistry created afterward. the origin file is
        captured from fn so userconfig.extension_for can attribute it later."""
        if event not in VALID_HOOK_EVENTS:
            raise ValueError(f"unknown hook event: {event!r}. valid: {VALID_HOOK_EVENTS}")
        origin = None
        code = getattr(fn, "__code__", None)
        if code is not None:
            origin = code.co_filename
        cls._registered.append((event, fn, origin))

    @classmethod
    def registered(cls):
        """the globally-registered hooks as (event, fn, origin) triples."""
        return list(cls._registered)

    @classmethod
    def reset_global(cls):
        """drop every globally-registered hook (test isolation)."""
        cls._registered = []

    @classmethod
    def from_list(cls, hooks):
        """build a registry (globals already baked in) and add a list of
        (event, fn) pairs on top (None -> just the globals)."""
        registry = cls()
        if hooks is None:
            return registry
        for event, fn in hooks:
            registry.register(event, fn)
        return registry

    def register(self, event, fn):
        if event not in VALID_HOOK_EVENTS:
            raise ValueError(f"unknown hook event: {event!r}. valid: {VALID_HOOK_EVENTS}")
        self._entries.append((event, fn))

    def unregister(self, event, fn):
        for i, entry in enumerate(self._entries):
            if entry[0] != event: continue
            if entry[1] != fn: continue
            del self._entries[i]
            return

    def pairs(self):
        """the (event, fn) pairs a child/clone would inherit."""
        out = []
        for event, fn in self._entries:
            out.append((event, fn))
        return out

    def fire(self, event, ctx):
        responses = []
        for hook_event, fn in self._entries:
            if hook_event != event: continue
            try:
                responses.append(fn(ctx))
            except Exception:
                log.exception("hook %r for %s raised",
                              getattr(fn, '__name__', repr(fn)), event)
        return responses


def hook(event):
    """decorator: register a function as a global hook for `event`, e.g.
    @cai.hook("after_turn"). it is baked into every HooksRegistry built
    afterward, so once cai.userconfig.load() imports the extensions every run
    fires it. see HooksRegistry."""
    def decorator(fn):
        HooksRegistry.register_global(event, fn)
        return fn
    return decorator
