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

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

from cai.environment import Environment
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
    """a plain collection of (event, fn) pairs, fired in registration order.
    the install-wide hooks live on the Environment (cai.hook registers there);
    an Agent composes env hooks + its own hooks= list into one of these per
    run."""

    def __init__(self):
        self._entries = []

    @classmethod
    def from_list(cls, hooks):
        """build a registry from a list of (event, fn) pairs (None -> empty)."""
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


# the run-scoped state an in-process tool needs to dispatch another tool the way
# the loop does. a ContextVar - not a global - so two agents dispatching on two
# threads stay isolated, matching paths._scratch_provider.
_run_gate = ContextVar("cai_run_gate", default=None)


@dataclass(frozen=True)
class RunGate:
    """what call_llm publishes around its dispatch loop so an in-process tool can
    call another tool on the agent's behalf and still be gated. a tool that
    dispatches for the model (the python tool's tool_call()) reads it with
    current_gate() and routes through gated_dispatch, so an inner call fires the
    same before/after_tool_call hooks a top-level call does - a gate can veto it -
    without the inner result ever entering the conversation."""
    hooks: HooksRegistry
    dispatch: object   # callable(name, args) -> str, the run's tools_dispatch
    model: str
    config: Optional[Mapping]
    ui: UI
    messages: list
    usage: Optional[dict]
    hooks_data: Optional[dict]


def set_gate(gate):
    """publish the run gate for the current context; returns a reset token."""
    return _run_gate.set(gate)


def reset_gate(token):
    _run_gate.reset(token)


def current_gate():
    """the run gate for the current context, or None outside a run loop."""
    return _run_gate.get()


def gated_dispatch(gate, name, args, call_id="tool"):
    """dispatch one tool through the before/after_tool_call hooks: fire before (a
    False response vetoes, returning the same Error string the loop uses),
    dispatch, fire after, return the result. unlike the loop it does NOT append a
    tool message or fire messages_mutated - an inner call's result goes back to
    its caller, never into the conversation."""
    tool_call = ToolCall(name=name, arguments=json.dumps(args), args=args, id=call_id)
    data = dict(gate.hooks_data or {})
    before = HookContext(event=HookEvent.BEFORE_TOOL_CALL,
                         messages=gate.messages,
                         model=gate.model,
                         config=gate.config,
                         ui=gate.ui,
                         usage=gate.usage,
                         tool_call=tool_call,
                         data=data)
    vetoed = False
    for response in gate.hooks.fire(HookEvent.BEFORE_TOOL_CALL, before):
        if response is False:
            vetoed = True
    if vetoed:
        return f"Error: tool '{name}' was aborted by a before_tool_call hook"
    result = gate.dispatch(name, args)
    if result is None:
        result = ""
    result = str(result)
    after = HookContext(event=HookEvent.AFTER_TOOL_CALL,
                        messages=gate.messages,
                        model=gate.model,
                        config=gate.config,
                        ui=gate.ui,
                        usage=gate.usage,
                        tool_call=tool_call,
                        content=result,
                        data=data)
    gate.hooks.fire(HookEvent.AFTER_TOOL_CALL, after)
    return result


def hook(event):
    """decorator: register a hook for `event` on the current Environment, e.g.
    @cai.hook("after_turn"). it lands on the env being load()ed - else the
    process default - so once Environment.load() imports the extensions every
    run of an agent on that env fires it. see Environment / HooksRegistry."""
    if event not in VALID_HOOK_EVENTS:
        raise ValueError(f"unknown hook event: {event!r}. valid: {VALID_HOOK_EVENTS}")

    def decorator(fn):
        Environment.target().register_hook(event, fn)
        return fn
    return decorator
