"""wired: serve an Agent over a byte channel using the wire protocol.

WiredAgent wraps an Agent and drives it from messages read off a channel: a
client SUBMITs turns (streamed back as EVENT + a final RESULT) and issues CONTROL
ops to read or replace the agent's state - messages, skills, tools. The channel
is anything with recv()/sendall() (a socket, one end of a socket pair) and is
owned by the caller; the wire framing lives in cai.wire.Wire.

A served run's UI prompts travel the wire too: the agent is given a WireUI, so a
hook calling ctx.ui.confirm/select/text/notify mid-run reaches the client as a
PROMPT and blocks for its REPLY.

UnixWiredAgent bakes the transport in: it owns a UnixSocketServer and serves the
agent over it, so a caller hands it an Agent and a path instead of wiring up the
listener, accept, and channel themselves.

Minimal by design: one message at a time, synchronous, on the caller's thread.
Steering, interrupts, and attach/snapshot are later layers."""
from __future__ import annotations

import itertools
import logging

from cai.agent import _tool_name
from cai.channel import UnixSocketServer
from cai.ui import BaseUI
from cai.wire import Wire


log = logging.getLogger("cai")


def _tool_names(tools):
    """the exposed names of an agent's tools, for sending over the wire (a
    callable can't cross JSON, but its name can)."""
    names = []
    for tool in tools:
        names.append(_tool_name(tool))
    return names


class WireUI(BaseUI):
    """the served agent's UI, bound to the wire: each prompt a hook raises mid-run
    is sent to the client as a PROMPT and blocks for the matching REPLY on the
    same channel. a dropped client (EOF) falls back to the BaseUI default, so a
    served run never hangs waiting for an answer that will not come.

    it shares the WiredAgent's Wire (one reader per channel): a prompt fires from
    inside a turn, while serve() is parked in that turn and not reading, so the
    prompt reads the REPLY itself - no second reader competes for the socket."""

    interactive = True

    def __init__(self, wire):
        self._wire = wire
        self._ids = itertools.count(1)

    def confirm(self, message, *, default=False, detail=""):
        answered, value = self._prompt("confirm",
                                       message,
                                       default=default,
                                       detail=detail)
        if not answered:
            return BaseUI.confirm(self, message, default=default, detail=detail)
        return bool(value)

    def select(self, message, options, *, default=None, detail=""):
        options = list(options)
        answered, value = self._prompt("select",
                                       message,
                                       options=options,
                                       default=default,
                                       detail=detail)
        if not answered:
            return BaseUI.select(self, message, options, default=default, detail=detail)
        if value in options:
            return value
        if isinstance(value, int) and 0 <= value < len(options):
            return options[value]
        return BaseUI.select(self, message, options, default=default, detail=detail)

    def text(self, message, *, default="", secret=False):
        answered, value = self._prompt("text", message, default=default, secret=secret)
        if not answered:
            return None
        return value

    def notify(self, message, *, level="info"):
        # one-way: notify expects no reply, so send and return.
        try:
            self._wire.send_prompt(None, "notify", message, level=level)
        except OSError:
            pass

    def _prompt(self, kind, message, *, options=None, default=None, detail="", secret=False):
        """send a PROMPT and block reading the channel for its REPLY. returns
        (answered, value): answered is False on a dropped client, so the caller
        applies its own default."""
        pid = str(next(self._ids))
        try:
            self._wire.send_prompt(pid,
                                   kind,
                                   message,
                                   options=options,
                                   default=default,
                                   detail=detail,
                                   secret=secret)
        except OSError:
            return False, None
        while True:
            messages = self._wire.recv()
            if messages is None:
                return False, None
            for msg in messages:
                if msg.get("type") != Wire.REPLY: continue
                if msg.get("id") != pid: continue
                return True, msg.get("value")


class WiredAgent:
    """wraps an Agent and serves it over a byte channel using the wire protocol.

    a client sends SUBMIT messages; each one runs a turn on the wrapped agent,
    streaming the run's Events out as EVENT messages and ending with a RESULT
    carrying the final answer. CONTROL messages read or replace agent state
    (get/set messages, skills, tools) and answer with a CONTROL_RESULT.

    a turn or control op that raises is reported back (a RESULT carrying the
    'Error:' text, or a failed CONTROL_RESULT) so one bad message ends cleanly
    instead of tearing down the serve loop."""

    def __init__(self, agent, channel):
        self.agent = agent
        self.wire = Wire(channel)
        # the served agent's human is now the wire client: route its hooks' UI
        # prompts over the channel (sharing this Wire, so a prompt reads its own
        # REPLY while serve() is parked in the turn that raised it).
        agent._ui = WireUI(self.wire)

    def serve(self):
        """read messages off the channel and handle each until it closes
        (a clean EOF or the socket dropping). blocking; runs on the caller's
        thread."""
        try:
            while True:
                messages = self.wire.recv()
                if messages is None: break
                for msg in messages:
                    self._handle(msg)
        except OSError:
            pass

    def _handle(self, msg):
        kind = msg.get("type")
        if kind == Wire.SUBMIT:
            self._run_turn(msg.get("text") or "")
            return
        if kind == Wire.CONTROL:
            self._run_control(msg.get("op"), msg.get("value"))
            return

    def _run_turn(self, text):
        """run one turn on the wrapped agent, relaying each Event to the channel
        and closing with the final answer (or an 'Error:' string on failure)."""
        try:
            run = self.agent.run(text)
            for event in run:
                self.wire.send_event(event)
            result = run.text
        except Exception as e:
            log.exception("wired turn failed")
            result = f"Error: {type(e).__name__}: {e}"
        self.wire.send_result(result)

    def _run_control(self, op, value):
        """dispatch one control op and answer with a CONTROL_RESULT. a raise
        becomes a failed result rather than killing the serve loop."""
        try:
            ok, result, error = self._control(op, value)
        except Exception as e:
            log.exception("wired control %r failed", op)
            ok, result, error = False, None, f"{type(e).__name__}: {e}"
        self.wire.send_control_result(ok, result, error)

    def _control(self, op, value):
        """run one control op, returning (ok, value, error). getters carry their
        value; setters return ok with no value. tools/skills are name lists."""
        agent = self.agent
        if op == "get_messages":
            return True, agent.get_messages(), None
        if op == "set_messages":
            agent.set_messages(value)
            return True, None, None
        if op == "get_skills":
            return True, agent.get_skills(), None
        if op == "set_skills":
            agent.set_skills(value)
            return True, None, None
        if op == "get_tools":
            return True, _tool_names(agent.get_tools()), None
        if op == "set_tools":
            agent.set_tools(value)
            return True, None, None
        return False, None, f"unknown control op: {op!r}"


class UnixWiredAgent:
    """serve an Agent over a unix-domain socket, with the listener baked in.

    binds a UnixSocketServer at `path`; serve() accepts one client at a time and
    runs it as a WiredAgent until the client disconnects, then loops to accept
    the next - the agent (and its conversation) persists across reconnects.
    close() shuts the listener and unlinks the socket file, which ends serve().

    one client at a time, synchronous: a second client waits in the accept
    backlog until the first disconnects. concurrent multi-client serving is a
    later layer."""

    def __init__(self, agent, path, *, backlog=8):
        self.agent = agent
        self.server = UnixSocketServer(path, backlog=backlog)

    @property
    def path(self):
        return self.server.path

    def serve(self):
        """accept and serve clients one at a time until the listener is closed.
        blocking; runs on the caller's thread (or one the caller spawns)."""
        while True:
            try:
                conn = self.server.accept()
            except OSError:
                return                       # listener closed -> stop serving
            try:
                WiredAgent(self.agent, conn).serve()
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def close(self):
        self.server.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
