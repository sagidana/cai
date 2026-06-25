"""wired: serve an Agent over a byte channel using the wire protocol.

WiredAgent wraps an Agent and drives it from messages read off a channel: a
client SUBMITs turns (streamed back as EVENT + a final RESULT) and issues CONTROL
ops to read or replace the agent's state - messages, skills, tools. The channel
is anything with recv()/sendall() (a socket, one end of a socket pair) and is
owned by the caller; the wire framing lives in cai.wire.Wire.

Minimal by design: one message at a time, synchronous, on the caller's thread.
Steering, interrupts, UI prompts, and attach/snapshot are later layers."""
from __future__ import annotations

import logging

from cai.agent import _tool_name
from cai.wire import Wire


log = logging.getLogger("cai")


def _tool_names(tools):
    """the exposed names of an agent's tools, for sending over the wire (a
    callable can't cross JSON, but its name can)."""
    names = []
    for tool in tools:
        names.append(_tool_name(tool))
    return names


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
