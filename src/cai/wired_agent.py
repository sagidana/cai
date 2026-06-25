"""wired: serve an Agent over a byte channel using the wire protocol.

WiredAgent wraps an Agent and drives it from messages read off a channel: a
client SUBMITs turns (streamed back as EVENT + a final RESULT) and issues CONTROL
ops to read or replace the agent's state - messages, skills, tools. The channel
is anything with recv()/sendall() (a socket, one end of a socket pair) and is
owned by the caller; the wire framing lives in cai.wire.Wire.

serve() runs two threads: a reader that owns recv() and demuxes inbound messages,
and a worker (the caller's thread) that runs turns and control ops serially. The
reader handles the fast control plane while a turn is in flight - STEER folds a
message into the running run, INTERRUPT kills it, REPLY answers a pending UI
prompt - so these land mid-turn rather than waiting for the turn to end. Turns and
state-touching CONTROL ops are queued to the worker, so the agent's conversation
is only ever mutated by one thread.

A served run's UI prompts travel the wire too: the agent is given a WireUI, so a
hook calling ctx.ui.confirm/select/text/notify mid-run reaches the client as a
PROMPT and blocks on a reply box the reader resolves from the matching REPLY.

UnixWiredAgent bakes the transport in: it owns a UnixSocketServer and serves the
agent over it, so a caller hands it an Agent and a path instead of wiring up the
listener, accept, and channel themselves.

One run at a time (one conversation), one client per connection. Multi-client
attach/snapshot are later layers."""
from __future__ import annotations

import itertools
import logging
import queue
import threading

from cai.agent import _tool_name
from cai.channel import UnixSocketServer
from cai.ui import BaseUI
from cai.wire import Wire


log = logging.getLogger("cai")

# worker-queue item telling the work loop to exit - same (kind, ...) shape as
# the real items ("submit", text) / ("control", op, value).
_STOP = ("stop",)


def _tool_names(tools):
    """the exposed names of an agent's tools, for sending over the wire (a
    callable can't cross JSON, but its name can)."""
    names = []
    for tool in tools:
        names.append(_tool_name(tool))
    return names


class WireUI(BaseUI):
    """the served agent's UI, bound to the wire. each prompt a hook raises mid-run
    is sent to the client as a PROMPT; the worker thread then blocks on a reply
    box until the WiredAgent's reader thread resolves it from the matching REPLY.
    close() (client gone / shutdown) wakes every pending prompt with no answer, so
    the caller falls back to the BaseUI default and a served run never hangs.

    this UI never reads the channel - the reader owns recv(); it only waits on a
    per-prompt Event the reader sets."""

    interactive = True

    def __init__(self, wire):
        self._wire = wire
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._pending = {}       # prompt id -> box {evt, value, answered}
        self._closed = False

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

    def resolve(self, pid, value):
        """called by the reader thread on a REPLY: hand the value to the waiting
        prompt and wake it. an unknown id (already answered / timed out) is a
        no-op."""
        with self._lock:
            box = self._pending.get(pid)
            if box is None: return
            box["value"] = value
            box["answered"] = True
        box["evt"].set()

    def close(self):
        """wake every pending prompt with no answer (client gone / shutdown) so
        each caller applies its BaseUI default, and refuse new prompts."""
        with self._lock:
            self._closed = True
            boxes = list(self._pending.values())
            self._pending.clear()
        for box in boxes:
            box["answered"] = False
            box["evt"].set()

    def _prompt(self, kind, message, *, options=None, default=None, detail="", secret=False):
        """send a PROMPT and block on a reply box until the reader resolves it
        (or close() wakes it). returns (answered, value): answered is False on a
        dropped client, so the caller applies its own default."""
        pid = str(next(self._ids))
        box = {}
        box["evt"] = threading.Event()
        box["value"] = None
        box["answered"] = False
        with self._lock:
            if self._closed:
                return False, None
            self._pending[pid] = box
        try:
            self._wire.send_prompt(pid,
                                   kind,
                                   message,
                                   options=options,
                                   default=default,
                                   detail=detail,
                                   secret=secret)
        except OSError:
            with self._lock:
                self._pending.pop(pid, None)
            return False, None
        box["evt"].wait()
        with self._lock:
            self._pending.pop(pid, None)
        if not box["answered"]:
            return False, None
        return True, box["value"]


class WiredAgent:
    """wraps an Agent and serves it over a byte channel using the wire protocol.

    serve() starts a reader thread that owns recv() and demuxes inbound messages,
    and runs a worker loop on the caller's thread:

      reader (recv):  SUBMIT/CONTROL -> worker queue   (touch agent state serially)
                      STEER -> agent.steer             (mid-run, thread-safe)
                      INTERRUPT -> agent.stop           (mid-run, thread-safe)
                      REPLY -> WireUI.resolve           (answer a pending prompt)
      worker:         runs each queued turn / control op, streaming EVENTs and a
                      final RESULT, or a CONTROL_RESULT.

    the 'kill' control op retires the agent and ends the serve loop (the reader
    fires agent.kill() at once so it bites a running turn). a turn or control op
    that raises is reported back (a RESULT carrying the 'Error:' text, or a
    failed CONTROL_RESULT) so one bad message ends cleanly."""

    def __init__(self, agent, channel):
        self.agent = agent
        self.wire = Wire(channel)
        self.ui = WireUI(self.wire)
        # the served agent's human is now the wire client: route its hooks' UI
        # prompts over the channel.
        agent._ui = self.ui
        self._work = queue.Queue()       # ("submit", text) | ("control", op, value)
        self._lock = threading.Lock()
        self._closed = False

    def serve(self):
        """serve the agent until the channel closes. a reader thread demuxes
        inbound messages; this (the caller's) thread is the worker. blocking."""
        reader = threading.Thread(target=self._read_loop,
                                  daemon=True,
                                  name="cai-wired-reader")
        reader.start()
        self._work_loop()

    def _read_loop(self):
        """own recv() and route each inbound message until the channel closes."""
        try:
            while True:
                messages = self.wire.recv()
                if messages is None: break
                for msg in messages:
                    self._dispatch(msg)
        except OSError:
            pass
        finally:
            self._shutdown()

    def _dispatch(self, msg):
        """route one inbound message. fast control-plane ops (steer / interrupt /
        reply) run here on the reader thread; anything that touches agent state
        (a turn, a control op) is queued to the worker so it stays serial."""
        kind = msg.get("type")
        if kind == Wire.SUBMIT:
            self._work.put(("submit", msg.get("text") or ""))
            return
        if kind == Wire.CONTROL:
            op = msg.get("op")
            if op == "kill":
                # kill must bite a running turn now, not wait its turn in the
                # worker queue - so retire the agent here on the reader thread.
                # the queued op still runs, to ack and end the serve loop.
                self.agent.kill()
            self._work.put(("control", op, msg.get("value")))
            return
        if kind == Wire.STEER:
            self.agent.steer(msg.get("text") or "")
            return
        if kind == Wire.INTERRUPT:
            self.agent.stop()
            return
        if kind == Wire.REPLY:
            self.ui.resolve(msg.get("id"), msg.get("value"))
            return

    def _work_loop(self):
        """run queued turns and control ops one at a time, until shutdown."""
        while True:
            item = self._work.get()
            kind = item[0]
            if kind == "stop": return

            try:
                if kind == "submit":
                    self._run_turn(item[1])
                elif kind == "control":
                    self._run_control(item[1], item[2])
                    if item[1] == "kill":
                        # the kill op has been acked; end the serve loop.
                        self._shutdown()
            except OSError:
                pass     # client gone mid-send; the reader's shutdown stops us

    def _shutdown(self):
        """the channel closed: abort an in-flight turn, wake any pending prompt,
        and tell the worker to exit."""
        with self._lock:
            if self._closed: return
            self._closed = True
        self.agent.stop()
        self.ui.close()
        self._work.put(_STOP)

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
        if op == "kill":
            agent.kill()
            return True, None, None
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
