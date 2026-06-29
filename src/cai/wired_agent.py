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

A WiredAgent serves one Agent to MANY wires at once. attach() adds a connection
and disconnect() drops one; a single reader thread select()s across every wire
plus a wake pipe (attach/disconnect write the pipe to break a blocked select and
rebuild its set). Output is a broadcast: a run's EVENTs and RESULT, and a mid-run
PROMPT, go to every connected wire - so all clients observe the one shared
conversation, and the first REPLY to a prompt wins. The exception is
CONTROL_RESULT, the reply to a get/set request: it is unicast back to the one
wire that asked, since a client blocks reading the channel for its own answer and
must not swallow another's.

UnixWiredAgent bakes the transport in: it owns a UnixSocketServer and one shared
WiredAgent, accepting every client into it with attach(), so a caller hands it an
Agent and a path instead of wiring up the listener, accept, and channels.

One run at a time (the single conversation is driven by one worker thread), many
clients per agent in a broadcast. The agent (and its conversation) persists across
(re)connects. The Agent itself is never modified to make this work."""
from __future__ import annotations

import itertools
import logging
import queue
import select
import socket
import threading

from cai.agent import _tool_name
from cai.agents_registry import AgentsRegistry
from cai.channel import UnixSocketServer
from cai.ui import BaseUI
from cai.wire import Wire


log = logging.getLogger("cai")

# control ops that mutate the live conversation: these are queued to the worker so
# they apply between turns and never race the run loop. every other control op
# (reads, and atomic config writes like set_model / set_selected_*) is answered
# inline on the reader thread, so a poll is served even while a turn is running.
_DEFERRED_OPS = ("set_messages", "load")


def _tool_names(tools):
    """the exposed names of an agent's tools, for sending over the wire (a
    callable can't cross JSON, but its name can)."""
    names = []
    for tool in tools:
        names.append(_tool_name(tool))
    return names


class WireUI(BaseUI):
    """the served agent's UI, bound to a sender that broadcasts. each prompt a
    hook raises mid-run is sent (to every connected wire) as a PROMPT; the worker
    thread then blocks on a reply box until the WiredAgent's reader thread resolves
    it from the FIRST matching REPLY - so with many clients the first to answer
    wins. close() (shutdown) wakes every pending prompt with no answer, so the
    caller falls back to the BaseUI default and a served run never hangs.

    the sender is anything with a send_prompt(...) - a single Wire, or the
    WiredAgent itself (which fans the prompt out to all its wires and raises
    OSError when there is no client to answer).

    this UI never reads the channel - the reader owns recv(); it only waits on a
    per-prompt Event the reader sets."""

    interactive = True

    def __init__(self, sender):
        self._sender = sender
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
            self._sender.send_prompt(None, "notify", message, level=level)
        except OSError:
            pass

    def status(self, message):
        # one-way like notify: a transient status-line note, best-effort so a
        # busy stream is never stalled by frequent progress updates.
        try:
            self._sender.send_prompt(None, "status", message, besteffort=True)
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

    def cancel_pending(self):
        """wake every in-flight prompt with no answer (e.g. the last client just
        left) so each caller falls back to its default - WITHOUT closing the UI,
        since a later client may reattach and prompt again."""
        with self._lock:
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
            self._sender.send_prompt(pid,
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
    """serves one Agent to many wires at once over the wire protocol.

    a single reader thread select()s across every attached wire (plus a wake pipe)
    and demuxes inbound messages; a worker thread runs turns and control ops one at
    a time, so the agent's one conversation is only ever touched serially:

      reader (select): SUBMIT/CONTROL -> worker queue  (touch agent state serially)
                       STEER -> agent.steer            (folds into a running turn;
                                                        when idle, queued as a submit)
                       INTERRUPT -> agent.stop          (mid-run, thread-safe)
                       REPLY -> WireUI.resolve          (first answer wins)
      worker:          runs each queued turn / control op. a turn's EVENTs and
                       RESULT broadcast to all wires; a CONTROL_RESULT is unicast
                       back to the one wire that asked.

    attach() registers a new wire and disconnect() drops one (both wake the reader
    so it rebuilds its select set); the agent persists across (re)connects. the
    'kill' control op retires the agent and ends serving (the reader fires
    agent.kill() at once so it bites a running turn). a turn or control op that
    raises is reported back (a RESULT carrying the 'Error:' text, or a failed
    CONTROL_RESULT) so one bad message ends cleanly.

    constructing with a channel attaches it at once, so the single-client form
    WiredAgent(agent, channel).serve() keeps working."""

    def __init__(self, agent, channel=None):
        self.agent = agent
        self.ui = WireUI(self)          # sender is self: prompts broadcast to all
        # the served agent's humans are now the wire clients: route its hooks' UI
        # prompts over the channels.
        agent._ui = self.ui
        self._wires = set()             # the live Wire connections
        self._wires_lock = threading.Lock()
        self._work = queue.Queue()      # ("submit", text) | ("control", op, value, wire)
        self._lock = threading.Lock()
        self._closed = False
        # a self-pipe so attach()/disconnect()/shutdown() can break the reader out
        # of a blocked select() to rebuild its wire set (or exit).
        self._wake_r, self._wake_w = socket.socketpair()
        if channel is not None:
            self.attach(channel)

    def attach(self, channel):
        """register a new client connection and wake the reader to watch it."""
        wire = Wire(channel)
        with self._wires_lock:
            self._wires.add(wire)
        self._wake()
        return wire

    def disconnect(self, wire):
        """drop a client connection, close its channel, and wake the reader. when
        the last client leaves, abort an in-flight turn and free any pending prompt
        (nobody is left to answer) - but keep serving for the next (re)connect."""
        with self._wires_lock:
            if wire not in self._wires:
                return
            self._wires.discard(wire)
            remaining = len(self._wires)
        try:
            wire.channel.close()
        except OSError:
            pass
        if remaining == 0:
            self.agent.stop()
            self.ui.cancel_pending()
        self._wake()

    def serve(self):
        """serve until shutdown. a reader thread select()s the wires; this (the
        caller's) thread is the worker. blocking."""
        reader = threading.Thread(target=self._read_loop,
                                  daemon=True,
                                  name="cai-wired-reader")
        reader.start()
        self._work_loop()

    def start(self):
        """spawn the reader AND worker as background threads and return at once -
        for a caller (UnixWiredAgent) that runs its own accept loop on this thread."""
        threading.Thread(target=self._read_loop,
                         daemon=True,
                         name="cai-wired-reader").start()
        threading.Thread(target=self._work_loop,
                         daemon=True,
                         name="cai-wired-worker").start()

    def shutdown(self):
        """retire the agent and end serving for every client."""
        self._shutdown()

    def _read_loop(self):
        """select() across the wake pipe and every attached wire; route each wire's
        inbound messages, dropping a wire on EOF. runs until shutdown."""
        while not self._closed:
            with self._wires_lock:
                wires = list(self._wires)
            channels = [self._wake_r]

            for wire in wires: channels.append(wire.channel)

            try:
                readable, _, _ = select.select(channels, [], [])
            except (OSError, ValueError):
                continue                 # a channel closed under us; rebuild the set

            if self._wake_r in readable: self._drain_wake()

            for wire in wires:
                if wire.channel not in readable: continue

                try:
                    messages = wire.recv()
                except OSError:
                    messages = None

                if messages is None:
                    self.disconnect(wire)
                    continue

                for msg in messages:
                    self._dispatch(wire, msg)

    def _wake(self):
        """nudge the reader's select() (a wire set change, or shutdown)."""
        try:
            self._wake_w.send(b"\x00")
        except OSError:
            pass

    def _drain_wake(self):
        try:
            self._wake_r.recv(65536)
        except OSError:
            pass

    def _dispatch(self, wire, msg):
        """route one inbound message from `wire`. steer / interrupt / reply run
        here on the reader thread, as do read and atomic-config control ops -
        answered inline via _run_control so a poll is served even while a turn
        runs. only a turn and the conversation-mutating control ops (_DEFERRED_OPS)
        are queued to the worker, which applies them serially. kill bites the run
        now and is also queued so it acks and ends serving. a control op carries
        its origin wire so its reply is unicast back."""
        kind = msg.get("type")
        if kind == Wire.SUBMIT:
            # a None text is a continue: _run_turn -> agent.run(None) re-enters
            # the loop without appending a user turn. real submits always carry
            # non-empty text (the TUI guards empty input), so nothing else hits
            # this path.
            self._work.put(("submit", msg.get("text")))
            return
        if kind == Wire.CONTROL:
            op = msg.get("op")
            if op == "kill":
                self.agent.kill()
                self._work.put(("control", op, msg.get("value"), wire))
                return
            if op in _DEFERRED_OPS:
                self._work.put(("control", op, msg.get("value"), wire))
                return
            self._run_control(op, msg.get("value"), wire)
            return
        if kind == Wire.STEER:
            # while a run is in flight steer() queues the text (folded in at the next
            # boundary) and returns True. while idle it returns False without acting,
            # so we run it as a normal submit on the worker - which broadcasts to
            # every client - instead of on this reader thread.
            text = msg.get("text") or ""
            if not self.agent.steer(text, run_on_idle=False):
                self._work.put(("submit", text))
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
            if item == "stop": return
            kind = item[0]

            try:
                if kind == "submit":
                    self._run_turn(item[1])
                elif kind == "control":
                    self._run_control(item[1], item[2], item[3])
                    if item[1] == "kill":
                        # the kill op has been acked; end serving.
                        self._shutdown()
            except OSError:
                pass     # a client went mid-send; the broadcast already dropped it

    def _shutdown(self):
        """retire the agent, wake any pending prompt, tell the worker to exit, and
        close every wire and the wake pipe so the reader stops."""
        with self._lock:
            if self._closed: return
            self._closed = True
        self.agent.stop()
        self.ui.close()
        self._work.put("stop")
        with self._wires_lock:
            wires = list(self._wires)
            self._wires.clear()
        for wire in wires:
            try:
                wire.channel.close()
            except OSError:
                pass
        self._wake()                     # break the reader's select() so it exits
        try:
            self._wake_w.close()
            self._wake_r.close()
        except OSError:
            pass

    def _broadcast(self, method, *args):
        """call `method`(*args) on every live wire as a best-effort send: a wire
        whose peer isn't draining its socket drops the message (it loses that
        event) instead of blocking the broadcast to every other wire - one stuck
        client can no longer wedge the run. a wire whose socket is dead raises
        OSError and is dropped. the set is snapshotted so a disconnect mid-broadcast
        can't mutate it."""
        with self._wires_lock:
            wires = list(self._wires)
        for wire in wires:
            send = getattr(wire, method)
            try:
                send(*args, besteffort=True)
            except OSError:
                self.disconnect(wire)

    def send_prompt(self, id, kind, message, **kw):
        """broadcast a PROMPT to every live wire (the WireUI's sender). raises
        OSError when no client is connected, so the WireUI falls back to its
        default rather than waiting on nobody."""
        with self._wires_lock:
            wires = list(self._wires)
        if not wires:
            raise OSError("no clients connected")
        for wire in wires:
            try:
                wire.send_prompt(id, kind, message, besteffort=True, **kw)
            except OSError:
                self.disconnect(wire)

    def _run_turn(self, text):
        """run one turn on the wrapped agent, broadcasting each Event to all wires
        and closing with the final answer (or an 'Error:' string on failure)."""
        try:
            run = self.agent.run(text)
            for event in run:
                self._broadcast("send_event", event)
            result = run.text
        except Exception as e:
            log.exception("wired turn failed")
            result = f"Error: {type(e).__name__}: {e}"
        self._broadcast("send_result", result)

    def _run_control(self, op, value, wire):
        """dispatch one control op and answer with a CONTROL_RESULT unicast back to
        the wire that asked. a raise becomes a failed result rather than killing the
        serve loop."""
        try:
            ok, result, error = self._control(op, value)
        except Exception as e:
            log.exception("wired control %r failed", op)
            ok, result, error = False, None, f"{type(e).__name__}: {e}"
        try:
            wire.send_control_result(ok, result, error)
        except OSError:
            pass

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
        if op == "get_selected_skills":
            return True, agent.get_selected_skills(), None
        if op == "get_available_skills":
            return True, agent.get_available_skills(), None
        if op == "set_selected_skills":
            agent.set_selected_skills(value)
            return True, None, None
        if op == "get_selected_tools":
            return True, agent.get_selected_tools(), None
        if op == "get_available_tools":
            return True, agent.get_available_tools(), None
        if op == "set_selected_tools":
            agent.set_selected_tools(value)
            return True, None, None
        if op == "set_model":
            agent.set_model(value)
            return True, None, None
        if op == "set_system_prompt_base":
            agent.set_system_prompt_base(value)
            return True, None, None
        if op == "get_info":
            children = list(agent.children)
            value = {}
            value["name"] = agent.name
            value["model"] = agent.model
            value["children"] = children
            value["system_prompt"] = agent.system_prompt
            value["system_prompt_base"] = agent._system_prompt
            return True, value, None
        if op == "save":
            # value is the target path, or None for a default timestamped file;
            # the path written comes back to the client. control ops run on the
            # worker thread between turns, so this never races a live run.
            return True, agent.save(value), None
        if op == "load":
            agent.load(value)
            return True, None, None
        return False, None, f"unknown control op: {op!r}"


class UnixWiredAgent:
    """serve an Agent over a unix-domain socket to many clients, listener baked in.

    binds a UnixSocketServer at `path` and owns one shared WiredAgent; serve()
    runs that hub and accepts every client into it with attach(), so all clients
    drive and observe the same agent in a broadcast - a turn any client submits
    streams to all - and the agent (and its conversation) persists across
    (re)connects. close() shuts the listener and the hub and unlinks the socket
    file, which ends serve().

    path defaults to the common agents folder: ~/.config/cai/agents/<name>.sock,
    keyed by the agent's name, so every UnixWiredAgent registers in one place. a
    caller wanting a private socket (a test, an isolated child) passes its own."""

    def __init__(self, agent, path=None, *, backlog=8):
        if path is None:
            AgentsRegistry.ensure_dir()
            path = AgentsRegistry.sock_path(agent.name)
        self.agent = agent
        self.server = UnixSocketServer(path, backlog=backlog)
        self.wired = WiredAgent(agent)

    @property
    def path(self):
        return self.server.path

    def serve(self):
        """run the shared hub and accept clients into it until the listener is
        closed. blocking; runs on the caller's thread (or one the caller spawns)."""
        self.wired.start()
        while True:
            try:
                conn = self.server.accept()
            except OSError:
                return                       # listener closed -> stop serving
            self.wired.attach(conn)

    def close(self):
        self.server.close()
        self.wired.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
