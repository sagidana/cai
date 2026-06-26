"""A non-responsive wire must not wedge the run.

WiredAgent._broadcast pushes every turn EVENT to all attached wires. The TUI's
AgentClient drains only its stream wire continuously; the control wire is read
only during a control() call. Before the fix, a blocking sendall to the undrained
control wire filled its socket buffer and parked the agent's worker thread mid
sendall - the run never finished, the status froze on "waiting", and Ctrl-C was
swallowed.

The fix makes the broadcast best-effort: a wire whose peer isn't draining loses
the event (the sender abandons it) instead of blocking everyone, and the receiver
resyncs past the partial line the abandon leaves behind. These tests pin both
halves down against the real WiredAgent and the real Wire framing.

Fully offline: socket.socketpairs are the channels, a FakeApi streams the bytes.
"""
import socket
import threading
import time

from cai.wire import Wire
from cai.wired_agent import WiredAgent

from test_wired_agent import make_agent


class FloodApi:
    """one turn whose answer is `count` chunks of `size` bytes - enough EVENT
    traffic to overflow a small socket buffer on an undrained wire."""

    def __init__(self, count=300, size=2000):
        self.count = count
        self.size = size

    def chat(self, messages, model, **kwargs):
        count = self.count
        blob = "x" * self.size
        def gen():
            for _ in range(count):
                yield (blob, None, None, {})
        return gen()


def _small_pair():
    """a socketpair whose buffers are shrunk so a few KB of unread data backs the
    sender up - makes the backpressure deterministic instead of buffer-size luck."""
    srv, cli = socket.socketpair()
    for sock in (srv, cli):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2048)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
    return srv, cli


def _drain(wire, stop):
    """stand in for the TUI worker: keep reading the stream wire until stop is set,
    so it doesn't back up. discards everything - the test asserts on agent state,
    not on what (best-effort, lossy) traffic actually arrives here."""
    while not stop.is_set():
        try:
            messages = wire.recv()
        except OSError:
            return
        if messages is None:
            return


def _wait_assistant(agent, timeout):
    """True once the worker has finished the turn - the assistant reply is appended
    to the live conversation. polls agent state directly, so it doesn't depend on
    the lossy RESULT reaching any wire."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        messages = agent.messages
        if messages and messages[-1].get("role") == "assistant":
            return True
        time.sleep(0.02)
    return False


def test_undrained_wire_does_not_wedge_the_run():
    """the regression: a second wire is attached but never drained (the TUI's
    control wire). the worker must still finish the turn - it drops events to the
    stuck wire rather than blocking on it - even though the best-effort RESULT may
    never reach a consumer (the TUI worker's tolerance for that is a later layer)."""
    agent = make_agent(api=FloodApi())
    wired = WiredAgent(agent)

    # one wire drained (the TUI's stream), one attached but never drained (its
    # control wire) so it backs up - the exact shape of the two connections.
    srv_stream, cli_stream = socket.socketpair()
    srv_ctrl, cli_ctrl = _small_pair()
    wired.attach(srv_stream)
    wired.attach(srv_ctrl)             # attached, never drained
    wired.start()

    stream = Wire(cli_stream)
    stop = threading.Event()
    drainer = threading.Thread(target=_drain, args=(stream, stop), daemon=True)
    drainer.start()

    try:
        stream.send_submit("flood")
        assert _wait_assistant(agent, 5.0), "worker wedged on the undrained wire"
    finally:
        stop.set()
        for sock in (cli_stream, cli_ctrl, srv_stream, srv_ctrl):
            try:
                sock.close()
            except OSError:
                pass


def test_besteffort_send_drops_under_backpressure_without_blocking():
    """a best-effort send to a peer that isn't reading returns False (dropped)
    instead of blocking, once the socket buffer fills."""
    srv, cli = _small_pair()
    wire = Wire(srv)
    event = make_event("y" * 4000)

    dropped = False
    for _ in range(200):              # cli never reads -> the buffer fills fast
        if wire.send_event(event, besteffort=True): continue
        dropped = True
        break

    assert dropped, "best-effort send never reported a drop on a full buffer"
    srv.close()
    cli.close()


def test_feed_resyncs_past_a_corrupted_line():
    """an abandoned message leaves a partial (newline-less) prefix; the first whole
    message that follows is swallowed into that garbage line (its newline closes
    the prefix), and feed() resyncs cleanly from the second whole message on - it
    drops the garbage instead of raising on it."""
    wire = Wire(socket.socketpair()[0])

    partial = b'{"type": "event", "event": {"text": "tru'      # no newline: abandoned
    lost = Wire.encode({"type": Wire.RESULT, "text": "swallowed"})
    good = Wire.encode({"type": Wire.RESULT, "text": "after"})

    out = wire.feed(partial + lost + good)

    assert len(out) == 1
    assert out[0]["type"] == Wire.RESULT
    assert out[0]["text"] == "after"


def make_event(text):
    from cai.events import Event, EventType
    return Event(type=EventType.CONTENT, text=text)
