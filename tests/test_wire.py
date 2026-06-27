"""Tests for cai.wire - the newline-delimited JSON protocol (Wire).

Fully offline: a FakeChannel stands in for a socket, buffering outbound bytes
and replaying programmed recv chunks. No real sockets or threads here - the
WiredAgent integration (threads + socketpair) lives in test_wired_agent.
"""
import json

from cai.events import Event
from cai.wire import Wire


# --------------------------------------------------------------------------
# fakes / helpers
# --------------------------------------------------------------------------

class FakeChannel:
    """a socket-like channel: sendall buffers outbound bytes; recv pops the next
    programmed inbound chunk (exhausted = EOF, returned as b'')."""

    def __init__(self, inbox=None):
        self.sent = b""
        self.inbox = list(inbox or [])

    def sendall(self, data):
        self.sent += data

    def recv(self, n):
        if not self.inbox:
            return b""
        return self.inbox.pop(0)


def decode(data):
    """decode every framed message in a blob of sent bytes."""
    return Wire(FakeChannel()).feed(data)


def control_result(ok, value=None, error=None):
    msg = {}
    msg["type"] = Wire.CONTROL_RESULT
    msg["ok"] = ok
    msg["value"] = value
    msg["error"] = error
    return Wire.encode(msg)


def prompt(kind, message, *, id="1", options=None, default=None, detail="", secret=False, level="info"):
    msg = {}
    msg["type"] = Wire.PROMPT
    msg["id"] = id
    msg["kind"] = kind
    msg["message"] = message
    msg["options"] = options
    msg["default"] = default
    msg["detail"] = detail
    msg["secret"] = secret
    msg["level"] = level
    return msg


class FakeUI:
    """records each prompt and returns a canned answer."""

    def __init__(self, confirm=True, select="picked", text="typed"):
        self.calls = []
        self._confirm = confirm
        self._select = select
        self._text = text

    def confirm(self, message, *, default=False, detail=""):
        self.calls.append(("confirm", message, default, detail))
        return self._confirm

    def select(self, message, options, *, default=None, detail=""):
        self.calls.append(("select", message, options, default, detail))
        return self._select

    def text(self, message, *, default="", secret=False):
        self.calls.append(("text", message, default, secret))
        return self._text

    def notify(self, message, *, level="info"):
        self.calls.append(("notify", message, level))

    def status(self, message):
        self.calls.append(("status", message))


# --------------------------------------------------------------------------
# framing: encode / feed
# --------------------------------------------------------------------------

def test_encode_is_one_newline_framed_line():
    data = Wire.encode({"type": "submit", "text": "hi"})
    assert data.endswith(b"\n")
    assert data.count(b"\n") == 1
    assert json.loads(data) == {"type": "submit", "text": "hi"}


def test_feed_reassembles_a_split_message():
    w = Wire(FakeChannel())
    framed = Wire.encode({"type": "result", "text": "answer"})
    head, tail = framed[:5], framed[5:]
    assert w.feed(head) == []                       # no newline yet -> nothing
    assert w.feed(tail) == [{"type": "result", "text": "answer"}]


def test_feed_returns_multiple_messages_in_one_chunk():
    a = Wire.encode({"type": "submit", "text": "one"})
    b = Wire.encode({"type": "submit", "text": "two"})
    out = Wire(FakeChannel()).feed(a + b)
    assert out == [{"type": "submit", "text": "one"},
                   {"type": "submit", "text": "two"}]


def test_feed_skips_blank_lines():
    framed = b"\n" + Wire.encode({"type": "submit", "text": "hi"}) + b"\n"
    assert Wire(FakeChannel()).feed(framed) == [{"type": "submit", "text": "hi"}]


def test_feed_empty_chunk_yields_nothing():
    assert Wire(FakeChannel()).feed(b"") == []


# --------------------------------------------------------------------------
# Event (de)serialization
# --------------------------------------------------------------------------

def test_event_round_trips_field_for_field():
    ev = Event(type="tool_result",
               tool_name="grep",
               tool_args={"q": "x"},
               tool_call_id="c1",
               tool_result="found",
               is_error=True,
               usage={"prompt": 3})
    back = Wire.event_from_dict(Wire.event_to_dict(ev))
    assert back.type == "tool_result"
    assert back.tool_name == "grep"
    assert back.tool_args == {"q": "x"}
    assert back.tool_call_id == "c1"
    assert back.tool_result == "found"
    assert back.is_error is True
    assert back.usage == {"prompt": 3}


def test_event_from_dict_defaults_missing_fields():
    back = Wire.event_from_dict({"type": "content"})
    assert back.text is None
    assert back.is_error is False
    assert back.usage is None


# --------------------------------------------------------------------------
# outbound message constructors
# --------------------------------------------------------------------------

def test_send_event():
    ch = FakeChannel()
    ev = Event(type="content", text="hi")
    Wire(ch).send_event(ev)
    sent = decode(ch.sent)
    assert sent[0]["type"] == Wire.EVENT
    assert sent[0]["event"]["type"] == "content"
    assert sent[0]["event"]["text"] == "hi"


def test_send_result():
    ch = FakeChannel()
    Wire(ch).send_result("done")
    assert decode(ch.sent) == [{"type": Wire.RESULT, "text": "done"}]


def test_send_submit():
    ch = FakeChannel()
    Wire(ch).send_submit("go")
    assert decode(ch.sent) == [{"type": Wire.SUBMIT, "text": "go"}]


def test_send_control():
    ch = FakeChannel()
    Wire(ch).send_control("set_messages", [{"role": "user", "content": "x"}])
    sent = decode(ch.sent)
    assert sent[0]["type"] == Wire.CONTROL
    assert sent[0]["op"] == "set_messages"
    assert sent[0]["value"] == [{"role": "user", "content": "x"}]


def test_send_control_result():
    ch = FakeChannel()
    Wire(ch).send_control_result(True, value=["a"], error=None)
    sent = decode(ch.sent)
    assert sent[0]["type"] == Wire.CONTROL_RESULT
    assert sent[0]["ok"] is True
    assert sent[0]["value"] == ["a"]
    assert sent[0]["error"] is None


def test_send_prompt_carries_every_field():
    ch = FakeChannel()
    Wire(ch).send_prompt("9",
                         "select",
                         "pick one",
                         options=["a", "b"],
                         default=1,
                         detail="why",
                         secret=False,
                         level="info")
    sent = decode(ch.sent)
    assert sent[0]["type"] == Wire.PROMPT
    assert sent[0]["id"] == "9"
    assert sent[0]["kind"] == "select"
    assert sent[0]["message"] == "pick one"
    assert sent[0]["options"] == ["a", "b"]
    assert sent[0]["default"] == 1
    assert sent[0]["detail"] == "why"


def test_send_reply():
    ch = FakeChannel()
    Wire(ch).send_reply("9", True)
    assert decode(ch.sent) == [{"type": Wire.REPLY, "id": "9", "value": True}]


# --------------------------------------------------------------------------
# inbound: recv
# --------------------------------------------------------------------------

def test_recv_returns_decoded_messages():
    ch = FakeChannel(inbox=[Wire.encode({"type": "submit", "text": "hi"})])
    assert Wire(ch).recv() == [{"type": "submit", "text": "hi"}]


def test_recv_none_on_eof():
    assert Wire(FakeChannel(inbox=[])).recv() is None


# --------------------------------------------------------------------------
# client request/reply: control()
# --------------------------------------------------------------------------

def test_control_sends_request_and_returns_result():
    ch = FakeChannel(inbox=[control_result(True, value=["m"], error=None)])
    w = Wire(ch)
    ok, value, error = w.control("get_messages")
    assert (ok, value, error) == (True, ["m"], None)
    sent = decode(ch.sent)
    assert sent[0]["type"] == Wire.CONTROL
    assert sent[0]["op"] == "get_messages"


def test_control_disconnected_on_eof():
    w = Wire(FakeChannel(inbox=[]))
    assert w.control("get_messages") == (False, None, "disconnected")


# --------------------------------------------------------------------------
# client prompt answering: answer()
# --------------------------------------------------------------------------

def test_answer_ignores_non_prompt():
    ch = FakeChannel()
    handled = Wire(ch).answer({"type": Wire.RESULT, "text": "x"}, FakeUI())
    assert handled is False
    assert ch.sent == b""


def test_answer_confirm_sends_reply():
    ch = FakeChannel()
    ui = FakeUI(confirm=True)
    handled = Wire(ch).answer(prompt("confirm", "ok?", id="3", default=False, detail="d"), ui)
    assert handled is True
    assert ui.calls == [("confirm", "ok?", False, "d")]
    sent = decode(ch.sent)
    assert sent[0]["type"] == Wire.REPLY
    assert sent[0]["id"] == "3"
    assert sent[0]["value"] is True


def test_answer_select_passes_options_and_replies():
    ch = FakeChannel()
    ui = FakeUI(select="green")
    Wire(ch).answer(prompt("select", "color?", options=["red", "green"], default=0), ui)
    assert ui.calls[0][0] == "select"
    assert ui.calls[0][2] == ["red", "green"]
    assert decode(ch.sent)[0]["value"] == "green"


def test_answer_text_replies_with_typed_value():
    ch = FakeChannel()
    ui = FakeUI(text="bob")
    Wire(ch).answer(prompt("text", "name?", default=""), ui)
    assert ui.calls[0][0] == "text"
    assert decode(ch.sent)[0]["value"] == "bob"


def test_answer_notify_is_one_way():
    ch = FakeChannel()
    ui = FakeUI()
    handled = Wire(ch).answer(prompt("notify", "heads up", level="warn"), ui)
    assert handled is True
    assert ui.calls == [("notify", "heads up", "warn")]
    assert ch.sent == b""              # notify expects no reply


def test_answer_status_is_one_way():
    ch = FakeChannel()
    ui = FakeUI()
    handled = Wire(ch).answer(prompt("status", "compacting…"), ui)
    assert handled is True
    assert ui.calls == [("status", "compacting…")]
    assert ch.sent == b""              # status expects no reply
