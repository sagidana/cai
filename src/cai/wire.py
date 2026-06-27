"""wire: a minimal newline-delimited JSON protocol for driving an Agent over a
byte channel, in one class.

A Wire wraps a connected channel (anything with recv()/sendall() - a socket, one
end of a socket pair) and speaks the whole protocol: it frames outbound messages,
reassembles inbound ones from arbitrary recv chunks, and knows the message kinds.
The protocol is deliberately tiny - just enough for a client to run turns on a
remote agent:

  client -> agent   SUBMIT  {text}        start a user turn
  agent  -> client  EVENT   {event}       one cai.events.Event as the run streams
  agent  -> client  RESULT  {text}        the final answer; marks the turn's end
  client -> agent   CONTROL {op, value}   a request/reply op (get/set state)
  agent  -> client  CONTROL_RESULT {ok, value, error}   the op's answer
  agent  -> client  PROMPT  {id, kind, ...}   a UI prompt a hook raised
  client -> agent   REPLY   {id, value}   the human's answer to a PROMPT
  client -> agent   STEER   {text}        fold a message into the running turn
  client -> agent   INTERRUPT             abort the in-flight run

  w = Wire(channel)
  w.send_submit("hello")              # client side
  for msg in w.recv():                # -> list of decoded message dicts
      ...
  w.send_event(event); w.send_result(text)   # agent side
  ok, value, error = w.control("get_messages")   # client request/reply op
  w.answer(msg, ui)                   # client side: answer a PROMPT via a local UI

The control ops are get_info, get_messages/set_messages, set_model,
get_selected_skills/get_available_skills/set_selected_skills, and the matching
tools trio. PROMPT carries a UI request (confirm/select/text, or a
one-way notify) that a hook raised mid-run via HookContext.ui; the client
answers with a REPLY (none for notify). Everything is synchronous: the agent
handles one message fully before the next, and a blocked prompt reads the
channel itself until its REPLY arrives. Skills and tools cross as name lists; a
function tool reads back by name but cannot be re-added from a name (a callable
does not fit in JSON).

send() is thread-safe (a lock guards the channel write), because a served agent
sends from both its worker thread (events/result/prompts) and its reader thread
(control results). attach/snapshot are later layers."""
from __future__ import annotations

import json
import logging
import socket
import threading

from cai.events import Event


log = logging.getLogger("cai")


class Wire:
    """the wire protocol over one byte channel: message kinds, framing, and the
    decode buffer in a single object. one Wire per connection."""

    # message kinds. agent -> client: EVENT, RESULT, CONTROL_RESULT.
    # client -> agent: SUBMIT, CONTROL.
    EVENT = "event"
    RESULT = "result"
    SUBMIT = "submit"
    CONTROL = "control"
    CONTROL_RESULT = "control_result"
    PROMPT = "prompt"
    REPLY = "reply"
    STEER = "steer"
    INTERRUPT = "interrupt"

    def __init__(self, channel):
        self.channel = channel
        self._buf = b""
        self._send_lock = threading.Lock()

    # --- outbound: build a typed message, frame it, write it to the channel ---
    def send_event(self, event, besteffort=False):
        msg = {}
        msg["type"] = self.EVENT
        msg["event"] = self.event_to_dict(event)
        return self.send(msg, besteffort=besteffort)

    def send_result(self, text, besteffort=False):
        msg = {}
        msg["type"] = self.RESULT
        msg["text"] = text
        return self.send(msg, besteffort=besteffort)

    def send_submit(self, text):
        msg = {}
        msg["type"] = self.SUBMIT
        msg["text"] = text
        self.send(msg)

    def send_steer(self, text):
        msg = {}
        msg["type"] = self.STEER
        msg["text"] = text
        self.send(msg)

    def send_interrupt(self):
        msg = {}
        msg["type"] = self.INTERRUPT
        self.send(msg)

    def send_control(self, op, value=None):
        msg = {}
        msg["type"] = self.CONTROL
        msg["op"] = op
        msg["value"] = value
        self.send(msg)

    def send_control_result(self, ok, value=None, error=None):
        msg = {}
        msg["type"] = self.CONTROL_RESULT
        msg["ok"] = ok
        msg["value"] = value
        msg["error"] = error
        self.send(msg)

    def send_prompt(self,
                    id,
                    kind,
                    message,
                    *,
                    options=None,
                    default=None,
                    detail="",
                    secret=False,
                    level="info",
                    besteffort=False):
        msg = {}
        msg["type"] = self.PROMPT
        msg["id"] = id
        msg["kind"] = kind
        msg["message"] = message
        msg["options"] = options
        msg["default"] = default
        msg["detail"] = detail
        msg["secret"] = secret
        msg["level"] = level
        return self.send(msg, besteffort=besteffort)

    def send_reply(self, id, value):
        msg = {}
        msg["type"] = self.REPLY
        msg["id"] = id
        msg["value"] = value
        self.send(msg)

    def send(self, message, besteffort=False):
        """frame one message dict and write it to the channel. thread-safe: a
        served agent sends from both its worker and reader threads.

        besteffort=True (broadcast traffic) won't block on a peer that isn't
        draining its socket: it abandons the message the instant the buffer is full
        and returns False, so one stuck client can't stall the send to every other
        wire. a message abandoned mid-write leaves a partial line the receiver's
        feed() resyncs past. the default reliable send blocks until the whole
        message is out and returns True."""
        data = self.encode(message)
        with self._send_lock:
            if besteffort:
                return self._write_nonblocking(data)
            self.channel.sendall(data)
            return True

    def _write_nonblocking(self, data):
        """write as much of `data` as the kernel buffer takes right now, without
        blocking. returns True once it all went out; False the moment the peer
        can't take more (backpressure), leaving a partial line behind on purpose. a
        dead-socket OSError propagates so the caller can drop the wire."""
        view = memoryview(data)
        while view:
            try:
                sent = self.channel.send(view, socket.MSG_DONTWAIT)
            except BlockingIOError:
                return False
            if sent == 0:
                return False
            view = view[sent:]
        return True

    def control(self, op, value=None):
        """client side: send a control request and block for its reply,
        returning (ok, value, error). use between turns - it reads until the
        next CONTROL_RESULT. None on the wire (EOF) reads as disconnected."""
        self.send_control(op, value)
        while True:
            messages = self.recv()
            if messages is None:
                return False, None, "disconnected"
            for msg in messages:
                if msg.get("type") != self.CONTROL_RESULT: continue
                return msg.get("ok", False), msg.get("value"), msg.get("error")

    def answer(self, msg, ui):
        """client side: if msg is a PROMPT, answer it through a local UI and send
        the REPLY (a one-way notify gets no reply). returns True when it handled
        the message, so a client loop can `if w.answer(msg, ui): continue`."""
        if msg.get("type") != self.PROMPT:
            return False
        kind = msg.get("kind")
        message = msg.get("message")
        default = msg.get("default")
        detail = msg.get("detail") or ""
        if kind == "notify":
            ui.notify(message, level=msg.get("level", "info"))
            return True
        if kind == "status":
            ui.status(message)
            return True
        if kind == "confirm":
            value = ui.confirm(message, default=bool(default), detail=detail)
        elif kind == "select":
            value = ui.select(message, msg.get("options") or [], default=default, detail=detail)
        elif kind == "text":
            base = default
            if base is None: base = ""
            value = ui.text(message, default=base, secret=msg.get("secret", False))
        else:
            value = default
        self.send_reply(msg.get("id"), value)
        return True

    # --- inbound: read the channel and decode whole messages ---
    def recv(self):
        """read one chunk from the channel and return the list of whole messages
        it completes (possibly empty). None at end of stream (EOF)."""
        chunk = self.channel.recv(65536)
        if not chunk:
            return None
        return self.feed(chunk)

    def feed(self, chunk):
        """reassemble whole messages from one recv chunk. a channel splits the
        stream on no particular boundary, so a partial trailing line is held until
        its newline arrives. a line that won't decode is dropped, not raised: a
        best-effort broadcast abandons a message mid-write under backpressure, so
        the receiver must resync at the next newline rather than die on the garbage
        the abandoned bytes leave behind."""
        out = []

        if not chunk: return out
        self._buf += chunk

        while True:
            nl = self._buf.find(b"\n")
            if nl < 0: break
            line = self._buf[:nl]
            self._buf = self._buf[nl + 1:]
            if not line.strip(): continue
            try:
                message = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                log.warning("wire: dropping undecodable line (%d bytes)", len(line))
                continue
            out.append(message)
        return out

    # --- framing + Event (de)serialization: stateless, usable without a channel ---
    @staticmethod
    def encode(message):
        """serialize one message to a framed, newline-terminated line. default=str
        so a stray non-serializable value degrades to its string form instead of
        killing the whole stream."""
        line = json.dumps(message, default=str)
        return (line + "\n").encode("utf-8")

    @staticmethod
    def event_to_dict(event):
        """flatten a cai.events.Event to a plain dict, field for field, so it
        round-trips losslessly over the wire."""
        out = {}
        out["type"] = event.type
        out["text"] = event.text
        out["tool_name"] = event.tool_name
        out["tool_args"] = event.tool_args
        out["tool_call_id"] = event.tool_call_id
        out["tool_result"] = event.tool_result
        out["is_error"] = event.is_error
        out["usage"] = event.usage
        return out

    @staticmethod
    def event_from_dict(data):
        """rebuild a cai.events.Event from the dict event_to_dict produced."""
        event = Event(type=data["type"])
        event.text = data.get("text")
        event.tool_name = data.get("tool_name")
        event.tool_args = data.get("tool_args")
        event.tool_call_id = data.get("tool_call_id")
        event.tool_result = data.get("tool_result")
        event.is_error = data.get("is_error", False)
        event.usage = data.get("usage")
        return event
