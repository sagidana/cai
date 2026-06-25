"""wire: a minimal newline-delimited JSON protocol for driving an Agent over a
byte channel, in one class.

A Wire wraps a connected channel (anything with recv()/sendall() - a socket, one
end of a socket pair) and speaks the whole protocol: it frames outbound messages,
reassembles inbound ones from arbitrary recv chunks, and knows the message kinds.
The protocol is deliberately tiny - just enough for a client to run turns on a
remote agent:

  client -> agent   SUBMIT {text}     start a user turn
  agent  -> client  EVENT  {event}    one cai.events.Event as the run streams
  agent  -> client  RESULT {text}     the final answer; marks the turn's end

  w = Wire(channel)
  w.send_submit("hello")              # client side
  for msg in w.recv():                # -> list of decoded message dicts
      ...
  w.send_event(event); w.send_result(text)   # agent side

Steering, interrupts, prompts, attach/snapshot, and control ops are later layers."""
from __future__ import annotations

import json

from cai.events import Event


class Wire:
    """the wire protocol over one byte channel: message kinds, framing, and the
    decode buffer in a single object. one Wire per connection."""

    # message kinds. agent -> client: EVENT, RESULT. client -> agent: SUBMIT.
    EVENT = "event"
    RESULT = "result"
    SUBMIT = "submit"

    def __init__(self, channel):
        self.channel = channel
        self._buf = b""

    # --- outbound: build a typed message, frame it, write it to the channel ---
    def send_event(self, event):
        msg = {}
        msg["type"] = self.EVENT
        msg["event"] = self.event_to_dict(event)
        self.send(msg)

    def send_result(self, text):
        msg = {}
        msg["type"] = self.RESULT
        msg["text"] = text
        self.send(msg)

    def send_submit(self, text):
        msg = {}
        msg["type"] = self.SUBMIT
        msg["text"] = text
        self.send(msg)

    def send(self, message):
        """frame one message dict and write it to the channel."""
        self.channel.sendall(self.encode(message))

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
        stream on no particular boundary, so a partial trailing line is held
        until its newline arrives."""
        out = []

        if not chunk: return out
        self._buf += chunk

        while True:
            nl = self._buf.find(b"\n")
            if nl < 0: break
            line = self._buf[:nl]
            self._buf = self._buf[nl + 1:]
            if not line.strip(): continue
            out.append(json.loads(line.decode("utf-8")))
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
