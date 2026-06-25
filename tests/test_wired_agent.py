"""Tests for cai.wired_agent - WiredAgent and WireUI over the wire protocol.

Integration-style but fully offline: a real socket.socketpair is the channel, a
background thread runs WiredAgent.serve, and the client side drives it with a
Wire. The model is a FakeApi that streams canned content - no network, no
config, no API key.
"""
import os
import socket
import threading

import pytest

from cai import channel
from cai.agent import Agent
from cai.hooks import HookEvent
from cai.llm import SteerQueue
from cai.skills import SkillsRegistry
from cai.tools import ToolRegistry
from cai.wire import Wire
from cai.wired_agent import UnixWiredAgent, WiredAgent, WireUI, _tool_names


# --------------------------------------------------------------------------
# fakes / helpers
# --------------------------------------------------------------------------

class FakeApi:
    """streams `chunks` as one turn's answer (no tool calls -> final). raise=exc
    makes chat raise, to exercise the error path."""

    def __init__(self, chunks=None, raise_exc=None):
        self.chunks = chunks
        if self.chunks is None:
            self.chunks = ["ok"]
        self.raise_exc = raise_exc

    def chat(self, messages, model, **kwargs):
        if self.raise_exc is not None:
            raise self.raise_exc
        chunks = self.chunks
        def gen():
            for chunk in chunks:
                yield (chunk, None, None, {})
        return gen()


def make_agent(tools=None, skills=None, hooks=None, api=None):
    """build an Agent without touching config/network (bypass __init__)."""
    agent = Agent.__new__(Agent)
    agent.name = "test"
    agent.model = "m"
    agent.api = api or FakeApi()
    agent._system_prompt = None
    agent._tools = tools or []
    agent._skills = skills or []
    agent.tools_registry = ToolRegistry.for_tools([])
    agent.skills_registry = SkillsRegistry.for_skills([], tools_registry=agent.tools_registry)
    agent._hooks = hooks
    agent._ui = None
    agent.interrupt = threading.Event()
    agent._steer = SteerQueue()
    agent.messages = []
    return agent


class ClientUI:
    """client-side UI that records prompts and returns canned answers."""

    interactive = True

    def __init__(self, confirm=True, select="green", text="bob"):
        self.calls = []
        self._confirm = confirm
        self._select = select
        self._text = text

    def confirm(self, message, *, default=False, detail=""):
        self.calls.append(("confirm", message, detail))
        return self._confirm

    def select(self, message, options, *, default=None, detail=""):
        self.calls.append(("select", message, options))
        return self._select

    def text(self, message, *, default="", secret=False):
        self.calls.append(("text", message, secret))
        return self._text

    def notify(self, message, *, level="info"):
        self.calls.append(("notify", message, level))


@pytest.fixture
def serve():
    """serve an agent on a background thread; yield the client-side Wire. all
    sockets are closed on teardown, which ends the serve loop."""
    socks = []
    def _serve(agent):
        server_sock, client_sock = socket.socketpair()
        socks.append(server_sock)
        socks.append(client_sock)
        wired = WiredAgent(agent, server_sock)
        thread = threading.Thread(target=wired.serve, daemon=True)
        thread.start()
        return Wire(client_sock)
    yield _serve
    for sock in socks:
        try:
            sock.close()
        except OSError:
            pass


def run_turn(wire, text, ui=None):
    """submit a turn and drain it; answer any PROMPT via `ui`. returns
    (events, result_text)."""
    wire.send_submit(text)
    events = []
    result = None
    while result is None:
        messages = wire.recv()
        for msg in messages:
            if ui is not None and wire.answer(msg, ui):
                continue
            if msg["type"] == Wire.EVENT:
                events.append(Wire.event_from_dict(msg["event"]))
            elif msg["type"] == Wire.RESULT:
                result = msg["text"]
    return events, result


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------

def test_serving_installs_a_wire_ui_on_the_agent():
    agent = make_agent()
    server_sock, client_sock = socket.socketpair()
    WiredAgent(agent, server_sock)
    assert isinstance(agent._ui, WireUI)
    server_sock.close()
    client_sock.close()


def test_tool_names_maps_callables_and_strings():
    def my_tool(x: str) -> str:
        return x
    assert _tool_names([my_tool, "srv__search"]) == ["my_tool", "srv__search"]


# --------------------------------------------------------------------------
# turns: SUBMIT -> EVENT* -> RESULT
# --------------------------------------------------------------------------

def test_turn_streams_events_then_result(serve):
    wire = serve(make_agent(api=FakeApi(chunks=["hello ", "world"])))
    events, result = run_turn(wire, "hi")
    streamed = ""
    for ev in events:
        if ev.type != "content": continue
        streamed += ev.text
    assert streamed == "hello world"
    assert result == "hello world"


def test_conversation_persists_across_turns(serve):
    agent = make_agent(api=FakeApi(chunks=["first"]))
    wire = serve(agent)
    run_turn(wire, "one")
    # second turn: the FakeApi answers "first" again, but messages keep growing
    agent.api = FakeApi(chunks=["second"])
    run_turn(wire, "two")
    ok, messages, error = wire.control("get_messages")
    roles = []
    for m in messages:
        roles.append(m["role"])
    assert roles == ["user", "assistant", "user", "assistant"]
    assert messages[0]["content"] == "one"
    assert messages[2]["content"] == "two"


def test_turn_error_comes_back_as_result(serve):
    wire = serve(make_agent(api=FakeApi(raise_exc=RuntimeError("boom"))))
    events, result = run_turn(wire, "hi")
    assert result.startswith("Error:")
    assert "boom" in result


# --------------------------------------------------------------------------
# control ops
# --------------------------------------------------------------------------

def test_control_get_messages_after_a_turn(serve):
    wire = serve(make_agent(api=FakeApi(chunks=["answer"])))
    run_turn(wire, "hi")
    ok, messages, error = wire.control("get_messages")
    assert ok is True
    assert error is None
    assert messages[-1]["content"] == "answer"


def test_control_set_then_get_messages(serve):
    wire = serve(make_agent())
    seed = [{"role": "user", "content": "seeded"}]
    ok, value, error = wire.control("set_messages", seed)
    assert ok is True
    ok, messages, error = wire.control("get_messages")
    assert messages == seed


def test_control_get_skills_empty(serve):
    wire = serve(make_agent())
    ok, skills, error = wire.control("get_skills")
    assert ok is True
    assert skills == []


def test_control_set_skills_records_names(serve):
    # an unknown skill is warned-and-skipped on activation, but the name set
    # still updates - no real skill file or MCP server is touched.
    wire = serve(make_agent())
    ok, value, error = wire.control("set_skills", ["does-not-exist"])
    assert ok is True
    ok, skills, error = wire.control("get_skills")
    assert skills == ["does-not-exist"]


def test_control_get_tools_returns_names(serve):
    def my_tool(x: str) -> str:
        return x
    wire = serve(make_agent(tools=[my_tool, "srv__search"]))
    ok, tools, error = wire.control("get_tools")
    assert ok is True
    assert set(tools) == {"my_tool", "srv__search"}


def test_control_set_tools_rejects_a_function_name(serve):
    # a function tool reads back by name but cannot be rebuilt from one over the
    # wire (a callable does not fit in JSON) - set_tools should fail cleanly.
    wire = serve(make_agent())
    ok, value, error = wire.control("set_tools", ["my_tool"])
    assert ok is False
    assert error is not None


def test_control_unknown_op_errs(serve):
    wire = serve(make_agent())
    ok, value, error = wire.control("frobnicate")
    assert ok is False
    assert "unknown control op" in error


# --------------------------------------------------------------------------
# UI prompts over the wire
# --------------------------------------------------------------------------

def test_hook_ui_prompts_travel_to_the_client(serve):
    seen = {}
    def on_final(ctx):
        seen["confirm"] = ctx.ui.confirm("proceed?", default=False, detail="risky")
        seen["select"] = ctx.ui.select("color?", ["red", "green"], default=0)
        seen["text"] = ctx.ui.text("name?", default="")
        ctx.ui.notify("thinking done", level="warn")

    agent = make_agent(api=FakeApi(chunks=["answer"]),
                       hooks=[(HookEvent.ON_FINAL_RESPONSE, on_final)])
    wire = serve(agent)
    ui = ClientUI(confirm=True, select="green", text="bob")
    events, result = run_turn(wire, "hi", ui=ui)

    assert result == "answer"
    assert seen["confirm"] is True
    assert seen["select"] == "green"
    assert seen["text"] == "bob"
    kinds = []
    for call in ui.calls:
        kinds.append(call[0])
    assert kinds == ["confirm", "select", "text", "notify"]


# --------------------------------------------------------------------------
# UnixWiredAgent: the unix-socket listener baked in
# --------------------------------------------------------------------------

def run_turn_over(client_wire, text):
    """drain one turn on an already-connected client Wire."""
    client_wire.send_submit(text)
    result = None
    while result is None:
        messages = client_wire.recv()
        for msg in messages:
            if msg["type"] == Wire.RESULT:
                result = msg["text"]
    return result


def test_unix_wired_agent_serves_over_a_socket(tmp_path):
    path = str(tmp_path / "a.sock")
    agent = make_agent(api=FakeApi(chunks=["hello"]))
    served = UnixWiredAgent(agent, path)
    assert os.path.exists(path)
    thread = threading.Thread(target=served.serve, daemon=True)
    thread.start()

    client = channel.connect(path)
    wire = Wire(client)
    assert run_turn_over(wire, "hi") == "hello"
    client.close()

    served.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not os.path.exists(path)        # socket file unlinked on close


def test_unix_wired_agent_persists_across_reconnects(tmp_path):
    path = str(tmp_path / "a.sock")
    agent = make_agent(api=FakeApi(chunks=["answer"]))
    served = UnixWiredAgent(agent, path)
    thread = threading.Thread(target=served.serve, daemon=True)
    thread.start()

    # first client: one turn, then disconnect
    first = channel.connect(path)
    run_turn_over(Wire(first), "one")
    first.close()

    # second client: the same agent, conversation carried over
    second = channel.connect(path)
    ok, messages, error = Wire(second).control("get_messages")
    second.close()
    assert ok is True
    assert messages[0]["content"] == "one"
    assert messages[-1]["content"] == "answer"

    served.close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_wire_ui_falls_back_to_default_when_client_gone():
    # no client will ever answer: each prompt resolves to the BaseUI default
    # instead of blocking the run forever.
    server_sock, client_sock = socket.socketpair()
    ui = WireUI(Wire(server_sock))
    client_sock.close()

    box = {}
    def ask():
        box["confirm"] = ui.confirm("ok?", default=True)
        box["text"] = ui.text("name?", default="z")
        box["select"] = ui.select("pick", ["a", "b"], default=1)

    thread = threading.Thread(target=ask)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert box == {"confirm": True, "text": None, "select": "b"}
    server_sock.close()
