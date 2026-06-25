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


class ToolThenTextApi:
    """turn 1 asks for `tool_name`, turn 2 answers with text. records what it was
    handed on turn 2 so an injected steer turn can be asserted."""

    def __init__(self, tool_name, answer="final"):
        self.tool_name = tool_name
        self.answer = answer
        self.calls = 0
        self.turn2_messages = None

    def chat(self, messages, model, **kwargs):
        self.calls += 1
        n = self.calls
        if n == 2:
            self.turn2_messages = list(messages)
        name = self.tool_name
        answer = self.answer
        def gen():
            if n == 1:
                call = {"id": "c1", "type": "function",
                        "function": {"name": name, "arguments": "{}"}}
                yield (None, None, [call], {})
            else:
                yield (answer, None, None, {})
        return gen()


def make_agent(tools=None, skills=None, hooks=None, api=None):
    """build an Agent without touching config/network (bypass __init__). callable
    tools are registered for dispatch; MCP-name strings are kept in _tools (so
    get_tools sees them) but not loaded."""
    agent = Agent.__new__(Agent)
    agent.name = "test"
    agent.model = "m"
    agent.api = api or FakeApi()
    agent._system_prompt = None
    agent._tools = tools or []
    agent._skills = skills or []
    callables = []
    for tool in (tools or []):
        if callable(tool):
            callables.append(tool)
    agent.tools_registry = ToolRegistry.for_tools(callables)
    agent.skills_registry = SkillsRegistry.for_skills([], tools_registry=agent.tools_registry)
    agent._hooks = hooks
    agent._ui = None
    agent.reasoning_effort = None
    agent.temperature = None
    agent.max_steps = None
    agent.stream = True
    agent.interrupt = threading.Event()
    agent._killed = threading.Event()
    agent._steer = SteerQueue()
    agent.messages = []
    agent.children = []
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


def test_wire_ui_close_wakes_a_pending_prompt():
    # a prompt is in flight (registered and blocking); close() must wake it with
    # no answer so the caller falls back to the default.
    server_sock, client_sock = socket.socketpair()
    ui = WireUI(Wire(server_sock))
    result = {}
    def ask():
        result["confirm"] = ui.confirm("ok?", default=True)
    thread = threading.Thread(target=ask)
    thread.start()
    while True:                              # wait until the prompt is registered
        with ui._lock:
            if ui._pending: break
    ui.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result["confirm"] is True
    server_sock.close()
    client_sock.close()


# --------------------------------------------------------------------------
# control plane while a turn is in flight: STEER / INTERRUPT / mid-run PROMPT
# --------------------------------------------------------------------------

def read_until_tool_call(wire):
    """drain events until turn 1 dispatches its tool (a TOOL_CALL event)."""
    while True:
        for msg in wire.recv():
            if msg["type"] != Wire.EVENT: continue
            if Wire.event_from_dict(msg["event"]).type == "tool_call":
                return


def read_to_result(wire):
    while True:
        for msg in wire.recv():
            if msg["type"] == Wire.RESULT:
                return msg["text"]


def test_steer_over_wire_folds_in_mid_run(serve):
    applied = threading.Event()
    def slow_tool() -> str:
        applied.wait(timeout=2)              # hold turn 1 until the steer is applied
        return "tool-done"

    api = ToolThenTextApi(tool_name="slow_tool")
    agent = make_agent(tools=[slow_tool], api=api)
    # signal once the reader applies the steer, so the tool releases deterministically
    inner_steer = agent.steer
    def steer_and_signal(text):
        inner_steer(text)
        applied.set()
    agent.steer = steer_and_signal

    wire = serve(agent)
    wire.send_submit("do X")
    read_until_tool_call(wire)               # turn 1's drain already happened (empty)
    wire.send_steer("also consider Y")
    result = read_to_result(wire)

    assert result == "final"
    seen = []
    for m in api.turn2_messages:
        seen.append((m["role"], m.get("content")))
    assert ("tool", "tool-done") in seen
    assert ("user", "also consider Y") in seen   # folded in after the tool, before turn 2


def test_interrupt_over_wire_stops_the_run(serve):
    applied = threading.Event()
    def slow_tool() -> str:
        applied.wait(timeout=2)              # hold turn 1 until the interrupt is applied
        return "tool-done"

    api = ToolThenTextApi(tool_name="slow_tool")
    agent = make_agent(tools=[slow_tool], api=api)
    inner_stop = agent.stop
    def stop_and_signal():
        inner_stop()
        applied.set()
    agent.stop = stop_and_signal

    wire = serve(agent)
    wire.send_submit("go")
    read_until_tool_call(wire)
    wire.send_interrupt()
    result = read_to_result(wire)

    assert api.calls == 1                     # the second model turn never ran
    assert result == ""                       # partial: no content this run


def test_prompt_answered_during_a_multi_turn_run(serve):
    asked = {}
    def before_tool(ctx):
        asked["confirm"] = ctx.ui.confirm("run the tool?", default=False)

    def the_tool() -> str:
        return "tool-ran"

    api = ToolThenTextApi(tool_name="the_tool")
    agent = make_agent(tools=[the_tool],
                       api=api,
                       hooks=[(HookEvent.BEFORE_TOOL_CALL, before_tool)])
    wire = serve(agent)
    ui = ClientUI(confirm=True)
    events, result = run_turn(wire, "go", ui=ui)

    assert asked["confirm"] is True           # prompt answered while the run was in flight
    assert result == "final"


# --------------------------------------------------------------------------
# the kill control op: retire the agent and end the serve loop
# --------------------------------------------------------------------------

def test_kill_control_op_retires_agent_and_ends_serve():
    agent = make_agent(api=FakeApi(chunks=["hi"]))
    server_sock, client_sock = socket.socketpair()
    thread = threading.Thread(target=WiredAgent(agent, server_sock).serve, daemon=True)
    thread.start()
    wire = Wire(client_sock)

    assert run_turn_over(wire, "first") == "hi"     # a normal turn first
    ok, value, error = wire.control("kill")
    assert ok is True
    assert agent.killed is True

    thread.join(timeout=5)
    assert not thread.is_alive()                    # serve ended after kill
    client_sock.close()
    server_sock.close()


def test_kill_during_a_run_aborts_it():
    applied = threading.Event()
    def slow_tool() -> str:
        applied.wait(timeout=2)                     # hold turn 1 until kill is applied
        return "tool-done"

    api = ToolThenTextApi(tool_name="slow_tool")
    agent = make_agent(tools=[slow_tool], api=api)
    inner_kill = agent.kill
    def kill_and_signal():
        inner_kill()
        applied.set()
    agent.kill = kill_and_signal

    server_sock, client_sock = socket.socketpair()
    thread = threading.Thread(target=WiredAgent(agent, server_sock).serve, daemon=True)
    thread.start()
    wire = Wire(client_sock)

    wire.send_submit("go")
    read_until_tool_call(wire)
    wire.send_control("kill")
    # drain: a partial RESULT for the aborted turn, then the CONTROL_RESULT ok
    killed_ok = None
    while killed_ok is None:
        for msg in wire.recv():
            if msg["type"] == Wire.CONTROL_RESULT:
                killed_ok = msg["ok"]

    assert api.calls == 1                           # the second model turn never ran
    assert killed_ok is True
    assert agent.killed is True
    thread.join(timeout=5)
    assert not thread.is_alive()
    client_sock.close()
    server_sock.close()
