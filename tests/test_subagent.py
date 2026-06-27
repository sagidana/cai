"""Tests for cai.subagent - launching, polling, and killing sub-agents.

Fully offline: the child agent's model is a FakeApi that streams canned content,
and each child is served over a real unix socket on background threads, driven by
the parent as a wire client. No network, no config, no API key.
"""
import time
import threading

import pytest

from cai.agent import Agent
from cai.tools import ToolsRegistry
from cai.skills import SkillsRegistry
from cai.subagent import _inherit_tools
from cai.subagent import _inherit_skills
from cai.subagent import _launch_agent
from cai.subagent import _wait_agent
from cai.subagent import _kill_agent


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """give each test its own ~/.config/cai so child sockets land in a fresh
    agents dir - no cross-test leftovers, no polluting the real home."""
    monkeypatch.setenv("HOME", str(tmp_path))


class FakeApi:
    """streams `chunks` as one turn's answer (no tool calls -> final)."""

    def __init__(self, chunks=None):
        self.chunks = chunks
        if self.chunks is None:
            self.chunks = ["ok"]

    def chat(self, messages, model, **kwargs):
        chunks = self.chunks
        def gen():
            for chunk in chunks:
                yield (chunk, None, None, {})
        return gen()


class BlockingApi:
    """streams empty deltas forever, so a run stays in flight until interrupted.
    `started` fires once the model call begins, so a test can wait for the child
    to actually be running before it kills it."""

    def __init__(self):
        self.started = threading.Event()

    def chat(self, messages, model, **kwargs):
        started = self.started
        def gen():
            started.set()
            while True:
                yield ("", None, None, {})
                time.sleep(0.005)
        return gen()


class GatedApi:
    """blocks the turn until `release` is set, then streams `chunks`. `started`
    fires when the model call begins, so a test can attach a waiter to the running
    child before letting it finish - making result delivery deterministic despite
    the child tearing itself down the instant the run completes."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.started = threading.Event()
        self.release = threading.Event()

    def chat(self, messages, model, **kwargs):
        chunks = self.chunks
        started = self.started
        release = self.release
        def gen():
            started.set()
            release.wait()
            for chunk in chunks:
                yield (chunk, None, None, {})
        return gen()


def make_parent(api, tools=None, skills=None):
    """a parent Agent without config/network (bypass __init__), carrying just the
    state the sub-agent tools read: model, api, registries, hooks, children."""
    parent = Agent.__new__(Agent)
    parent.name = "parent"
    parent.model = "m"
    parent.api = api
    parent._system_prompt = "PARENT PROMPT"
    # the registries are the agent's source of truth. callable tools register for
    # dispatch (so get_tools sees them); the skill names are set directly, without
    # activating real skill files.
    callables = []
    for tool in (tools or []):
        if callable(tool):
            callables.append(tool)
    parent.tools_registry = ToolsRegistry.for_tools(callables)
    parent.skills_registry = SkillsRegistry.for_skills([], tools_registry=parent.tools_registry)
    parent.skills_registry._names = list(skills or ["subagents"])
    parent._hooks = None
    parent.children = []
    return parent


# --------------------------------------------------------------------------
# reduce-only inheritance
# --------------------------------------------------------------------------

def foo(x: str) -> str:
    return x


def bar() -> str:
    return "b"


def test_inherit_tools_is_reduce_only_and_keeps_request_order():
    parent = make_parent(FakeApi(), tools=[foo, bar])
    assert _inherit_tools(parent, ["foo", "ghost"]) == [foo]
    assert _inherit_tools(parent, ["bar", "foo"]) == [bar, foo]
    assert _inherit_tools(parent, None) == []


def test_inherit_skills_is_reduce_only():
    parent = make_parent(FakeApi(), skills=["subagents", "fs"])
    assert _inherit_skills(parent, ["fs", "nope"]) == ["fs"]
    assert _inherit_skills(parent, None) == []


# --------------------------------------------------------------------------
# launch -> wait
# --------------------------------------------------------------------------

def test_launch_records_the_child_id():
    parent = make_parent(FakeApi(chunks=["done"]))
    message = _launch_agent(parent, "do the thing", "worker")
    assert "worker" in message
    assert parent.children == ["worker"]


def test_launch_dedupes_duplicate_names_and_returns_the_new_one():
    parent = make_parent(FakeApi(chunks=["x"]))
    first = _launch_agent(parent, "a", "twin")
    second = _launch_agent(parent, "b", "twin")
    assert "twin" in first
    assert "twin-2" in second
    assert parent.children == ["twin", "twin-2"]


def test_wait_attached_mid_run_returns_the_answer():
    api = GatedApi(["the answer"])
    parent = make_parent(api)
    _launch_agent(parent, "task", "worker")
    assert api.started.wait(5)            # child running, gated before any output
    box = {}
    def waiter():
        box["out"] = _wait_agent(parent, "worker", timeout=5)
    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.3)                       # let the waiter attach to the broadcast
    api.release.set()                     # now the run completes and broadcasts
    thread.join(10)
    assert box["out"] == "the answer"


def test_child_tools_are_restricted_to_the_requested_subset():
    api = GatedApi(["k"])
    parent = make_parent(api, tools=[foo, bar])
    _launch_agent(parent, "t", "w", tools=["foo", "ghost"])
    assert api.started.wait(5)
    box = {}
    def waiter():
        box["out"] = _wait_agent(parent, "w", timeout=5)
    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.3)
    api.release.set()
    thread.join(10)
    assert box["out"] == "k"


def test_wait_on_unknown_agent_is_an_error():
    parent = make_parent(FakeApi())
    out = _wait_agent(parent, "nope", timeout=1)
    assert "no sub-agent" in out


# --------------------------------------------------------------------------
# timeout + kill
# --------------------------------------------------------------------------

def test_wait_timeout_reports_still_running():
    api = BlockingApi()
    parent = make_parent(api)
    _launch_agent(parent, "loop forever", "runner")
    assert api.started.wait(5)
    out = _wait_agent(parent, "runner", timeout=1)
    assert "still running" in out
    _kill_agent(parent, "runner")


def test_kill_winds_a_running_child_down():
    api = BlockingApi()
    parent = make_parent(api)
    _launch_agent(parent, "loop forever", "runner")
    assert api.started.wait(5)
    out = _kill_agent(parent, "runner")
    assert "Killing" in out
    # the child aborts and tears down; once its socket is gone, wait reports it
    # finished (or catches the aborted run's empty result first - both terminal).
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        msg = _wait_agent(parent, "runner", timeout=1)
        if "finished" in msg or msg == "":
            break
        time.sleep(0.05)
    assert "finished" in msg or msg == ""


# --------------------------------------------------------------------------
# bootstrap via the 'subagents' skill on a real Agent
# --------------------------------------------------------------------------

def test_subagents_skill_bootstraps_the_tools():
    agent = Agent(model="m", api=FakeApi(), skills=["subagents"])
    assert agent.tools_registry.has("launch_agent")
    assert agent.tools_registry.has("wait_agent")
    assert agent.tools_registry.has("kill_agent")
    assert agent.children == []
    agent.close()


def test_bootstrapped_tools_drive_a_child_end_to_end():
    api = GatedApi(["hi"])
    agent = Agent(model="m", api=api, skills=["subagents"])
    out = agent.tools_registry.dispatch("launch_agent", {"prompt": "p", "name": "c1"})
    assert "c1" in out
    assert api.started.wait(5)
    box = {}
    def waiter():
        box["out"] = agent.tools_registry.dispatch("wait_agent",
                                                    {"agent_id": "c1", "timeout": 5})
    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.3)
    api.release.set()
    thread.join(10)
    assert box["out"] == "hi"
    agent.close()


def test_push_is_skipped_when_the_result_was_already_delivered():
    # when a wait_agent collected the result inline it sets the child's signal;
    # the owner's push must then be a no-op, so the parent isn't told twice.
    from cai.wired_agent import UnixWiredAgent
    from cai.subagent import _push_result_to_parent

    parent = Agent(model="m", api=FakeApi(chunks=["x"]), skills=["subagents"])
    served = UnixWiredAgent(parent)
    thread = threading.Thread(target=served.serve, daemon=True)
    thread.start()
    try:
        delivered = threading.Event()
        delivered.set()                       # wait_agent already returned the result
        _push_result_to_parent(parent.name, "c1", "the result", delivered)
        time.sleep(0.3)                       # give any (erroneous) steered turn time to land
        for m in list(parent.messages):
            assert "Sub-agent 'c1' finished" not in (m.get("content") or "")
    finally:
        served.close()
        parent.close()
        thread.join(timeout=5)


def test_child_result_steers_an_idle_served_parent():
    # a served, idle parent: when its child finishes, the owner thread pushes the
    # result as a STEER over the parent's socket, and the idle parent runs a turn
    # on it - so the result lands in the parent's conversation without wait_agent.
    from cai.wired_agent import UnixWiredAgent

    api = GatedApi(["child output"])
    parent = Agent(model="m", api=api, skills=["subagents"])
    served = UnixWiredAgent(parent)
    thread = threading.Thread(target=served.serve, daemon=True)
    thread.start()
    try:
        out = parent.tools_registry.dispatch("launch_agent", {"prompt": "p", "name": "c1"})
        assert "c1" in out
        assert api.started.wait(5)
        api.release.set()                 # child finishes, then steers its result up
        landed = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            for m in list(parent.messages):
                content = m.get("content") or ""
                if m.get("role") == "user" and "c1" in content and "child output" in content:
                    landed = content
                    break
            if landed is not None:
                break
            time.sleep(0.02)
        assert landed is not None         # the child's result reached the parent as a user turn
    finally:
        served.close()
        parent.close()
        thread.join(timeout=5)
