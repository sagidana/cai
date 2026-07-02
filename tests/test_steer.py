"""Tests for cooperative steering - cai.agent.SteerQueue, cai.llm.call_llm
folding pending messages in at turn boundaries, and Agent.steer() plumbing.

Fully offline: a FakeApi streams canned chunks and records what it saw on a
given turn, so the injected user turn can be observed. No network, no config.
"""
import threading

import pytest

from cai.agent import Agent, RunInFlight
from cai.environment import Environment
from cai.llm import call_llm, SteerQueue
from cai.skills import SkillsRegistry
from cai.tools import ToolsRegistry


# --------------------------------------------------------------------------
# fakes / helpers
# --------------------------------------------------------------------------

def tool_call(name, call_id="c1", arguments="{}"):
    function = {}
    function["name"] = name
    function["arguments"] = arguments
    call = {}
    call["id"] = call_id
    call["type"] = "function"
    call["function"] = function
    return call


class ToolThenTextApi:
    """turn 1 asks for a tool, turn 2 answers with text. records the messages it
    was handed on turn 2, so an injected steer turn can be asserted."""

    def __init__(self):
        self.calls = 0
        self.turn2_messages = None

    def chat(self, messages, model, **kwargs):
        self.calls += 1
        n = self.calls
        if n == 2:
            self.turn2_messages = list(messages)
        def gen():
            if n == 1:
                yield (None, None, [tool_call("poke")], {})
            else:
                yield ("final", None, None, {})
        return gen()


def bare_agent(tools_registry, api):
    agent = Agent.__new__(Agent)
    agent.name = "test"
    agent.model = "m"
    agent.api = api
    agent.env = Environment()
    agent._system_prompt = None
    agent.tools_registry = tools_registry
    agent.skills_registry = SkillsRegistry.for_skills([], tools_registry=tools_registry)
    agent._hooks = None
    agent._ui = None
    agent.reasoning_effort = None
    agent.temperature = None
    agent.max_steps = None
    # run() reads this when wrapping the tool dispatcher; __init__ sets it, so a
    # helper that bypasses __init__ must too, or run() hits AttributeError.
    agent.tool_result_max_chars = None
    agent.stream = True
    agent.interrupt = threading.Event()
    agent._killed = threading.Event()
    agent._steer = SteerQueue()
    agent._run_lock = threading.Lock()
    agent.messages = []
    agent.children = []
    return agent


def drain(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


# --------------------------------------------------------------------------
# SteerQueue
# --------------------------------------------------------------------------

def test_steer_queue_push_then_drain():
    q = SteerQueue()
    q.push("a")
    q.push("b")
    assert q.drain() == ["a", "b"]


def test_steer_queue_drain_clears():
    q = SteerQueue()
    q.push("a")
    q.drain()
    assert q.drain() == []


# --------------------------------------------------------------------------
# call_llm directly
# --------------------------------------------------------------------------

def test_call_llm_folds_steer_in_at_turn_boundary():
    api = ToolThenTextApi()
    pending = []
    def steer():
        out = list(pending)
        pending.clear()
        return out
    def dispatch(name, args):
        pending.append("also consider Y")      # steer arrives during turn 1's tool
        return "poked"

    messages = [{"role": "user", "content": "do X"}]
    events, text = drain(call_llm(messages,
                                  "m",
                                  api,
                                  tools=[{"type": "function", "function": {"name": "poke"}}],
                                  tools_dispatch=dispatch,
                                  steer=steer))
    assert text == "final"
    # the steered user turn lands after the tool result, before turn 2
    roles = []
    for m in messages:
        roles.append(m["role"])
    assert roles == ["user", "assistant", "tool", "user", "assistant"]
    assert messages[3] == {"role": "user", "content": "also consider Y"}


def test_call_llm_without_steer_is_unaffected():
    class Api:
        def chat(self, messages, model, **kwargs):
            def gen():
                yield ("hi", None, None, {})
            return gen()

    messages = [{"role": "user", "content": "x"}]
    events, text = drain(call_llm(messages, "m", Api()))   # steer=None
    assert text == "hi"
    roles = []
    for m in messages:
        roles.append(m["role"])
    assert roles == ["user", "assistant"]


# --------------------------------------------------------------------------
# Agent.steer() plumbing
# --------------------------------------------------------------------------

def test_agent_steer_from_a_tool_reaches_the_next_turn():
    holder = {}
    def poke() -> str:
        holder["agent"].steer("also consider Y")
        return "poked"

    agent = bare_agent(ToolsRegistry.for_tools([poke]), ToolThenTextApi())
    holder["agent"] = agent
    run = agent.run("do X")
    run.wait()

    assert run.text == "final"
    seen = []
    for m in agent.api.turn2_messages:
        seen.append((m["role"], m.get("content")))
    assert ("user", "also consider Y") in seen
    roles = []
    for m in agent.messages:
        roles.append(m["role"])
    assert roles == ["user", "assistant", "tool", "user", "assistant"]


def test_agent_steer_while_idle_runs_a_turn_at_once():
    # with no run in flight there is nothing to fold into, so a steer behaves like
    # a submit: it runs a turn here and now, rather than waiting for the next run.
    class OneTurnApi:
        def chat(self, messages, model, **kwargs):
            def gen():
                yield ("done", None, None, {})
            return gen()

    agent = bare_agent(ToolsRegistry.for_tools([]), OneTurnApi())
    assert agent.steer("seeded steer") is True   # idle + default run_on_idle
    seen = []
    for m in agent.messages:
        seen.append((m["role"], m.get("content")))
    assert ("user", "seeded steer") in seen
    assert ("assistant", "done") in seen


def test_agent_steer_run_on_idle_false_queues_without_running():
    # an idle agent with run_on_idle=False queues the text and returns False,
    # leaving the caller to drive the turn its own way - whichever run comes
    # next drains the queued text at its first boundary.
    agent = bare_agent(ToolsRegistry.for_tools([]), ToolThenTextApi())
    assert agent.steer("x", run_on_idle=False) is False
    assert agent.messages == []                  # nothing ran
    assert agent.steer_pending() is True
    assert agent._steer.drain() == ["x"]         # queued for the next run


def test_agent_steer_while_running_queues_and_returns_true():
    # a run in flight: steer always queues (run_on_idle is irrelevant) and reports
    # True - the in-flight run folds it in at its next boundary.
    agent = bare_agent(ToolsRegistry.for_tools([]), ToolThenTextApi())
    agent._run_lock.acquire()                    # pretend a run is streaming
    assert agent.steer("later", run_on_idle=False) is True
    assert agent.steer_pending() is False        # the "run" will drain it
    assert agent._steer.drain() == ["later"]


# --------------------------------------------------------------------------
# run serialization (RunInFlight)
# --------------------------------------------------------------------------

def test_concurrent_run_raises_run_in_flight():
    started = threading.Event()
    release = threading.Event()

    class BlockApi:
        def chat(self, messages, model, **kwargs):
            def gen():
                started.set()
                release.wait(2)
                yield ("done", None, None, {})
            return gen()

    agent = bare_agent(ToolsRegistry.for_tools([]), BlockApi())
    first = agent.run("hi")
    thread = threading.Thread(target=first.wait, daemon=True)
    thread.start()
    assert started.wait(2)
    with pytest.raises(RunInFlight):
        agent.run("again").wait()
    release.set()
    thread.join(2)
    assert first.text == "done"


def test_losing_run_leaves_no_user_turn_behind():
    # the prompt is appended under the run lock, at iteration: a run that loses
    # the race raises without polluting the conversation with its user turn.
    started = threading.Event()
    release = threading.Event()

    class BlockApi:
        def chat(self, messages, model, **kwargs):
            def gen():
                started.set()
                release.wait(2)
                yield ("done", None, None, {})
            return gen()

    agent = bare_agent(ToolsRegistry.for_tools([]), BlockApi())
    first = agent.run("winner")
    thread = threading.Thread(target=first.wait, daemon=True)
    thread.start()
    assert started.wait(2)
    with pytest.raises(RunInFlight):
        agent.run("loser").wait()
    release.set()
    thread.join(2)
    contents = []
    for m in agent.messages:
        if m["role"] != "user": continue
        contents.append(m["content"])
    assert contents == ["winner"]


def test_concurrent_idle_steers_deliver_every_text_exactly_once():
    # the TOCTOU hammer: many threads steer an idle agent at once. whichever
    # run wins the lock drains the queue; the losers' texts ride along (or stay
    # queued for the final drain-run). every text lands exactly once.
    class EchoApi:
        def chat(self, messages, model, **kwargs):
            def gen():
                yield ("ok", None, None, {})
            return gen()

    agent = bare_agent(ToolsRegistry.for_tools([]), EchoApi())
    threads = []
    for i in range(8):
        def _steer(n=i):
            agent.steer(f"steer-{n}")
        threads.append(threading.Thread(target=_steer))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    # a text pushed after the winning run's last drain stays queued; one final
    # drain-run delivers the tail.
    if agent.steer_pending():
        agent.run(None).wait()

    contents = []
    for m in agent.messages:
        if m["role"] != "user": continue
        contents.append(m["content"])
    expected = []
    for i in range(8):
        expected.append(f"steer-{i}")
    assert sorted(contents) == sorted(expected)
