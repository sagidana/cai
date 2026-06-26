"""Tests for cooperative steering - cai.agent.SteerQueue, cai.llm.call_llm
folding pending messages in at turn boundaries, and Agent.steer() plumbing.

Fully offline: a FakeApi streams canned chunks and records what it saw on a
given turn, so the injected user turn can be observed. No network, no config.
"""
import threading

from cai.agent import Agent
from cai.llm import call_llm, SteerQueue
from cai.skills import SkillsRegistry
from cai.tools import ToolRegistry


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
    agent._system_prompt = None
    agent.tools_registry = tools_registry
    agent.skills_registry = SkillsRegistry.for_skills([], tools_registry=tools_registry)
    agent._hooks = None
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

    agent = bare_agent(ToolRegistry.for_tools([poke]), ToolThenTextApi())
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


def test_agent_steer_queued_before_run_is_picked_up():
    class OneTurnApi:
        def __init__(self):
            self.turn1_messages = None
        def chat(self, messages, model, **kwargs):
            self.turn1_messages = list(messages)
            def gen():
                yield ("done", None, None, {})
            return gen()

    agent = bare_agent(ToolRegistry.for_tools([]), OneTurnApi())
    agent.steer("seeded steer")                 # pushed while idle
    run = agent.run("the prompt")
    run.wait()
    # first turn sees the prompt then the steer, both as user turns
    seen = []
    for m in agent.api.turn1_messages:
        seen.append((m["role"], m.get("content")))
    assert ("user", "the prompt") in seen
    assert ("user", "seeded steer") in seen
