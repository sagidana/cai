"""Tests for the cooperative interrupt - cai.llm.call_llm winding down on a set
threading.Event, and Agent.stop() / Run plumbing it through.

Fully offline: a FakeApi streams canned chunks (and can flip the interrupt
itself, to simulate a kill landing mid-stream). No network, no config.
"""
import threading

from cai.agent import Agent
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


class CountApi:
    """streams `chunks` as one turn's content; counts how many turns were
    actually requested."""

    def __init__(self, chunks=None):
        self.calls = 0
        self.chunks = chunks
        if self.chunks is None:
            self.chunks = ["hello"]

    def chat(self, messages, model, **kwargs):
        self.calls += 1
        chunks = self.chunks
        def gen():
            for chunk in chunks:
                yield (chunk, None, None, {})
        return gen()


class ToolThenTextApi:
    """turn 1 asks for a tool, turn 2 answers with text - so a kill during the
    tool can be seen to stop the second turn."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, model, **kwargs):
        self.calls += 1
        n = self.calls
        def gen():
            if n == 1:
                yield (None, None, [tool_call("killer")], {})
            else:
                yield ("SECOND TURN", None, None, {})
        return gen()


def bare_agent(tools_registry, api):
    """an Agent without config/network (bypass __init__)."""
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
    agent._running = threading.Event()
    agent.messages = []
    agent.children = []
    return agent


def drain(gen):
    """run a call_llm generator to completion, returning (events, final_text)."""
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


# --------------------------------------------------------------------------
# call_llm directly
# --------------------------------------------------------------------------

def test_interrupt_set_before_start_skips_the_model_call():
    api = CountApi(chunks=["hello"])
    interrupt = threading.Event()
    interrupt.set()
    events, text = drain(call_llm([], "m", api, interrupt=interrupt))
    assert text == ""
    assert api.calls == 0                       # killed before the first turn


def test_interrupt_mid_stream_returns_partial():
    interrupt = threading.Event()

    class FlipApi:
        def chat(self, messages, model, **kwargs):
            def gen():
                yield ("partial ", None, None, {})
                interrupt.set()                 # kill lands between chunks
                yield ("DROPPED", None, None, {})
            return gen()

    messages = []
    events, text = drain(call_llm(messages, "m", FlipApi(), interrupt=interrupt))
    assert text == "partial "                   # second chunk never folded in
    assert messages == []                       # no final assistant message appended


def test_uninterrupted_call_llm_is_unaffected():
    api = CountApi(chunks=["hello ", "world"])
    events, text = drain(call_llm([], "m", api, interrupt=threading.Event()))
    assert text == "hello world"
    assert api.calls == 1


def test_no_interrupt_event_runs_normally():
    api = CountApi(chunks=["hi"])
    events, text = drain(call_llm([], "m", api))   # interrupt=None
    assert text == "hi"


# --------------------------------------------------------------------------
# Agent.stop() / Run plumbing
# --------------------------------------------------------------------------

def test_stop_before_iterating_returns_empty():
    agent = bare_agent(ToolsRegistry.for_tools([]), CountApi(chunks=["hello"]))
    run = agent.run("hi")
    agent.stop()
    run.wait()
    assert run.text == ""
    assert agent.api.calls == 0


def test_stop_from_a_tool_halts_further_turns():
    holder = {}
    def killer() -> str:
        holder["agent"].stop()
        return "killed from inside the tool"

    agent = bare_agent(ToolsRegistry.for_tools([killer]), ToolThenTextApi())
    holder["agent"] = agent
    run = agent.run("go")
    run.wait()
    assert agent.api.calls == 1                  # the second model turn never ran
    assert run.text == ""                        # partial: no content produced
    assert agent.messages[-1]["role"] == "tool"  # transcript ends on the tool result


def test_run_clears_a_stale_interrupt():
    agent = bare_agent(ToolsRegistry.for_tools([]), CountApi(chunks=["hello"]))
    agent.stop()                                 # left set from a prior (killed) run
    run = agent.run("hi")                        # run() should clear it
    run.wait()
    assert run.text == "hello"
    assert agent.api.calls == 1


# --------------------------------------------------------------------------
# Agent.kill() - the permanent stop
# --------------------------------------------------------------------------

def test_kill_from_a_tool_halts_run_and_retires_agent():
    holder = {}
    def killer() -> str:
        holder["agent"].kill()
        return "killed from inside the tool"

    agent = bare_agent(ToolsRegistry.for_tools([killer]), ToolThenTextApi())
    holder["agent"] = agent
    run = agent.run("go")
    run.wait()
    assert agent.api.calls == 1                  # the second model turn never ran
    assert run.text == ""
    assert agent.killed is True


def test_killed_agent_refuses_further_runs():
    agent = bare_agent(ToolsRegistry.for_tools([]), CountApi(chunks=["hello"]))
    agent.kill()
    run = agent.run("hi")                        # killed -> aborts at once
    run.wait()
    assert run.text == ""
    assert agent.api.calls == 0
