"""Tests for the model-facing tool dispatch guard - cai.agent._selected_dispatch.

Selection is an enforcement boundary on the run loop, not just visibility: a
model that names a registered-but-unselected tool (a wrap target, a deselected
tool, the always-registered sub-agent tools) must be refused, while
ToolsRegistry.dispatch stays the programmatic run-anything API.

Fully offline: a FakeApi emits canned tool calls. No network, no config.
"""
import threading

from cai.agent import Agent, _selected_dispatch
from cai.environment import Environment
from cai.events import EventType
from cai.llm import SteerQueue
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


class CallsThenTextApi:
    """turn 1 asks for the given tool calls, turn 2 answers with text."""

    def __init__(self, calls):
        self.calls = calls
        self.turns = 0

    def chat(self, messages, model, **kwargs):
        self.turns += 1
        n = self.turns
        calls = self.calls
        def gen():
            if n == 1:
                yield (None, None, calls, {})
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


def tool_results(events):
    results = {}
    for event in events:
        if event.type != EventType.TOOL_RESULT: continue
        results[event.tool_name] = event.tool_result
    return results


# --------------------------------------------------------------------------
# _selected_dispatch (unit)
# --------------------------------------------------------------------------

def test_selected_tool_dispatches():
    def ping() -> str:
        return "pong"
    registry = ToolsRegistry.for_tools([ping])
    dispatch = _selected_dispatch(registry)
    assert dispatch("ping", {}) == "pong"


def test_unselected_tool_is_refused_but_stays_dispatchable():
    ran = []
    def hidden() -> str:
        ran.append(True)
        return "secret"
    registry = ToolsRegistry()
    registry.register(hidden)
    dispatch = _selected_dispatch(registry)
    assert dispatch("hidden", {}) == "Error: unknown tool 'hidden'"
    assert ran == []
    # the programmatic API is untouched: registered means runnable.
    assert registry.dispatch("hidden", {}) == "secret"


def test_deselected_tool_is_refused():
    def ping() -> str:
        return "pong"
    registry = ToolsRegistry.for_tools([ping])
    registry.deselect("ping")
    dispatch = _selected_dispatch(registry)
    assert dispatch("ping", {}) == "Error: unknown tool 'ping'"


def test_selection_is_checked_per_call():
    def ping() -> str:
        return "pong"
    registry = ToolsRegistry()
    registry.register(ping)
    dispatch = _selected_dispatch(registry)
    assert dispatch("ping", {}).startswith("Error:")
    registry.select("ping")
    assert dispatch("ping", {}) == "pong"


# --------------------------------------------------------------------------
# through the run loop
# --------------------------------------------------------------------------

def test_run_refuses_unselected_tool_and_runs_selected_one():
    ran = []
    def visible() -> str:
        return "seen"
    def hidden() -> str:
        ran.append(True)
        return "secret"
    registry = ToolsRegistry.for_tools([visible])
    registry.register(hidden)
    calls = [tool_call("visible", call_id="c1"), tool_call("hidden", call_id="c2")]
    agent = bare_agent(registry, CallsThenTextApi(calls))
    events, _text = drain(agent._stream("go"))
    results = tool_results(events)
    assert results["visible"] == "seen"
    assert results["hidden"] == "Error: unknown tool 'hidden'"
    assert ran == []


def test_run_still_dispatches_wrap_targets_through_their_wrapper():
    # the wrapper is selected, its target only registered: the model calls the
    # wrapper, the wrapper's injected dispatch reaches the target - while a
    # direct model call to the target is refused.
    def shout(text: str) -> str:
        return text.upper()
    def polite(call, text: str) -> str:
        return call(text=text) + ", please"
    polite._cai_wrap_target = "shout"
    registry = ToolsRegistry()
    registry.register(shout)
    registry.select(polite)
    calls = [tool_call("polite", call_id="c1", arguments='{"text": "go"}'),
             tool_call("shout", call_id="c2", arguments='{"text": "no"}')]
    agent = bare_agent(registry, CallsThenTextApi(calls))
    events, _text = drain(agent._stream("go"))
    results = tool_results(events)
    assert results["polite"] == "GO, please"
    assert results["shout"] == "Error: unknown tool 'shout'"
