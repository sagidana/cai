"""Tests for Agent.gate - the single-turn quality gate built on
run(strict_format=...). Fully offline: a fake non-streaming api returns canned
answers, so the regex constraint, the retry-until-an-option behaviour, and the
isolation from the agent's own history can be checked without a model."""
from cai.agent import Agent


# ---------------------------------------------------------------------------
# fakes / helpers
# ---------------------------------------------------------------------------

class GateApi:
    """non-streaming fake: each chat() pops the next canned answer and returns it
    as a (content, reasoning, tool_calls, usage) tuple (gate forces stream off).
    records the messages each call saw."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.seen = []

    def chat(self, messages, model, **kwargs):
        self.seen.append(list(messages))
        content = self.answers.pop(0)
        return content, "", None, {}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_gate_returns_the_chosen_option():
    api = GateApi(["yes"])
    agent = Agent(model="m", api=api)
    result = agent.gate(["yes", "no"], "Is the sky blue?")
    assert result == "yes"


def test_gate_retries_until_a_valid_option():
    api = GateApi(["maybe", "no"])   # the first reply is not one of the options
    agent = Agent(model="m", api=api)
    result = agent.gate(["yes", "no"], "?")
    assert result == "no"
    assert len(api.seen) == 2


def test_gate_does_not_touch_the_agents_history():
    api = GateApi(["yes"])
    agent = Agent(model="m", api=api)
    agent.gate(["yes", "no"], "?")
    assert agent.messages == []


def test_gate_persona_lists_the_options():
    api = GateApi(["yes"])
    agent = Agent(model="m", api=api)
    agent.gate(["yes", "no"], "?")
    system = api.seen[0][0]
    assert system["role"] == "system"
    assert "yes" in system["content"]
    assert "no" in system["content"]


def test_gate_custom_system_prompt_overrides_persona():
    api = GateApi(["a"])
    agent = Agent(model="m", api=api)
    agent.gate(["a", "b"], "pick", system_prompt="be the judge")
    system = api.seen[0][0]
    assert system["content"].startswith("be the judge")
