"""Tests for cai.strict - the per-run answer-format enforcement layered above
call_llm. Fully offline: a fake make_stream stands in for a call_llm run,
appending the model's answer to `messages` (as call_llm does) and returning its
text, so the retry loop, scaffold stripping, and the json / regex / regex-each-
line checkers can be driven without a model.

Mirrors the legacy cai test_format.py, retargeted from the old call_fn-returns-a-
tuple shape onto the new generator-returns-text shape."""
import json
import threading

import pytest

from cai.events import Event, EventType
from cai.llm import LLMError
from cai.strict import enforce_strict_format


# ---------------------------------------------------------------------------
# fakes / helpers
# ---------------------------------------------------------------------------

def streamer(messages, answers, seen_system=None, seen_messages=None):
    """build a fake make_stream(system_prompt) -> generator. each call pops the
    next canned answer, optionally records the system prompt and the messages the
    'model' saw, appends an assistant message (as call_llm does), and returns the
    answer text as the generator's value."""
    remaining = list(answers)

    def make_stream(system_prompt):
        make_stream.count += 1
        if seen_system is not None:
            seen_system.append(system_prompt)
        text = remaining.pop(0)

        def gen():
            if seen_messages is not None:
                snapshot = []
                for m in messages:
                    snapshot.append(dict(m))
                seen_messages.append(snapshot)
            if text:
                yield Event(type=EventType.CONTENT, text=text)
            assistant = {}
            assistant["role"] = "assistant"
            assistant["content"] = text
            messages.append(assistant)
            return text

        return gen()

    make_stream.count = 0
    return make_stream


def drain(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


def user(content):
    return {"role": "user", "content": content}


def assistant(content):
    return {"role": "assistant", "content": content}


# ---------------------------------------------------------------------------
# json
# ---------------------------------------------------------------------------

def test_json_first_try_normalises():
    messages = [user("give me json")]
    make_stream = streamer(messages, ['  {"a":  1}  '])
    _events, text = drain(enforce_strict_format(make_stream, "json", None, messages))
    assert text == '{"a": 1}'
    assert make_stream.count == 1
    # only the conversation plus the final answer survive - no scaffold.
    assert messages == [user("give me json"), assistant('{"a": 1}')]


def test_json_retries_with_feedback_then_succeeds():
    messages = [user("q")]
    seen = []
    make_stream = streamer(messages, ["oops", '{"a": 1}'], seen_messages=seen)
    _events, text = drain(enforce_strict_format(make_stream, "json", None, messages))
    assert json.loads(text) == {"a": 1}
    assert make_stream.count == 2
    # the second attempt saw feedback about the invalid JSON.
    blob = ""
    for m in seen[1]:
        blob = blob + " " + m["content"]
    assert "was not valid JSON" in blob
    # the retry scaffolding is gone afterwards.
    assert messages == [user("q"), assistant('{"a": 1}')]


def test_json_gives_up_after_max_attempts():
    messages = [user("q")]
    make_stream = streamer(messages, ["nope", "nope", "nope"])
    with pytest.raises(LLMError, match="did not match required format"):
        drain(enforce_strict_format(make_stream, "json", None, messages, max_attempts=3))
    assert make_stream.count == 3
    # give-up keeps nothing: the conversation is left as it started.
    assert messages == [user("q")]


def test_no_strict_format_is_passthrough():
    messages = [user("q")]
    make_stream = streamer(messages, ["anything"])
    _events, text = drain(enforce_strict_format(make_stream, None, None, messages))
    assert text == "anything"
    assert make_stream.count == 1
    # passthrough is the bare make_stream run - the answer it appended stays.
    assert messages == [user("q"), assistant("anything")]


def test_only_winning_attempt_events_are_yielded():
    messages = [user("q")]
    make_stream = streamer(messages, ["oops", '{"a": 1}'])
    events, _text = drain(enforce_strict_format(make_stream, "json", None, messages))
    contents = []
    for e in events:
        if e.type == EventType.CONTENT:
            contents.append(e.text)
    # the discarded 'oops' attempt never reached the consumer.
    assert contents == ['{"a": 1}']


# ---------------------------------------------------------------------------
# system prompt augmentation / tail reminder
# ---------------------------------------------------------------------------

def test_guidance_appended_to_system_prompt():
    messages = [user("q")]
    seen_system = []
    make_stream = streamer(messages, ['{"a": 1}'], seen_system=seen_system)
    drain(enforce_strict_format(make_stream, "json", "be terse", messages))
    # the base prompt is kept, with the format guidance appended after it.
    assert seen_system[0].startswith("be terse")
    assert "JSON" in seen_system[0]


def test_tail_reminder_present_during_call():
    messages = [user("q")]
    seen = []
    make_stream = streamer(messages, ['{"a": 1}'], seen_messages=seen)
    drain(enforce_strict_format(make_stream, "json", None, messages))
    during = seen[0]
    assert during[-1]["role"] == "user"
    assert "JSON" in during[-1]["content"]


# ---------------------------------------------------------------------------
# regex / regex-each-line
# ---------------------------------------------------------------------------

def test_regex_pass():
    messages = [user("q")]
    make_stream = streamer(messages, ["answer: 42"])
    _events, text = drain(enforce_strict_format(make_stream, r"regex:^answer: \d+$", None, messages))
    assert text == "answer: 42"


def test_regex_retry_then_pass():
    messages = [user("q")]
    make_stream = streamer(messages, ["no digits", "answer: 42"])
    _events, text = drain(enforce_strict_format(make_stream, r"regex:\d+", None, messages))
    assert text == "answer: 42"
    assert make_stream.count == 2


def test_regex_each_line_pass():
    messages = [user("q")]
    make_stream = streamer(messages, ["1: a\n2: b"])
    _events, text = drain(enforce_strict_format(make_stream, r"regex-each-line:^\d+: ", None, messages))
    assert text == "1: a\n2: b"


def test_regex_each_line_one_bad_line_fails_then_passes():
    messages = [user("q")]
    make_stream = streamer(messages, ["1: a\nbad line", "1: a\n2: b"])
    _events, text = drain(enforce_strict_format(make_stream, r"regex-each-line:^\d+: ", None, messages))
    assert text == "1: a\n2: b"
    assert make_stream.count == 2


# ---------------------------------------------------------------------------
# interrupt
# ---------------------------------------------------------------------------

def test_interrupt_stops_retrying_and_keeps_partial():
    messages = [user("q")]
    interrupt = threading.Event()
    interrupt.set()
    make_stream = streamer(messages, ["oops"])
    _events, text = drain(enforce_strict_format(make_stream, "json", None, messages, interrupt=interrupt))
    # a kill mid-run keeps the partial answer instead of burning more attempts.
    assert text == "oops"
    assert make_stream.count == 1
    assert messages == [user("q"), assistant("oops")]
