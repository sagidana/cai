"""Tests for cai.lines - the `cai --line-by-line` scheduler.

Pure scheduling mechanics: the runs are fakes carrying only the contract
lines.run relies on (iterable events, .text, an interrupt Event, close()),
so ordering, parallelism, error isolation and teardown are all exercised
without an LLM."""
import threading

import pytest

from cai import lines
from cai.api import ApiError
from cai.events import Event, EventType


class FakeRun:
    """one line's fake run: yields `events` (default none), answers `answer`.
    `gate` (an Event) holds __iter__ until set, to force out-of-order
    completion; `exc` raises from the iteration, exercising the error path."""

    def __init__(self, answer, gate=None, exc=None, on_close=None, events=None):
        self.answer = answer
        self.gate = gate
        self.exc = exc
        self.on_close = on_close
        self.events = events or []
        self.text = answer
        self.interrupt = threading.Event()
        self.closed = False

    def __iter__(self):
        if self.gate is not None:
            assert self.gate.wait(5)
        if self.exc is not None:
            raise self.exc
        return iter(self.events)

    def close(self):
        self.closed = True
        if self.on_close is not None:
            self.on_close()


def _factory(runs_by_line, made):
    def make_run(line):
        run = runs_by_line[line]
        made.append(run)
        return run
    return make_run


def test_sequential_answers_in_input_order(capsys):
    runs = {}
    runs["one"] = FakeRun("first")
    runs["two"] = FakeRun("second")
    made = []

    code = lines.run(_factory(runs, made), ["one", "two"], cores=1)

    assert code == 0
    assert capsys.readouterr().out == "one\tfirst\ntwo\tsecond\n"
    assert len(made) == 2
    assert runs["one"].closed
    assert runs["two"].closed


def test_parallel_output_stays_in_input_order(capsys):
    # line 0 is held until line 1 has already finished; the reorder buffer
    # must still print them in input order.
    gate = threading.Event()
    runs = {}
    runs["slow"] = FakeRun("slow-answer", gate=gate)
    runs["fast"] = FakeRun("fast-answer")
    made = []
    releaser = threading.Timer(0.2, gate.set)
    releaser.start()

    code = lines.run(_factory(runs, made), ["slow", "fast"], cores=2)

    releaser.join()
    assert code == 0
    assert capsys.readouterr().out == "slow\tslow-answer\nfast\tfast-answer\n"


def test_a_failing_line_reports_and_the_rest_still_run(capsys):
    runs = {}
    runs["bad"] = FakeRun(None, exc=ApiError("boom"))
    runs["good"] = FakeRun("fine")
    made = []

    code = lines.run(_factory(runs, made), ["bad", "good"], cores=1)

    assert code == 1
    out = capsys.readouterr().out
    assert out == "bad\tError: boom\ngood\tfine\n"    # the failed line still holds its slot
    assert runs["bad"].closed
    assert runs["good"].closed


def test_an_unexpected_exception_is_contained(capsys):
    runs = {}
    runs["bad"] = FakeRun(None, exc=RuntimeError("surprise"))
    runs["good"] = FakeRun("fine")
    made = []

    code = lines.run(_factory(runs, made), ["bad", "good"], cores=2)

    assert code == 1
    out = capsys.readouterr().out
    assert "Error: RuntimeError: surprise" in out
    assert "fine" in out


def test_make_run_failure_becomes_a_line_error(capsys):
    def make_run(line):
        raise OSError("no such tool")

    code = lines.run(make_run, ["x"], cores=1)

    assert code == 1
    assert capsys.readouterr().out == "x\tError: OSError: no such tool\n"


def test_empty_input_is_a_clean_noop(capsys):
    code = lines.run(lambda line: pytest.fail("must not spawn"), [], cores=4)
    assert code == 0
    assert capsys.readouterr().out == ""


def test_lines_run_as_they_arrive_not_at_eof(capsys):
    # the source holds its second line back until the first line's run has
    # finished: work must start on line one while the input is still open.
    first_closed = threading.Event()
    runs = {}
    runs["a"] = FakeRun("A", on_close=first_closed.set)
    runs["b"] = FakeRun("B")
    made = []

    def source():
        yield "a"
        assert first_closed.wait(5), "first line never ran while input was open"
        yield "b"

    code = lines.run(_factory(runs, made), source(), cores=1)

    assert code == 0
    assert capsys.readouterr().out == "a\tA\nb\tB\n"


def test_trace_carries_tool_calls_to_stderr_grouped(monkeypatch, capsys):
    notes = []
    monkeypatch.setattr(lines, "_note", notes.append)
    events = [Event(type=EventType.TOOL_CALL, tool_name="fs__read", tool_args={"path": "x"}),
              Event(type=EventType.TOOL_RESULT, tool_name="fs__read", tool_result="12345"),
              Event(type=EventType.CONTENT, text="the answer")]
    runs = {}
    runs["a"] = FakeRun("the answer", events=events)
    made = []

    code = lines.run(_factory(runs, made), ["a"], cores=1)

    assert code == 0
    trace = "\n".join(notes)
    assert "-> fs__read(path=x)" in trace
    assert "<- fs__read: 5 chars" in trace
    assert "the answer" not in trace              # content IS the answer, not trace
    assert capsys.readouterr().out == "a\tthe answer\n"


def test_trace_reasoning_follows_the_settings(monkeypatch):
    events = [Event(type=EventType.REASONING, text="pondering"),
              Event(type=EventType.TOOL_CALL, tool_name="t", tool_args={})]

    notes = []
    monkeypatch.setattr(lines, "_note", notes.append)
    runs = {}
    runs["a"] = FakeRun("x", events=events)
    lines.run(_factory(runs, []), ["a"], cores=1, show_reasoning=True)
    assert "pondering" in "\n".join(notes)

    notes.clear()
    runs["a"] = FakeRun("x", events=events)
    lines.run(_factory(runs, []), ["a"], cores=1, show_reasoning=False)
    assert "pondering" not in "\n".join(notes)


def test_multiline_answers_stay_newline_terminated(capsys):
    runs = {}
    runs["a"] = FakeRun("two\nlines")
    runs["b"] = FakeRun("tail-newline\n")
    made = []

    code = lines.run(_factory(runs, made), ["a", "b"], cores=1)

    assert code == 0
    assert capsys.readouterr().out == "a\ttwo\nlines\nb\ttail-newline\n"
