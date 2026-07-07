"""Tests for cai.cli helpers - the headless run driver and flag validation."""
import threading

import pytest

from cai import cli
from cai.events import Event, EventType


class FakeDriveRun:
    """the slice of a Run that _drive consumes: iterable Events, then text."""

    def __init__(self, events, text="answer"):
        self._events = events
        self.text = text
        self.stream = True
        self.interrupt = threading.Event()

    def __iter__(self):
        return iter(self._events)


def test_drive_streams_reasoning_by_default(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_STDOUT_TTY", True)
    events = [Event(type=EventType.REASONING, text="pondering"),
              Event(type=EventType.CONTENT, text="answer")]
    assert cli._drive(FakeDriveRun(events)) == 0
    out = capsys.readouterr().out
    assert "pondering" in out
    assert "answer" in out


def test_drive_respects_show_reasoning_off(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_STDOUT_TTY", True)
    events = [Event(type=EventType.REASONING, text="pondering"),
              Event(type=EventType.CONTENT, text="answer")]
    assert cli._drive(FakeDriveRun(events), show_reasoning=False) == 0
    out = capsys.readouterr().out
    assert "pondering" not in out
    assert "answer" in out


def test_line_by_line_needs_a_prompt():
    with pytest.raises(SystemExit):
        cli.main(["--line-by-line"])


def test_line_by_line_rejects_watch():
    with pytest.raises(SystemExit):
        cli.main(["--line-by-line", "--watch", "-p", "x"])


def test_line_by_line_rejects_interactive():
    with pytest.raises(SystemExit):
        cli.main(["--line-by-line", "-i", "-p", "x"])


def test_cores_must_be_positive():
    with pytest.raises(SystemExit):
        cli.main(["--line-by-line", "-p", "x", "--cores", "0"])
