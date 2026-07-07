"""Tests for cai.tail - `cai --tail` following a served agent read-only.

Integration-style but fully offline: a real UnixWiredAgent serves a FakeApi
agent over a unix socket in tmp_path, the registry is pointed there, and
tail.run() attaches like any other client. No network, no config, no fzf -
the picker is exercised with a stubbed subprocess."""
import io
import socket
import subprocess
import sys
import threading
import time

import pytest

from test_wired_agent import FakeApi, make_agent, run_turn_over

from cai import channel
from cai import tail
from cai.agents_registry import AgentsRegistry
from cai.events import Event, EventType
from cai.wire import Wire
from cai.wired_agent import UnixWiredAgent


def _registry_at(monkeypatch, tmp_path):
    monkeypatch.setattr(AgentsRegistry, "dir", staticmethod(lambda: str(tmp_path)))


def _wait_for(predicate, message, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail(f"timed out waiting for {message}")


# --------------------------------------------------------------------------
# live_names / completion
# --------------------------------------------------------------------------

def test_live_names_skips_stale_sockets(monkeypatch, tmp_path):
    _registry_at(monkeypatch, tmp_path)
    # a stale socket file: bound once, listener long gone.
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(tmp_path / "dead.sock"))
    stale.close()

    agent = make_agent(api=FakeApi(chunks=["hello"]))
    served = UnixWiredAgent(agent, str(tmp_path / "alive.sock"))
    thread = threading.Thread(target=served.serve, daemon=True)
    thread.start()

    assert tail.live_names() == ["alive"]

    served.close()
    thread.join(timeout=5)


# --------------------------------------------------------------------------
# tailing a live agent
# --------------------------------------------------------------------------

def test_tail_replays_backlog_then_streams_live(monkeypatch, tmp_path):
    _registry_at(monkeypatch, tmp_path)
    # keep the test hermetic: the real settings read imports ~/.config/cai.
    monkeypatch.setattr(tail, "_show_reasoning", lambda: True)
    agent = make_agent(api=FakeApi(chunks=["hello"]))
    served = UnixWiredAgent(agent, str(tmp_path / "a.sock"))
    serve_thread = threading.Thread(target=served.serve, daemon=True)
    serve_thread.start()

    # the owner drives one turn first, so the tail has a backlog to replay.
    owner = channel.connect(str(tmp_path / "a.sock"))
    owner_wire = Wire(owner)
    assert run_turn_over(owner_wire, "hi") == "hello"

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    box = {}
    def _tail():
        box["code"] = tail.run("a")
    tail_thread = threading.Thread(target=_tail, daemon=True)
    tail_thread.start()

    # backlog: the stored user turn and answer, replayed on attach.
    _wait_for(lambda: "> hi" in out.getvalue(), "backlog user turn")
    _wait_for(lambda: "hello" in out.getvalue(), "backlog answer")

    # live: a turn another client submits broadcasts to the tail.
    assert run_turn_over(owner_wire, "again") == "hello"
    _wait_for(lambda: "> again" in out.getvalue(), "live user turn")
    _wait_for(lambda: out.getvalue().count("hello") >= 2, "live answer")

    owner.close()
    served.close()
    tail_thread.join(timeout=5)
    assert not tail_thread.is_alive()
    assert box["code"] == 0
    assert "[a finished]" in out.getvalue()
    serve_thread.join(timeout=5)


def test_tail_unknown_agent_fails(monkeypatch, tmp_path, capsys):
    _registry_at(monkeypatch, tmp_path)
    assert tail.run("ghost") == 1
    assert "no running agent named 'ghost'" in capsys.readouterr().err


def test_bare_tail_without_live_agents_says_so(monkeypatch, tmp_path, capsys):
    _registry_at(monkeypatch, tmp_path)
    assert tail.run("") == 1
    assert "no available running agent" in capsys.readouterr().out


# --------------------------------------------------------------------------
# the fzf picker
# --------------------------------------------------------------------------

def test_pick_returns_the_fzf_choice(monkeypatch):
    seen = {}
    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(cmd, 0, stdout="beta\n")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert tail._pick(["alpha", "beta"]) == "beta"
    assert seen["cmd"][0] == "fzf"
    assert "alpha\nbeta\n" == seen["input"]


def test_pick_cancelled_returns_none(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 130, stdout="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert tail._pick(["alpha"]) is None


def test_pick_without_fzf_lists_names(monkeypatch, capsys):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("fzf")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert tail._pick(["alpha", "beta"]) is None
    err = capsys.readouterr().err
    assert "fzf not found" in err
    assert "alpha" in err
    assert "beta" in err


def test_agent_completer_filters_by_prefix(monkeypatch):
    from cai import cli
    monkeypatch.setattr(tail, "live_names", lambda: ["alpha", "beta", "alps"])
    assert cli._agent_completer("al") == ["alpha", "alps"]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def test_replay_renders_the_stored_conversation():
    out = io.StringIO()
    printer = tail._Printer(out)
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "do it"},
        {"role": "assistant",
         "content": "on it",
         "tool_calls": [{"id": "c1",
                         "type": "function",
                         "function": {"name": "fs__read",
                                      "arguments": "{\"path\": \"x\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "12345"},
        {"role": "assistant", "content": "done"},
    ]
    tail._replay(printer, messages)
    text = out.getvalue()
    assert "rules" not in text                    # system turns are scaffolding
    assert "> do it" in text
    assert "on it" in text
    assert "-> fs__read(path=x)" in text
    assert "<- fs__read: 5 chars" in text
    assert "done" in text


def test_printer_honors_show_reasoning():
    out = io.StringIO()
    printer = tail._Printer(out, show_reasoning=False)
    printer.event(Event(type=EventType.REASONING, text="pondering"))
    printer.event(Event(type=EventType.CONTENT, text="answer"))
    assert "pondering" not in out.getvalue()
    assert "answer" in out.getvalue()


def test_printer_closes_a_midstream_line_before_a_tool_line():
    out = io.StringIO()
    printer = tail._Printer(out)
    printer.event(Event(type=EventType.CONTENT, text="thinking"))
    printer.event(Event(type=EventType.TOOL_CALL, tool_name="t", tool_args={}))
    assert "thinking\n" in out.getvalue()


def test_result_message_separates_turns():
    out = io.StringIO()
    printer = tail._Printer(out)
    tail._print_msg(printer, {"type": Wire.RESULT, "text": "x"})
    assert out.getvalue() == "\n"


def test_prompt_broadcasts_render_but_status_is_skipped():
    out = io.StringIO()
    printer = tail._Printer(out)
    tail._print_msg(printer, {"type": Wire.PROMPT, "kind": "confirm", "message": "sure?"})
    tail._print_msg(printer, {"type": Wire.PROMPT, "kind": "status", "message": "working"})
    text = out.getvalue()
    assert "[confirm] sure?" in text
    assert "working" not in text
