"""Tests for the read-only attach path of the :agents view.

_AttachView is the write(text, kind=, block=) surface the attach overlay
paints from - the same contract Screen offers, so the shared renderers
(tui._replay_messages, tui._Transcript) paint into it unchanged. _attach_agent
mirrors a served agent over a second wire: the stored conversation first, then
the run's EVENT broadcast - without ever driving the agent. The overlay's tty
loop is not under test (it needs a terminal); a fake screen stands in for it
and drives the drain callback directly.
"""
import threading

from test_wired_agent import FakeApi, make_agent, run_turn_over

from cai import channel
from cai.environment import Settings
from cai.screen.ansi import ansi_strip
from cai.screen.screen import Screen
from cai.tui import _AttachView, _attach_agent, _replay_messages
from cai.wire import Wire
from cai.wired_agent import UnixWiredAgent


def _plain_text(view):
    lines = []
    for line in view.get_lines(0, view.line_count()):
        lines.append(ansi_strip(line))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# _AttachView: the paint surface
# --------------------------------------------------------------------------

def test_attach_view_replays_a_conversation():
    view = _AttachView(80)
    messages = []
    messages.append({"role": "user", "content": "hello"})
    messages.append({"role": "assistant", "content": "world"})
    _replay_messages(view, messages, Settings())
    text = _plain_text(view)
    assert "> hello" in text
    assert "world" in text


def test_attach_view_separates_blocks_with_one_blank_row():
    view = _AttachView(80)
    view.write("> hello\n", kind=Screen.USER, block=True)
    view.write("world\n\n\n", kind=Screen.LLM, block=True)
    view.write("again\n", kind=Screen.LLM, block=True)
    lines = []
    for line in view.get_lines(0, view.line_count()):
        lines.append(ansi_strip(line).strip("▌ "))
    assert lines == ["> hello", "", "world", "", "again"]


def test_attach_view_versions_every_write():
    view = _AttachView(80)
    before = view.version
    view.write("chunk", kind=Screen.LLM)
    assert view.version > before


def test_attach_view_rewraps_to_a_new_width():
    view = _AttachView(80)
    view.write("x" * 100 + "\n", kind=Screen.LLM)
    wide = view.line_count()
    view.rewrap(40)
    assert view.line_count() > wide


# --------------------------------------------------------------------------
# _attach_agent: the read-only mirror over the wire
# --------------------------------------------------------------------------

class _OverlayScreen:
    """just enough of a Screen for _attach_agent: the width the view wraps at,
    and a prompt_attach_overlay that - instead of painting a tty - submits one
    more turn through the driver wire and drains the broadcast into the view
    until the streamed answer shows."""

    _cols = 80

    def __init__(self, driver, answer):
        self._driver = driver
        self._answer = answer
        self.view = None

    def prompt_attach_overlay(self, view, *, title, watch=None, drain_fn=None,
                              kill_fn=None):
        self.view = view
        self._driver.send_submit("two")
        while self._answer not in _plain_text(view):
            assert drain_fn() is True


def _patch_registry(tmp_path, monkeypatch):
    from cai.agents_registry import AgentsRegistry

    def _sock_path(name):
        return str(tmp_path / f"{name}.sock")

    monkeypatch.setattr(AgentsRegistry, "sock_path", staticmethod(_sock_path))
    return _sock_path


def test_attach_agent_mirrors_snapshot_then_live_stream(tmp_path, monkeypatch):
    sock_path = _patch_registry(tmp_path, monkeypatch)
    agent = make_agent(api=FakeApi(chunks=["echo"]))
    served = UnixWiredAgent(agent, sock_path(agent.name))
    thread = threading.Thread(target=served.serve, daemon=True)
    thread.start()
    driver = channel.connect(sock_path(agent.name))
    driver_wire = Wire(driver)
    stops = []

    try:
        assert run_turn_over(driver_wire, "one") == "echo"

        node = {}
        node["id"] = agent.name
        node["name"] = agent.name
        screen = _OverlayScreen(driver_wire, "> two")
        assert _attach_agent(screen, node, Settings(), stops.append) is True

        text = _plain_text(screen.view)
        assert "> one" in text          # the snapshot, replayed before streaming
        assert "> two" in text          # the live turn, streamed off the broadcast
        assert stops == []              # never killed, never driven
        # the watcher sent nothing on its wire: both turns belong to the driver.
        ok, messages, _error = driver_wire.control("get_messages")
        assert ok is True
        roles = []
        for message in messages:
            roles.append(message["role"])
        assert roles == ["user", "assistant", "user", "assistant"]
    finally:
        driver.close()
        served.close()
        thread.join(timeout=5)


def test_attach_agent_falls_back_when_the_socket_is_gone(tmp_path, monkeypatch):
    _patch_registry(tmp_path, monkeypatch)
    node = {}
    node["id"] = "ghost"
    node["name"] = "ghost"
    assert _attach_agent(None, node, Settings(), None) is False
