"""Tests for tui.ScreenUI - the TUI client's UI that renders a served hook's
prompts. The overlays themselves need a tty; here we cover the request the UI
builds, how it maps the result back, and the one-way notify/status routing."""
from cai.environment import Settings
from cai.tui import ScreenUI, _Transcript
from cai.screen import Screen


class FakeScreen:
    def __init__(self, result=None):
        self.requests = []
        self.writes = []
        self._result = result

    def submit_request(self, request):
        self.requests.append(request)
        return self._result

    def write(self, text, kind=None, block=False):
        self.writes.append((text, kind))


def _make_ui(screen, status=None):
    if status is None:
        status = FakeStatus()
    return ScreenUI(screen, status, _Transcript(screen, Settings()))


class FakeStatus:
    def __init__(self):
        self.notes = []

    def set_note(self, message):
        self.notes.append(message)


def test_confirm_builds_request_and_returns_bool():
    screen = FakeScreen(result=True)
    ui = _make_ui(screen)
    assert ui.confirm("ok?", detail="rm -rf /") is True
    request = screen.requests[0]
    assert request["kind"] == "confirm"
    assert request["title"] == "ok?"
    assert request["body"] == "rm -rf /"


def test_confirm_falls_back_to_default_when_screen_closed():
    ui = _make_ui(FakeScreen(result=None))
    assert ui.confirm("ok?", default=True) is True
    assert ui.confirm("ok?", default=False) is False


def test_select_returns_choice_then_default_by_index():
    screen = FakeScreen(result="b")
    ui = _make_ui(screen)
    assert ui.select("pick", ["a", "b", "c"]) == "b"
    assert screen.requests[0]["kind"] == "select"
    assert screen.requests[0]["options"] == ["a", "b", "c"]

    closed = _make_ui(FakeScreen(result=None))
    assert closed.select("pick", ["a", "b"], default=1) == "b"


def test_text_passes_result_through_and_cancels_to_none():
    screen = FakeScreen(result="typed")
    ui = _make_ui(screen)
    assert ui.text("name?", default="d", secret=True) == "typed"
    assert screen.requests[0] == {"kind": "text", "title": "name?",
                                  "default": "d", "secret": True}

    cancelled = _make_ui(FakeScreen(result=None))
    assert cancelled.text("name?") is None


def test_notify_writes_a_transcript_line_by_level():
    screen = FakeScreen()
    ui = _make_ui(screen)
    ui.notify("hi")
    ui.notify("boom", level="error")
    assert screen.writes[0] == ("[info] hi\n", Screen.META)
    assert screen.writes[1] == ("[error] boom\n", Screen.ERROR)


def test_status_sets_the_status_line_note():
    status = FakeStatus()
    _make_ui(FakeScreen(), status).status("compacting…")
    assert status.notes == ["compacting…"]
