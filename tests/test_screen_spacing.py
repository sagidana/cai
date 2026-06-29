"""Tests for the uniform one-blank-line block separation: the screen's
_block_separator decision over the buffer state, the buffer's ends_blank
probe, and the _Transcript state machine that streams CONTENT/REASONING into
one block while making every other event its own block."""
from cai.screen.screen import Screen
from cai.screen.buffer import ContentBuffer
from cai.tui import _Transcript
from cai.events import EventType


class _Carrier:
    """just enough of a Screen to call the unbound _block_separator: it only
    reads self._buffer."""

    def __init__(self, buffer):
        self._buffer = buffer


def _separator_after(*writes):
    buf = ContentBuffer(80)
    for text in writes:
        buf.append_text(text)
    return Screen._block_separator(_Carrier(buf))


def test_separator_is_empty_on_an_empty_buffer():
    assert _separator_after() == ""


def test_separator_terminates_and_blanks_a_partial_line():
    assert _separator_after("hello") == "\n\n"


def test_separator_adds_one_blank_after_a_finished_line():
    assert _separator_after("hello\n") == "\n"


def test_separator_adds_nothing_when_already_blank():
    assert _separator_after("hello\n\n") == ""


def test_ends_blank_probe():
    buf = ContentBuffer(80)
    assert buf.ends_blank() is False
    buf.append_text("hi")
    assert buf.ends_blank() is False        # partial, mid-line
    buf.append_text("\n")
    assert buf.ends_blank() is False        # finished but not blank
    buf.append_text("\n")
    assert buf.ends_blank() is True         # a real blank line


class _Event:
    def __init__(self, type, text="", tool_name="", tool_args=None,
                 tool_result="", is_error=False):
        self.type = type
        self.text = text
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_result = tool_result
        self.is_error = is_error


class _RecordingScreen:
    def __init__(self):
        self.writes = []                    # (text, block) per write

    def write(self, text, kind=None, block=False):
        self.writes.append((text, block))


def _blocks(screen):
    return [block for _text, block in screen.writes]


def test_consecutive_content_chunks_stream_into_one_block():
    screen = _RecordingScreen()
    t = _Transcript(screen)
    t.event(_Event(EventType.CONTENT, text="hel"))
    t.event(_Event(EventType.CONTENT, text="lo"))
    # first chunk opens the block, the second continues it.
    assert _blocks(screen) == [True, False]


def test_each_non_content_event_is_its_own_block():
    screen = _RecordingScreen()
    t = _Transcript(screen)
    t.event(_Event(EventType.USER, text="hi"))
    t.event(_Event(EventType.CONTENT, text="think"))
    t.event(_Event(EventType.CONTENT, text="ing"))
    t.event(_Event(EventType.TOOL_CALL, tool_name="ls"))
    t.event(_Event(EventType.TOOL_RESULT, tool_name="ls", tool_result="x"))
    t.event(_Event(EventType.CONTENT, text="done"))
    # user / first-content / tool-call / tool-result / resumed-content each
    # start a block; the middle content chunk continues its block.
    assert _blocks(screen) == [True, True, False, True, True, True]


def test_two_tool_calls_in_a_row_are_separate_blocks():
    screen = _RecordingScreen()
    t = _Transcript(screen)
    t.event(_Event(EventType.TOOL_CALL, tool_name="a"))
    t.event(_Event(EventType.TOOL_CALL, tool_name="b"))
    assert _blocks(screen) == [True, True]
