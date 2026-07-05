"""Tests for deterministic block separation and the message gutter frame:
Screen._start_block trims whatever trailing blank rows the previous block
streamed and lays down exactly one blank row; _Transcript strips a block's
leading newlines and chains a tool call with its result into one block; the
buffer re-applies a segment's gutter to every wrapped line and strips it
back out of selections."""
from cai.screen.screen import Screen
from cai.screen.buffer import ContentBuffer, GUTTER_GLYPH
from cai.screen.state import TUIState
from cai.screen.ansi import ansi_strip
from cai.environment import Settings
from cai.tui import _Transcript
from cai.events import EventType


class _Carrier:
    """just enough of a Screen to call the unbound _start_block: it reads
    self._buffer and clamps self._state."""

    def __init__(self, buffer):
        self._buffer = buffer
        self._state = TUIState()


def _blank_rows_before_block(*writes):
    """append writes, start a block, append its first line; return how many
    blank display rows separate the block from the previous content."""
    carrier = _Carrier(ContentBuffer(80))
    for text in writes:
        carrier._buffer.append_text(text)
    Screen._start_block(carrier)
    carrier._buffer.append_text("-> next\n")
    lines = carrier._buffer._lines
    blanks = 0
    idx = len(lines) - 2
    while idx >= 0:
        plain = ansi_strip(lines[idx]).strip()
        if plain != '' and plain != GUTTER_GLYPH: break
        blanks += 1
        idx -= 1
    return blanks


def test_block_on_empty_buffer_adds_no_gap():
    assert _blank_rows_before_block() == 0


def test_gap_is_one_after_a_partial_line():
    assert _blank_rows_before_block("hello") == 1


def test_gap_is_one_after_a_finished_line():
    assert _blank_rows_before_block("hello\n") == 1


def test_gap_is_one_no_matter_how_many_trailing_newlines():
    # the historic bug: trailing newlines streamed by the model made the
    # gap 1 + extra. now every ending collapses to exactly one blank row.
    for ending in ("\n\n", "\n\n\n", "\n\n\n\n\n"):
        assert _blank_rows_before_block("hello" + ending) == 1


def test_gap_is_one_after_trailing_newline_only_chunks():
    # trailing newlines arriving as their own streamed chunks, not attached
    # to the text chunk.
    assert _blank_rows_before_block("hello", "\n", "\n\n") == 1


def test_ends_blank_probe():
    buf = ContentBuffer(80)
    assert buf.ends_blank() is False
    buf.append_text("hi")
    assert buf.ends_blank() is False        # partial, mid-line
    buf.append_text("\n")
    assert buf.ends_blank() is False        # finished but not blank
    buf.append_text("\n")
    assert buf.ends_blank() is True         # a real blank line


def test_trim_keeps_internal_blanks():
    # paragraph breaks inside a block survive; only trailing blanks go.
    buf = ContentBuffer(80)
    buf.append_text("para one\n\npara two\n\n\n")
    buf.trim_trailing_blanks()
    plains = [ansi_strip(l) for l in buf._lines]
    assert plains == ["para one", "", "para two"]


def test_trim_survives_rewrap():
    # raw segments are trimmed too, so a resize reproduces the trimmed state.
    buf = ContentBuffer(80)
    buf.append_text("hello\n\n\n")
    buf.trim_trailing_blanks()
    buf.rewrap(40)
    assert [ansi_strip(l) for l in buf._lines] == ["hello"]


# --- gutter frame ---

_GUTTER = f'{GUTTER_GLYPH} '


def test_gutter_prefixes_every_wrapped_line():
    buf = ContentBuffer(12)
    buf.append_text("a" * 25 + "\n", gutter=_GUTTER)
    for line in buf._lines:
        assert ansi_strip(line).startswith(_GUTTER)
    # wrap width shrank by the gutter width
    assert len(ansi_strip(buf._lines[0])) == 12


def test_gutter_survives_rewrap():
    buf = ContentBuffer(80)
    buf.append_text("some text\n", gutter=_GUTTER)
    buf.rewrap(40)
    assert ansi_strip(buf._lines[0]) == f"{_GUTTER}some text"


def test_trim_drops_gutter_only_blank_rows():
    buf = ContentBuffer(80)
    buf.append_text("hello\n\n\n", gutter=_GUTTER)
    buf.trim_trailing_blanks()
    assert [ansi_strip(l) for l in buf._lines] == [f"{_GUTTER}hello"]


def test_gutter_change_mid_line_breaks_the_line():
    buf = ContentBuffer(80)
    buf.append_text("first", gutter=_GUTTER)
    buf.append_text("second\n", gutter='')
    assert ansi_strip(buf._lines[0]) == f"{_GUTTER}first"
    assert ansi_strip(buf._lines[1]) == "second"


def test_selection_strips_the_gutter():
    buf = ContentBuffer(80)
    buf.append_text("copy me\n", gutter=_GUTTER)
    assert buf.get_selection_text(0, 0, 0, 79, True) == "copy me"


# --- transcript ---

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
    t = _Transcript(screen, Settings())
    t.event(_Event(EventType.CONTENT, text="hel"))
    t.event(_Event(EventType.CONTENT, text="lo"))
    # first chunk opens the block, the second continues it.
    assert _blocks(screen) == [True, False]


def test_unit_changes_start_blocks_and_tool_pairs_chain():
    screen = _RecordingScreen()
    t = _Transcript(screen, Settings())
    t.event(_Event(EventType.USER, text="hi"))
    t.event(_Event(EventType.CONTENT, text="think"))
    t.event(_Event(EventType.CONTENT, text="ing"))
    t.event(_Event(EventType.TOOL_CALL, tool_name="ls"))
    t.event(_Event(EventType.TOOL_RESULT, tool_name="ls", tool_result="x"))
    t.event(_Event(EventType.CONTENT, text="done"))
    # user / first-content / tool-call / resumed-content each start a block;
    # the middle content chunk and the tool result continue theirs.
    assert _blocks(screen) == [True, True, False, True, False, True]


def test_consecutive_tool_exchanges_share_one_block():
    screen = _RecordingScreen()
    t = _Transcript(screen, Settings())
    t.event(_Event(EventType.TOOL_CALL, tool_name="a"))
    t.event(_Event(EventType.TOOL_CALL, tool_name="b"))
    assert _blocks(screen) == [True, False]


def test_leading_newlines_of_a_block_are_stripped():
    screen = _RecordingScreen()
    t = _Transcript(screen, Settings())
    t.event(_Event(EventType.CONTENT, text="\n\nAll good."))
    assert screen.writes == [("All good.", True)]


def test_newline_only_chunk_does_not_open_a_block():
    screen = _RecordingScreen()
    t = _Transcript(screen, Settings())
    t.event(_Event(EventType.CONTENT, text="\n\n"))
    t.event(_Event(EventType.CONTENT, text="All good."))
    # the whitespace-only chunk is dropped; the first real chunk opens the
    # block so the separator still lands before visible text.
    assert screen.writes == [("All good.", True)]


def test_user_trailing_whitespace_is_stripped():
    screen = _RecordingScreen()
    t = _Transcript(screen, Settings())
    t.event(_Event(EventType.USER, text="hello\n\n"))
    assert screen.writes == [("> hello\n", True)]


# --- replay (:redraw / :load / :history repaint) ---

from cai.tui import _replay_messages


class _KindScreen:
    def __init__(self):
        self.writes = []                    # (text, kind) per write

    def write(self, text, kind=None, block=False):
        self.writes.append((text, kind))


def _convo():
    call = {"id": "c1",
            "type": "function",
            "function": {"name": "ls", "arguments": '{"path": "/tmp"}'}}
    messages = []
    messages.append({"role": "user", "content": "hi"})
    messages.append({"role": "assistant",
                     "content": "",
                     "tool_calls": [call],
                     "_reasoning": "let me look"})
    messages.append({"role": "tool", "tool_call_id": "c1", "content": "12345"})
    messages.append({"role": "assistant", "content": "done", "_reasoning": "ok"})
    return messages


def test_replay_renders_turns_tools_and_reasoning_when_shown():
    screen = _KindScreen()
    cfg = Settings()
    cfg.show_reasoning = True
    _replay_messages(screen, _convo(), cfg)
    assert screen.writes == [("> hi\n", Screen.USER),
                             ("let me look\n", Screen.REASONING),
                             ("-> ls(path=/tmp)\n", Screen.TOOL),
                             ("<- ls: 5 chars\n", Screen.TOOL),
                             ("ok\n", Screen.REASONING),
                             ("done\n", Screen.LLM)]


def test_replay_skips_reasoning_when_hidden_or_unconfigured():
    hidden = _KindScreen()
    cfg = Settings()
    cfg.show_reasoning = False
    _replay_messages(hidden, _convo(), cfg)
    bare = _KindScreen()
    _replay_messages(bare, _convo())
    for screen in (hidden, bare):
        kinds = []
        for _text, kind in screen.writes:
            kinds.append(kind)
        assert Screen.REASONING not in kinds
        assert kinds == [Screen.USER, Screen.TOOL, Screen.TOOL, Screen.LLM]


def test_replay_maps_a_tool_reply_without_a_known_call_to_a_placeholder():
    screen = _KindScreen()
    _replay_messages(screen, [{"role": "tool", "tool_call_id": "gone", "content": "x"}])
    assert screen.writes == [("<- ?: 1 chars\n", Screen.TOOL)]
