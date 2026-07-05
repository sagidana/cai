"""Tests for the python tool's transcript view: the tool is first-class and
its input is always a script, so a `python` call renders its code argument as
an indented syntax-colored block under the '->' line - live, on replay, and in
the headless CLI - instead of squashing it into the one-line arg preview."""
from cai.environment import Settings
from cai.events import EventType
from cai.screen import Screen
from cai.screen.ansi import ansi_strip, SGR_DIM_GRAY, SGR_GREEN, SGR_MAGENTA
from cai.screen.render import DISPLAY_MAX_LINES, python_code_arg, render_python_code
from cai.tui import _replay_messages, _Transcript


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
        self.writes = []                    # (text, kind, block) per write

    def write(self, text, kind=None, block=False):
        self.writes.append((text, kind, block))


# --- the probe --------------------------------------------------------------

def test_python_code_arg_only_for_python_calls_with_code():
    assert python_code_arg("python", {"code": "print(1)"}) == "print(1)"
    assert python_code_arg("fs__read_file", {"code": "print(1)"}) is None
    assert python_code_arg("python", {"timeout": 5}) is None
    assert python_code_arg("python", {"code": "   "}) is None
    assert python_code_arg("python", {"code": 7}) is None
    assert python_code_arg("python", None) is None


# --- the code block ---------------------------------------------------------

def test_render_python_code_colors_and_indents():
    text = render_python_code("for x in range(3):\n    print('hi')  # loop")
    assert f"{SGR_MAGENTA}for" in text
    assert f"{SGR_GREEN}'hi'" in text
    assert f"{SGR_DIM_GRAY}# loop" in text
    assert ansi_strip(text).splitlines() == ["  for x in range(3):",
                                             "      print('hi')  # loop"]


def test_render_python_code_is_never_trimmed():
    # the script IS the tool call - unlike display blocks, it always shows whole
    lines = []
    for i in range(DISPLAY_MAX_LINES + 10):
        lines.append(f"x{i} = {i}")
    text = render_python_code("\n".join(lines))
    assert len(text.splitlines()) == DISPLAY_MAX_LINES + 10
    assert f"x{DISPLAY_MAX_LINES + 9} = {DISPLAY_MAX_LINES + 9}" in ansi_strip(text)


def test_render_python_code_falls_back_dim_when_untokenizable():
    text = render_python_code("def broken(:\n    'unclosed")
    assert SGR_DIM_GRAY in text
    assert ansi_strip(text).splitlines() == ["  def broken(:",
                                             "      'unclosed"]


def test_render_python_code_empty_is_empty():
    assert render_python_code("\n\n") == ""


# --- live transcript --------------------------------------------------------

def test_transcript_python_call_renders_header_and_code_block():
    screen = _RecordingScreen()
    t = _Transcript(screen, Settings())
    args = {"code": "print(1)", "timeout": 5}
    t.event(_Event(EventType.TOOL_CALL, tool_name="python", tool_args=args))
    assert len(screen.writes) == 2
    header, _, opened = screen.writes[0]
    assert header == "-> python(timeout=5)\n"
    assert opened is True
    block, kind, continued = screen.writes[1]
    assert ansi_strip(block) == "  print(1)\n"
    assert kind == Screen.TOOL
    assert continued is False
    # the following result still chains into the same tool block
    t.event(_Event(EventType.TOOL_RESULT, tool_name="python", tool_result="1"))
    assert screen.writes[2][2] is False


def test_transcript_python_call_without_code_stays_one_line():
    screen = _RecordingScreen()
    t = _Transcript(screen, Settings())
    t.event(_Event(EventType.TOOL_CALL, tool_name="python", tool_args={}))
    assert len(screen.writes) == 1
    assert screen.writes[0][0] == "-> python()\n"


# --- replay -----------------------------------------------------------------

def test_replay_python_call_renders_code_block():
    call = {"id": "c1",
            "type": "function",
            "function": {"name": "python", "arguments": '{"code": "print(1)"}'}}
    messages = []
    messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
    messages.append({"role": "tool", "tool_call_id": "c1", "content": "1"})
    screen = _RecordingScreen()
    _replay_messages(screen, messages)
    texts = []
    for text, kind, _block in screen.writes:
        assert kind == Screen.TOOL
        texts.append(ansi_strip(text))
    assert texts == ["-> python()\n", "  print(1)\n", "<- python: 1 chars\n"]
