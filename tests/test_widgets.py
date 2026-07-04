"""Tests for the hover-widget layer: the Chip primitive (a styled pill with
optional position and timeout), the chips builders (skills column left of the
tools column, uniform chip widths), the layout's top-right widget painting
with +N truncation, and the screen's widget registry flattening."""
import re
import threading
import time

from cai.screen.ansi import ansi_strip, cur_move
from cai.screen.chip import Chip
from cai.screen.layout import Layout
from cai.screen.screen import Screen
from cai.tui import _agent_chip_lines, _chip_lines


# --- the Chip primitive ---


def test_chip_pads_its_text():
    lines = Chip("git").lines()
    assert [ansi_strip(l) for l in lines] == [" git "]


def test_chip_styles_the_body():
    line = Chip("git", sgr="\033[36m").lines()[0]
    assert line.startswith("\033[36m")


def test_chip_defaults():
    chip = Chip("git")
    assert chip.position is None
    assert chip.timeout is None


# --- the chips builders ---


def test_chip_lines_skills_left_of_tools():
    lines = _chip_lines(["git"], ["fs__read", "fs__write"])
    assert len(lines) == 2
    first = ansi_strip(lines[0])
    assert first.index("git") < first.index("fs__read")


def test_chip_lines_columns_share_a_width():
    lines = _chip_lines([], ["fs", "fs__write"])
    assert len(ansi_strip(lines[0])) == len(ansi_strip(lines[1]))


def test_chip_lines_rows_align_when_skills_run_out():
    lines = _chip_lines(["git"], ["fs__read", "fs__write"])
    # the tools-only row is exactly one tools cell wide, so right-alignment
    # lands it in the tools column.
    tools_cell = len(" ✓ fs__write ")
    assert len(ansi_strip(lines[1])) == tools_cell


def test_chip_lines_blanks_the_tools_cell_when_tools_run_out():
    lines = _chip_lines(["git", "web"], ["fs"])
    # both rows span the full width: the second row pads the tools cell so
    # the skill chip stays in its column.
    assert len(ansi_strip(lines[0])) == len(ansi_strip(lines[1]))
    assert ansi_strip(lines[1]).endswith(" ")


def test_chip_lines_check_activated_entries():
    lines = _chip_lines(["git"], ["fs"])
    body = ansi_strip(lines[0])
    assert body.count("✓") == 2
    assert re.search(r"✓ git\b", body)
    assert re.search(r"✓ fs\b", body)


def test_chip_lines_empty():
    assert _chip_lines([], []) == []


def test_agent_chip_lines_share_a_width():
    lines = _agent_chip_lines(["scout", "researcher"])
    # one framed chip per agent, one row each
    assert len(lines) == 2
    assert len(ansi_strip(lines[0])) == len(ansi_strip(lines[1]))


def test_agent_chip_lines_empty():
    assert _agent_chip_lines([]) == []


def test_render_widgets_right_aligns_each_line(capsys):
    layout = Layout(10, 40)
    layout.render_widgets(["abc", "de"], 40)
    out = capsys.readouterr().out
    assert cur_move(1, 38) in out
    assert cur_move(2, 39) in out


def test_render_widgets_truncates_with_a_count(capsys):
    layout = Layout(10, 40)
    lines = []
    for i in range(10):
        lines.append(f"w{i}")
    layout.render_widgets(lines, 40)
    out = capsys.readouterr().out
    # content_rows is 8: seven lines painted, the eighth row is the +3 tag.
    assert "w6" in out
    assert "w7" not in out
    assert "+3" in out


class _Carrier:
    def __init__(self, widgets):
        self._widgets = widgets


def test_widget_lines_flatten_in_insertion_order_with_a_gap():
    widgets = {}
    widgets["chips"] = ["a", "b"]
    widgets["other"] = ["c"]
    assert Screen._widget_lines(_Carrier(widgets)) == ["a", "b", "", "c"]


# --- chips on the screen: stacking, anchoring, timeout ---


class _ChipHost:
    """a Screen stand-in with just the chip registry state. the focus stack
    is parked off 'main' so add/remove skip the terminal repaint."""

    add_chip = Screen.add_chip
    remove_chip = Screen.remove_chip
    _positioned_cells = Screen._positioned_cells

    def __init__(self):
        self._widgets = {}
        self._positioned_chips = {}
        self._chip_timers = {}
        self._render_lock = threading.RLock()
        self._focus_stack = ['overlay']


def test_add_chip_without_position_joins_the_widget_stack():
    host = _ChipHost()
    host.add_chip("note", Chip("hi"))
    assert host._widgets["note"] == Chip("hi").lines()
    assert host._positioned_chips == {}


def test_add_chip_with_position_anchors_its_cells():
    host = _ChipHost()
    host.add_chip("note", Chip("hi", position=(5, 2)))
    assert "note" not in host._widgets
    cells = host._positioned_cells()
    assert [(vrow, col) for vrow, col, _ in cells] == [(5, 2)]


def test_add_chip_replacement_can_move_between_stack_and_anchor():
    host = _ChipHost()
    host.add_chip("note", Chip("hi"))
    host.add_chip("note", Chip("hi", position=(1, 1)))
    assert "note" not in host._widgets
    assert "note" in host._positioned_chips


def test_remove_chip_forgets_both_registries():
    host = _ChipHost()
    host.add_chip("stacked", Chip("a"))
    host.add_chip("anchored", Chip("b", position=(0, 0)))
    host.remove_chip("stacked")
    host.remove_chip("anchored")
    host.remove_chip("unknown")
    assert host._widgets == {}
    assert host._positioned_chips == {}


def test_chip_timeout_removes_it():
    host = _ChipHost()
    host.add_chip("note", Chip("hi", timeout=0.05))
    assert "note" in host._widgets
    deadline = time.monotonic() + 2
    while "note" in host._widgets and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "note" not in host._widgets
    assert host._chip_timers == {}


def test_replacing_a_chip_cancels_the_old_timeout():
    host = _ChipHost()
    host.add_chip("note", Chip("hi", timeout=0.05))
    host.add_chip("note", Chip("hi"))
    time.sleep(0.2)
    assert "note" in host._widgets


def test_render_content_paints_positioned_cells(capsys):
    layout = Layout(10, 40)
    layout.render_content(["hello"], 8, 40,
                          positioned_cells=[(2, 5, "chip")])
    out = capsys.readouterr().out
    # 0-based (row 2, col 5) lands at 1-based terminal (3, 6)
    assert cur_move(3, 6) in out
    assert "chip" in out


def test_render_content_clips_positioned_cells(capsys):
    layout = Layout(10, 40)
    layout.render_content(["hello"], 8, 40,
                          positioned_cells=[(20, 0, "below"),
                                            (0, 38, "overflow")])
    out = capsys.readouterr().out
    assert "below" not in out
    assert "ov" in out
    assert "overflow" not in out


# --- :prompts entries ---

from cai.tui import _prompt_entries


def test_prompt_entries_dedupe_keeps_newest_first():
    history = ["fix the bug", "add tests", "fix the bug"]
    entries = _prompt_entries(history)
    assert list(entries) == ["fix the bug", "add tests"]


def test_prompt_entries_flatten_multiline_labels():
    history = ["line one\n  line two"]
    entries = _prompt_entries(history)
    assert list(entries) == ["line one line two"]
    assert entries["line one line two"] == "line one\n  line two"


def test_prompt_entries_skip_blanks():
    assert _prompt_entries(["", "  \n ", "real"]) == {"real": "real"}
