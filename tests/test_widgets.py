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
from cai.tui import _agent_chip_lines, _chip_lines, _pending_chip_lines, _Pending


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


# --- the pending chip builder ---


def test_pending_chip_lines_empty_when_both_zero():
    assert _pending_chip_lines(0, 0) == []


def test_pending_chip_lines_steer_above_user():
    lines = _pending_chip_lines(1, 2)
    assert len(lines) == 2
    assert "2 steering" in ansi_strip(lines[0])
    assert "1 queued" in ansi_strip(lines[1])


def test_pending_chip_lines_one_row_each():
    assert len(_pending_chip_lines(3, 0)) == 1
    assert "3 queued" in ansi_strip(_pending_chip_lines(3, 0)[0])
    assert len(_pending_chip_lines(0, 4)) == 1
    assert "4 steering" in ansi_strip(_pending_chip_lines(0, 4)[0])


def test_pending_chip_lines_rows_share_a_width():
    lines = _pending_chip_lines(1, 10)
    assert len(ansi_strip(lines[0])) == len(ansi_strip(lines[1]))


# --- the _Pending widget owner ---


class _WidgetHost:
    """a Screen stand-in that just records the last add/remove of a widget."""

    def __init__(self):
        self.widgets = {}

    def add_widget(self, name, lines):
        self.widgets[name] = lines

    def remove_widget(self, name):
        self.widgets.pop(name, None)


class _Cfg:
    def __init__(self, show_chips=True):
        self.show_chips = show_chips


def test_pending_counts_user_turns_up_and_down():
    host = _WidgetHost()
    pending = _Pending(host, _Cfg())
    pending.user_queued()
    pending.user_queued()
    assert "2 queued" in ansi_strip(host.widgets["pending"][0])
    pending.user_started()
    assert "1 queued" in ansi_strip(host.widgets["pending"][0])
    pending.user_started()
    assert "pending" not in host.widgets


def test_pending_user_started_never_goes_negative():
    host = _WidgetHost()
    pending = _Pending(host, _Cfg())
    pending.user_started()
    assert "pending" not in host.widgets


def test_pending_tracks_the_steer_count():
    host = _WidgetHost()
    pending = _Pending(host, _Cfg())
    pending.set_steer(3)
    assert "3 steering" in ansi_strip(host.widgets["pending"][0])
    pending.set_steer(0)
    assert "pending" not in host.widgets


def test_pending_hidden_when_show_chips_off():
    host = _WidgetHost()
    cfg = _Cfg(show_chips=False)
    pending = _Pending(host, cfg)
    pending.user_queued()
    pending.set_steer(2)
    assert "pending" not in host.widgets
    # turning chips back on and refreshing restores the widget
    cfg.show_chips = True
    pending.refresh()
    assert "pending" in host.widgets


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
