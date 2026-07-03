"""Tests for the hover-widget layer: the chips builder (skills column left of
the tools column, uniform pill widths), the layout's top-right widget painting
with +N truncation, and the screen's widget registry flattening."""
import re

from cai.screen.ansi import ansi_strip, cur_move
from cai.screen.layout import Layout
from cai.screen.screen import Screen
from cai.tui import _agent_chip_lines, _chip_lines


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
    tools_cell = len(" fs__write ")
    assert len(ansi_strip(lines[1])) == tools_cell


def test_chip_lines_blanks_the_tools_cell_when_tools_run_out():
    lines = _chip_lines(["git", "web"], ["fs"])
    # both rows span the full width: the second row pads the tools cell so
    # the skill pill stays in its column.
    assert len(ansi_strip(lines[0])) == len(ansi_strip(lines[1]))
    assert ansi_strip(lines[1]).endswith(" ")


def test_chip_lines_empty():
    assert _chip_lines([], []) == []


def test_agent_chip_lines_share_a_width():
    lines = _agent_chip_lines(["scout", "researcher"])
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
