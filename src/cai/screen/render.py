"""rendering of tool display blocks - the user-only content a tool returns
alongside its model-facing result (see cai.utils.ToolResult).

tools stay terminal-agnostic: a block carries plain text plus a render hint
(cai/render MCP meta), and this module owns the ANSI. known hints are diff
(unified diff), table (tab-separated rows, header first), and grep (vimgrep
file:line:col:text matches). unknown hints fall back to dim plain text, so
a misspelled hint degrades instead of breaking."""

import re

from .ansi import (
    SGR_CYAN, SGR_DIM_GRAY, SGR_GREEN, SGR_RED, SGR_RESET,
    SGR_MAGENTA, SGR_YELLOW, ansi_strip)


_PREVIEW_ROLE_COLOR = {
    'system':    SGR_MAGENTA,
    'user':      SGR_GREEN,
    'assistant': SGR_CYAN,
    'tool':      SGR_YELLOW,
}


def _preview_lines(convo, width, max_lines):
    """one-line-per-message preview of a conversation tail, colored by role.
    shared by the session picker and the :agents sub-agent preview."""
    from .state import _overlay_msg_text

    lines = []
    tail = convo
    if len(convo) > max_lines:
        tail = convo[-max_lines:]
    for i, msg in enumerate(tail):
        abs_i = (len(convo) - len(tail)) + i
        role = msg.get('role', '?')
        color = _PREVIEW_ROLE_COLOR.get(role, '')
        text = ansi_strip(_overlay_msg_text(msg).replace('\n', ' ').replace('\r', ' '))
        raw = f'#{abs_i} {role[:9].ljust(9)} {text}'[:width]
        if color:
            lines.append(f'{color}{raw}{SGR_RESET}')
        else:
            lines.append(raw)
    return lines

# a single display block never paints more than this many lines - a large
# edit shouldn't flood the transcript; the model-facing text already carries
# the full story for anyone who expands it.
DISPLAY_MAX_LINES = 40

# transcript indent matching the "  <- tool: ..." summary lines.
_INDENT = '  '

# a table cell wider than this is cut with an ellipsis so one long value
# doesn't pad its whole column into terminal wrapping.
_TABLE_CELL_MAX = 48


def _diff_line_style(line):
    if line.startswith('+++') or line.startswith('---'):
        return SGR_DIM_GRAY
    if line.startswith('@@'):
        return SGR_CYAN
    if line.startswith('+'):
        return SGR_GREEN
    if line.startswith('-'):
        return SGR_RED
    return SGR_DIM_GRAY


def _format_plain(lines):
    return [f"{SGR_DIM_GRAY}{line}{SGR_RESET}" for line in lines]


def _format_diff(lines):
    return [f"{_diff_line_style(line)}{line}{SGR_RESET}" for line in lines]


# vimgrep format: file:line:col:text (col optional).
_GREP_RE = re.compile(r'^(.+?):(\d+):(?:(\d+):)?(.*)$')


def _format_grep(lines):
    """vimgrep matches: path cyan, line/col location dim, matched text in
    the terminal's default color so it stands out. lines that don't parse
    (footers, "No matches found.") fall back to dim plain text."""
    out = []
    for line in lines:
        m = _GREP_RE.match(line)
        if not m:
            out.append(f"{SGR_DIM_GRAY}{line}{SGR_RESET}")
            continue
        path, row, col, text = m.groups()
        loc = f":{row}:"
        if col:
            loc = f":{row}:{col}:"
        out.append(f"{SGR_CYAN}{path}{SGR_RESET}"
                   f"{SGR_DIM_GRAY}{loc}{SGR_RESET}{text}{SGR_RESET}")
    return out


def _format_table(lines):
    """tab-separated rows, first row the header: cells padded so columns
    align, header cyan, data rows dim. the last cell of each row is left
    unpadded so rows carry no trailing spaces."""
    rows = []
    for line in lines:
        cells = []
        for cell in line.split('\t'):
            if len(cell) > _TABLE_CELL_MAX:
                cell = cell[:_TABLE_CELL_MAX - 1] + '…'
            cells.append(cell)
        rows.append(cells)

    widths = {}
    for cells in rows:
        for i, cell in enumerate(cells):
            widths[i] = max(widths.get(i, 0), len(cell))

    out = []
    for n, cells in enumerate(rows):
        padded_cells = []
        for i, cell in enumerate(cells):
            if i == len(cells) - 1:
                padded_cells.append(cell)
            else:
                padded_cells.append(cell.ljust(widths[i]))
        padded = '  '.join(padded_cells)

        style = SGR_DIM_GRAY
        if n == 0:
            style = SGR_CYAN
        out.append(f"{style}{padded}{SGR_RESET}")
    return out


_FORMATTERS = {
    'diff': _format_diff,
    'grep': _format_grep,
    'table': _format_table,
}


def render_display(blocks):
    """render a tool_result Event's display blocks into ANSI-styled text.
    each line is styled individually (style ... reset), so the result can be
    embedded in any surrounding style state; ends with '\\n' when non-empty."""
    out = []
    for block in blocks or []:
        text = (block.get('text') or '').strip('\n')
        if not text: continue
        lines = text.split('\n')
        hidden = len(lines) - DISPLAY_MAX_LINES
        if hidden > 0:
            lines = lines[:DISPLAY_MAX_LINES]
        fmt = _FORMATTERS.get(block.get('render'), _format_plain)
        # each line opens with a reset so an inherited surrounding style
        # (e.g. the faint TOOL kind prefix) cannot bleed into its color -
        # a color SGR alone would not clear an attribute like faint.
        for line in fmt(lines):
            out.append(f"{SGR_RESET}{_INDENT}{line}\n")
        if hidden > 0:
            out.append(f"{SGR_RESET}{_INDENT}{SGR_DIM_GRAY}… +{hidden} more lines{SGR_RESET}\n")
    return ''.join(out)
