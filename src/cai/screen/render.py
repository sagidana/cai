"""rendering of tool display blocks - the user-only content a tool returns
alongside its model-facing result (see cai.utils.ToolResult).

tools stay terminal-agnostic: a block carries plain text plus a render hint
(cai/render MCP meta), and this module owns the ANSI. known hints are diff
(unified diff), table (tab-separated rows, header first), and grep (vimgrep
file:line:col:text matches). unknown hints fall back to dim plain text, so
a misspelled hint degrades instead of breaking.

the first-class `python` tool gets the same treatment on its CALL side: its
input is always a python script, so the transcript shows the code argument as
an indented, syntax-colored block (render_python_code) instead of squashing it
into the one-line arg preview."""

import io
import keyword
import re
import tokenize

from .ansi import (
    SGR_CYAN, SGR_DIM_GRAY, SGR_GREEN, SGR_RED, SGR_RESET,
    SGR_MAGENTA, SGR_YELLOW, ansi_strip, ansi_sanitize, display_truncate)
from cai.pytool import PY_TOOL_NAME


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
        text = ansi_sanitize(_overlay_msg_text(msg))
        raw = display_truncate(f'#{abs_i} {role[:9].ljust(9)} {text}', width)
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


# python syntax colors, subtle on purpose: keywords magenta, strings green,
# comments dim, numbers yellow, everything else the terminal's default.
_PY_TOKEN_COLORS = {
    tokenize.STRING:  SGR_GREEN,
    tokenize.COMMENT: SGR_DIM_GRAY,
    tokenize.NUMBER:  SGR_YELLOW,
}

# f-strings tokenize as their own kinds from 3.12 on; absent members just
# don't get an entry (the inner expressions still color normally).
for _fstring in ('FSTRING_START', 'FSTRING_MIDDLE', 'FSTRING_END'):
    _kind = getattr(tokenize, _fstring, None)
    if _kind is not None:
        _PY_TOKEN_COLORS[_kind] = SGR_GREEN


def _py_token_color(kind, text):
    if kind == tokenize.NAME and keyword.iskeyword(text):
        return SGR_MAGENTA
    return _PY_TOKEN_COLORS.get(kind)


def _python_spans(code):
    """per-line color spans {line_index: [(start_col, end_col, color)]} from
    tokenize; end_col None means to end of line (a multi-line token). None as
    a whole when the code does not tokenize (an unfinished snippet)."""
    spans = {}
    reader = io.StringIO(code).readline
    try:
        for tok in tokenize.generate_tokens(reader):
            color = _py_token_color(tok.type, tok.string)
            if not color: continue
            srow, scol = tok.start
            erow, ecol = tok.end
            row = srow
            while row <= erow:
                start = 0
                if row == srow:
                    start = scol
                end = None
                if row == erow:
                    end = ecol
                spans.setdefault(row - 1, []).append((start, end, color))
                row += 1
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return None
    return spans


def _format_python(lines, spans):
    """apply the color spans to their lines; tokens never overlap and arrive
    in source order, so each line is a simple left-to-right stitch."""
    out = []
    for i, line in enumerate(lines):
        pieces = []
        pos = 0
        for start, end, color in spans.get(i, []):
            if end is None:
                end = len(line)
            if start > pos:
                pieces.append(line[pos:start])
            pieces.append(f"{color}{line[start:end]}{SGR_RESET}")
            pos = end
        pieces.append(line[pos:])
        out.append(''.join(pieces))
    return out


def python_code_arg(tool_name, tool_args):
    """the script of a `python` tool call, or None when the call isn't one
    (or carries no usable code) - the caller then falls back to the plain
    one-line arg preview."""
    if tool_name != PY_TOOL_NAME:
        return None
    code = (tool_args or {}).get('code')
    if not isinstance(code, str):
        return None
    if not code.strip():
        return None
    return code


def render_python_code(code):
    """ANSI view of a python tool call's script, shaped like a display block:
    indented under the '->' line, syntax-colored. deliberately NOT capped at
    DISPLAY_MAX_LINES - the script IS the tool call, so the transcript always
    shows all of it. a snippet that does not tokenize renders dim plain
    instead; ends with '\\n' when non-empty."""
    text = code.strip('\n')
    if not text:
        return ''
    lines = text.split('\n')
    spans = _python_spans(text)
    if spans is None:
        styled = _format_plain(lines)
    else:
        styled = _format_python(lines, spans)
    out = []
    for line in styled:
        out.append(f"{SGR_RESET}{_INDENT}{line}\n")
    return ''.join(out)


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
