"""layout manager for the alternate-screen TUI: the three-region screen -
content area, prompt/input, status line (bottom row)."""

import sys

from .ansi import (
    SGR_RESET, SGR_BOLD, SGR_DIM, SGR_DIM_GRAY, SGR_REVERSE,
    SGR_AZURE_ON_DGRAY, SGR_BOLD_AZURE, SGR_REVERSE_YELLOW,
    CUR_HIDE, CUR_SHOW, CURSOR_BLOCK,
    ERASE_LINE, ERASE_SCREEN,
    cur_move, ansi_strip, wrap_ansi,
)
from .state import Mode


def _count_visual_rows(line, prefix_len, cols):
    """visual (terminal-wrapped) row count for one logical prompt line."""
    total = prefix_len + len(line)
    return max(1, (total + cols - 1) // cols)


def _cursor_visual_pos(input_buf, cursor_pos, prompt_prefix, cont_prefix, cols, line_vrows):
    """return (rows_down_from_first_prompt_row, cursor_col)."""
    chars_before = ''.join(input_buf[:cursor_pos])
    line_idx = chars_before.count('\n')
    last_nl = chars_before.rfind('\n')
    cursor_in_line = len(chars_before) - last_nl - 1

    prefix = prompt_prefix
    if line_idx > 0:
        prefix = cont_prefix
    visual_col_abs = len(prefix) + cursor_in_line

    cursor_vrow = sum(line_vrows[:line_idx]) + visual_col_abs // cols
    cursor_col = visual_col_abs % cols
    return cursor_vrow, cursor_col


def _apply_spans(plain, spans):
    """wrap each (start, end) half-open span in plain with SGR_REVERSE_YELLOW."""
    parts = []
    prev = 0
    for s, e in sorted(spans):
        s = max(prev, min(s, len(plain)))
        e = max(s, min(e, len(plain)))
        if s > prev:
            parts.append(plain[prev:s])
        parts.append(SGR_REVERSE_YELLOW)
        parts.append(plain[s:e])
        parts.append(SGR_RESET)
        prev = e
    if prev < len(plain):
        parts.append(plain[prev:])
    return ''.join(parts)


_MODE_LABELS = {
    Mode.NORMAL:      '-- NORMAL --',
    Mode.INSERT:      '-- INSERT --',
    Mode.VISUAL:      '-- VISUAL --',
    Mode.VISUAL_LINE: '-- VISUAL LINE --',
    Mode.COMMAND:     '-- COMMAND --',
    Mode.SEARCH:      '',  # search prompt rendered inline
}


class Layout:
    """manages the alternate-screen three-region layout."""

    def __init__(self, rows, cols):
        self._rows = max(3, rows)
        self._cols = max(1, cols)
        self._input_height = 1

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def content_rows(self):
        """number of rows for the conversation content viewport."""
        return max(1, self._rows - 1 - self._input_height)

    @property
    def status_row(self):
        """1-indexed row for the status line (always the last row)."""
        return self._rows

    @property
    def input_start_row(self):
        """1-indexed row where the input area starts."""
        return self._rows - self._input_height

    @property
    def input_height(self):
        return self._input_height

    def update_input_height(self, input_buf, prompt_prefix, cont_prefix):
        """recompute input area height based on current buffer content."""
        buf_str = ''.join(input_buf)
        lines = buf_str.split('\n')
        cols = max(1, self._cols)

        total_vrows = 0
        for i, line in enumerate(lines):
            prefix = prompt_prefix
            if i > 0:
                prefix = cont_prefix
            total_vrows += _count_visual_rows(line, len(prefix), cols)

        max_input = max(1, self._rows // 3)
        self._input_height = max(1, min(total_vrows, max_input))

    def resize(self, rows, cols):
        self._rows = max(3, rows)
        self._cols = max(1, cols)

    def render_content(self,
                       lines,
                       content_rows,
                       cols,
                       cursor_row=None,
                       selection=None,
                       search_spans=None,
                       viewport_offset=0,
                       widget_lines=None,
                       positioned_cells=None):
        """render the content viewport directly to stdout.

        selection is (start_row, end_row, line_mode, start_col, end_col)
        where rows/cols refer to buffer line indices. search_spans maps a
        line index to a list of (start_col, end_col) half-open spans of
        matched substrings - each span is highlighted individually.

        widget_lines are hover-widget lines painted over the top-right of
        the viewport; positioned_cells are (vrow, col, text) placements with
        their own anchor (positioned chips), col 0-based and clipped to the
        viewport here. each widget cell is emitted right after its row's
        content, in the same output buffer - erase and overlay stay adjacent,
        so a stdio flush landing mid-frame never shows a widget-less row."""
        widget_cells = {}
        for vrow, col, text in self._widget_cells(widget_lines or [], cols):
            widget_cells.setdefault(vrow, []).append((col, text))
        for vrow, col, text in positioned_cells or []:
            if not 0 <= vrow < content_rows: continue
            if col < 0 or col >= cols: continue
            if col + len(ansi_strip(text)) > cols:
                text = ansi_strip(text)[:cols - col]
            widget_cells.setdefault(vrow, []).append((col + 1, text))
        out = []
        sel_sr = 0
        sel_er = 0
        sel_sc = 0
        sel_ec = 0
        sel_line_mode = False
        if selection is not None:
            sel_sr, sel_er, sel_line_mode, sel_sc, sel_ec = selection

        for vrow in range(content_rows):
            buf_line_idx = viewport_offset + vrow
            out.append(cur_move(vrow + 1, 1))
            out.append(ERASE_LINE)
            if buf_line_idx < len(lines):
                line = lines[buf_line_idx]

                line_spans = None
                if search_spans:
                    line_spans = search_spans.get(buf_line_idx)

                # highlight substring search matches (stripping ANSI on
                # matched lines - same trade-off as selection highlighting).
                if line_spans:
                    line = _apply_spans(ansi_strip(line), line_spans)
                # highlight visual selection (no search spans on this line)
                elif selection is not None and sel_sr <= buf_line_idx <= sel_er:
                    plain = ansi_strip(line)
                    if sel_line_mode:
                        # line-mode: entire line highlighted
                        line = SGR_REVERSE + plain + SGR_RESET
                    elif sel_sr == sel_er:
                        # single-line character selection
                        sc = min(sel_sc, len(plain))
                        ec = min(sel_ec + 1, len(plain))
                        line = (plain[:sc] + SGR_REVERSE +
                                plain[sc:ec] + SGR_RESET +
                                plain[ec:])
                    elif buf_line_idx == sel_sr:
                        sc = min(sel_sc, len(plain))
                        line = plain[:sc] + SGR_REVERSE + plain[sc:] + SGR_RESET
                    elif buf_line_idx == sel_er:
                        ec = min(sel_ec + 1, len(plain))
                        line = SGR_REVERSE + plain[:ec] + SGR_RESET + plain[ec:]
                    else:
                        # middle lines: fully highlighted
                        line = SGR_REVERSE + plain + SGR_RESET

                out.append(line)
            out.append(SGR_RESET)
            for wcol, wtext in widget_cells.get(vrow, ()):
                out.append(cur_move(vrow + 1, wcol))
                out.append(wtext)
                out.append(SGR_RESET)
        sys.stdout.write(''.join(out))

    def render_status(self,
                      mode,
                      status_text,
                      status_right='',
                      search_buf=None,
                      search_direction=1,
                      viewport_offset=0,
                      total_lines=0,
                      cols=80,
                      auto_scroll=True,
                      new_content_below=False,
                      command_buf=None,
                      cursor_row=0):
        """render the status bar at the bottom row."""
        out = [cur_move(self.status_row, 1), ERASE_LINE]

        if mode == Mode.COMMAND:
            # command mode: show ':' prefix + command text on status line
            cmd_text = ''
            if command_buf:
                cmd_text = ''.join(command_buf)
            out.append(f':{cmd_text}')
            out.append(SGR_RESET)
            sys.stdout.write(''.join(out))
            # position cursor right after the typed text
            sys.stdout.write(cur_move(self.status_row, 2 + len(cmd_text)))
            sys.stdout.write(CUR_SHOW)
            return

        if mode == Mode.SEARCH:
            # show search prompt in status bar
            prefix = '?'
            if search_direction == 1:
                prefix = '/'
            pattern = ''
            if search_buf:
                pattern = ''.join(search_buf)
            out.append(f'{prefix}{pattern}')
            out.append(SGR_RESET)
            sys.stdout.write(''.join(out))
            # position cursor right after the typed text
            sys.stdout.write(cur_move(self.status_row, 1 + len(prefix) + len(pattern)))
            sys.stdout.write(CUR_SHOW)
            return

        out.append(SGR_AZURE_ON_DGRAY)
        label = _MODE_LABELS.get(mode, '')
        left = f' {label}'
        if status_text:
            left = f' {label} | {status_text}'

        # right side: context usage and scroll percentage, in that order,
        # separated by ' | ' so the ctx reads as a left-neighbour of the
        # scroll position rather than folded into the left status text.
        right_parts = []
        if status_right:
            right_parts.append(status_right)
        if total_lines > 0:
            if auto_scroll:
                effective_row = total_lines - 1
            else:
                effective_row = min(cursor_row, total_lines - 1)
            pct = min(100, int((effective_row + 1) / total_lines * 100))
            right_parts.append(f'{pct}%')
        right = ''
        if right_parts:
            right = ' ' + ' | '.join(right_parts) + ' '

        if new_content_below and not auto_scroll:
            right = ' [new content below] ' + right

        # pad to fill the row
        pad = cols - len(ansi_strip(left)) - len(ansi_strip(right))
        if pad < 0:
            pad = 0
        status_line = left + ' ' * pad + right

        out.append(status_line[:cols])
        out.append(SGR_RESET)
        sys.stdout.write(''.join(out))

    def render_input(self,
                     input_buf,
                     cursor_pos,
                     mode,
                     prompt_prefix,
                     cont_prefix,
                     cols,
                     command_buf=None):
        """render the input/command area above the status line."""
        cols = max(1, cols)
        input_start = self.input_start_row

        if mode == Mode.COMMAND:
            # command mode renders on the status line (bottom row), like
            # vim. clear the input area first.
            sys.stdout.write(cur_move(input_start, 1) + ERASE_LINE)
            return

        # normal / insert / visual / search modes
        buf_str = ''.join(input_buf)
        lines = buf_str.split('\n')

        line_vrows = []
        for i, line in enumerate(lines):
            prefix = prompt_prefix
            if i > 0:
                prefix = cont_prefix
            line_vrows.append(_count_visual_rows(line, len(prefix), cols))

        # erase the full input area first - otherwise we'd wipe out the
        # continuation rows that the terminal just wrapped our text onto.
        for vr in range(self.status_row - input_start):
            sys.stdout.write(cur_move(input_start + vr, 1) + ERASE_LINE)

        # render each logical line, manually wrapped so each visual row is
        # placed explicitly rather than relying on terminal auto-wrap.
        vrow_offset = 0
        for i, line in enumerate(lines):
            prefix = prompt_prefix
            if i > 0:
                prefix = cont_prefix
            wrapped = wrap_ansi(f'{prefix}{line}', cols)
            if not wrapped:
                wrapped = ['']
            for wline in wrapped:
                abs_row = input_start + vrow_offset
                if abs_row >= self.status_row:
                    break
                sys.stdout.write(cur_move(abs_row, 1) + wline)
                vrow_offset += 1
            if input_start + vrow_offset >= self.status_row:
                break

        if mode == Mode.INSERT:
            # position cursor in prompt area
            cursor_vrow, cursor_col = _cursor_visual_pos(input_buf,
                                                         cursor_pos,
                                                         prompt_prefix,
                                                         cont_prefix,
                                                         cols,
                                                         line_vrows)
            abs_cursor_row = input_start + cursor_vrow
            sys.stdout.write(cur_move(abs_cursor_row, cursor_col + 1))
            sys.stdout.write(CUR_SHOW)

    def _widget_cells(self, lines, cols):
        """(vrow, col, text) placements for hover-widget lines: line i
        right-aligned on viewport row i, a stack taller than the viewport cut
        and closed with a dim +N row. empty lines (block gaps) consume their
        row but place nothing."""
        rows = self.content_rows
        shown = lines
        extra = 0
        if len(lines) > rows:
            shown = lines[:max(0, rows - 1)]
            extra = len(lines) - len(shown)
        cells = []
        vrow = 0
        for line in shown:
            width = len(ansi_strip(line))
            if width > cols:
                line = ansi_strip(line)[:cols]
                width = cols
            if width > 0:
                cells.append((vrow, cols - width + 1, line))
            vrow += 1
        if extra > 0:
            tag = f' +{extra} '
            cells.append((vrow, cols - len(tag) + 1, f'{SGR_DIM_GRAY}{tag}{SGR_RESET}'))
        return cells

    def render_widgets(self, lines, cols):
        """paint hover-widget lines over the content viewport, anchored
        top-right. each line only overwrites its own cells. streaming paths
        should pass widget_lines to render_content instead, which interleaves
        the widgets row by row so no flush boundary can separate a row's
        erase from its widget."""
        out = []
        for vrow, col, text in self._widget_cells(lines, cols):
            out.append(cur_move(vrow + 1, col))
            out.append(text)
            out.append(SGR_RESET)
        sys.stdout.write(''.join(out))

    def position_cursor(self, mode, cursor_row, viewport_offset, cursor_col=0):
        """final cursor placement after all regions have been rendered.
        only NORMAL and VISUAL modes need explicit positioning here (block
        cursor in the content area). INSERT, COMMAND, and SEARCH modes
        already had their cursor placed by render_input / render_status -
        touching the cursor here would undo that work."""
        if mode in (Mode.NORMAL, Mode.VISUAL, Mode.VISUAL_LINE):
            vrow = cursor_row - viewport_offset
            if 0 <= vrow < self.content_rows:
                sys.stdout.write(cur_move(vrow + 1, min(cursor_col + 1, self._cols)))
            sys.stdout.write(CURSOR_BLOCK + CUR_SHOW)

    def render_all(self,
                   buffer_lines,
                   viewport_offset,
                   mode,
                   status_text,
                   input_buf,
                   cursor_pos,
                   prompt_prefix,
                   cont_prefix,
                   status_right='',
                   search_buf=None,
                   search_direction=1,
                   search_spans=None,
                   selection=None,
                   total_lines=0,
                   command_buf=None,
                   auto_scroll=True,
                   new_content_below=False,
                   cursor_row=0,
                   cursor_col=0,
                   widget_lines=None,
                   positioned_cells=None):
        """full screen redraw (used on resize and initial draw)."""
        self.render_content(buffer_lines,
                            self.content_rows,
                            self._cols,
                            selection=selection,
                            search_spans=search_spans,
                            viewport_offset=viewport_offset,
                            widget_lines=widget_lines,
                            positioned_cells=positioned_cells)
        if mode in (Mode.COMMAND, Mode.SEARCH):
            # render input first (clears prompt area), then status last
            # so cursor ends up on the status line.
            self.render_input(input_buf,
                              cursor_pos,
                              mode,
                              prompt_prefix,
                              cont_prefix,
                              self._cols)
            self.render_status(mode,
                               status_text,
                               status_right=status_right,
                               search_buf=search_buf,
                               search_direction=search_direction,
                               viewport_offset=viewport_offset,
                               total_lines=total_lines,
                               cols=self._cols,
                               auto_scroll=auto_scroll,
                               new_content_below=new_content_below,
                               command_buf=command_buf,
                               cursor_row=cursor_row)
        else:
            # render status first, then input last so cursor ends up in the
            # prompt area (INSERT) or gets repositioned by position_cursor
            # (NORMAL/VISUAL).
            self.render_status(mode,
                               status_text,
                               status_right=status_right,
                               search_buf=search_buf,
                               search_direction=search_direction,
                               viewport_offset=viewport_offset,
                               total_lines=total_lines,
                               cols=self._cols,
                               auto_scroll=auto_scroll,
                               new_content_below=new_content_below,
                               cursor_row=cursor_row)
            self.render_input(input_buf,
                              cursor_pos,
                              mode,
                              prompt_prefix,
                              cont_prefix,
                              self._cols)
        self.position_cursor(mode, cursor_row, viewport_offset, cursor_col)
        sys.stdout.flush()
