"""Layout manager for the alternate-screen TUI.

Manages the three-region screen: content area, prompt/input, status line.
Layout order (top to bottom): content | prompt | status (bottom row).
"""

import sys

from .ansi import (
    SGR_RESET, SGR_BOLD, SGR_DIM, SGR_DIM_GRAY, SGR_REVERSE,
    SGR_AZURE_ON_DGRAY, SGR_BOLD_AZURE, SGR_REVERSE_YELLOW,
    CUR_HIDE, CUR_SHOW,
    ERASE_LINE, ERASE_SCREEN,
    cur_move, ansi_strip, wrap_ansi,
)
from .state import Mode


# ── Helpers ported from footer.py ─────────────────────────────────────────────

def _count_visual_rows(line: str, prefix_len: int, cols: int) -> int:
    """Visual (terminal-wrapped) row count for one logical prompt line."""
    total = prefix_len + len(line)
    return max(1, (total + cols - 1) // cols)


def _cursor_visual_pos(
    input_buf: list,
    cursor_pos: int,
    prompt_prefix: str,
    cont_prefix: str,
    cols: int,
    line_vrows: list,
) -> tuple:
    """Return (rows_down_from_first_prompt_row, cursor_col)."""
    chars_before = ''.join(input_buf[:cursor_pos])
    line_idx = chars_before.count('\n')
    last_nl = chars_before.rfind('\n')
    cursor_in_line = len(chars_before) - last_nl - 1
    prefix_len = len(prompt_prefix if line_idx == 0 else cont_prefix)
    visual_col_abs = prefix_len + cursor_in_line
    cursor_vrow = sum(line_vrows[:line_idx]) + visual_col_abs // cols
    cursor_col = visual_col_abs % cols
    return cursor_vrow, cursor_col


def _diversify_overlay(items: list, max_visible: int) -> list:
    """Pick a diverse subset of overlay items (ported from footer.py)."""
    if len(items) <= max_visible:
        return items
    prefix = items[0]
    for item in items[1:]:
        while not item.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                break
    _SEP = set('/-_:. ')
    groups: dict[str, list] = {}
    for item in items:
        rest = item[len(prefix):]
        key_end = 0
        for i, ch in enumerate(rest):
            key_end = i + 1
            if ch in _SEP:
                break
        key = rest[:key_end] if rest else ''
        groups.setdefault(key, []).append(item)
    result: list[str] = []
    group_lists = list(groups.values())
    idx = [0] * len(group_lists)
    while len(result) < max_visible:
        added = False
        for g, gl in enumerate(group_lists):
            if len(result) >= max_visible:
                break
            if idx[g] < len(gl):
                result.append(gl[idx[g]])
                idx[g] += 1
                added = True
        if not added:
            break
    result.sort()
    return result


def _window_overlay(overlay_matches: list, cmd_overlay_idx: int, max_visible: int) -> tuple:
    """Return (visible_items, adjusted_selection_index) for a windowed overlay."""
    total = len(overlay_matches)
    if total <= max_visible:
        return overlay_matches, cmd_overlay_idx
    if cmd_overlay_idx < 0:
        return _diversify_overlay(overlay_matches, max_visible), -1
    half = max_visible // 2
    start = cmd_overlay_idx - half
    start = max(0, min(start, total - max_visible))
    end = start + max_visible
    visible = overlay_matches[start:end]
    adj_idx = cmd_overlay_idx - start
    return visible, adj_idx


# ── Mode labels ───────────────────────────────────────────────────────────────

_MODE_LABELS = {
    Mode.NORMAL:      '-- NORMAL --',
    Mode.INSERT:      '-- INSERT --',
    Mode.VISUAL:      '-- VISUAL --',
    Mode.VISUAL_LINE: '-- VISUAL LINE --',
    Mode.COMMAND:     '-- COMMAND --',
    Mode.SEARCH:      '',  # search prompt rendered inline
}


# ── Layout class ──────────────────────────────────────────────────────────────

class Layout:
    """Manages the alternate-screen three-region layout."""

    def __init__(self, rows: int, cols: int):
        self._rows = max(3, rows)
        self._cols = max(1, cols)
        self._input_height = 1

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def content_rows(self) -> int:
        """Number of rows for the conversation content viewport."""
        return max(1, self._rows - 1 - self._input_height)

    @property
    def status_row(self) -> int:
        """1-indexed row for the status line (always the last row)."""
        return self._rows

    @property
    def input_start_row(self) -> int:
        """1-indexed row where the input area starts."""
        return self._rows - self._input_height

    def update_input_height(self, input_buf: list, prompt_prefix: str, cont_prefix: str) -> None:
        """Recompute input area height based on current buffer content."""
        buf_str = ''.join(input_buf)
        lines = buf_str.split('\n')
        cols = max(1, self._cols)
        total_vrows = sum(
            _count_visual_rows(ln, len(prompt_prefix if i == 0 else cont_prefix), cols)
            for i, ln in enumerate(lines)
        )
        max_input = max(1, self._rows // 3)
        self._input_height = max(1, min(total_vrows, max_input))

    def resize(self, rows: int, cols: int) -> None:
        self._rows = max(3, rows)
        self._cols = max(1, cols)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render_content(
        self,
        lines: list[str],
        content_rows: int,
        cols: int,
        cursor_row: int | None = None,
        selection: tuple | None = None,
        search_matches: 'set[int] | None' = None,
        viewport_offset: int = 0,
    ) -> None:
        """Render the content viewport directly to stdout.

        *selection* is ``(start_row, end_row, line_mode, start_col, end_col)``
        where rows/cols refer to buffer line indices.
        """
        out = []
        # Unpack selection once
        sel_sr = sel_er = sel_sc = sel_ec = 0
        sel_line_mode = False
        if selection is not None:
            sel_sr, sel_er, sel_line_mode, sel_sc, sel_ec = selection

        for vrow in range(content_rows):
            buf_line_idx = viewport_offset + vrow
            out.append(cur_move(vrow + 1, 1))
            out.append(ERASE_LINE)
            if buf_line_idx < len(lines):
                line = lines[buf_line_idx]

                # Highlight search matches
                if search_matches and buf_line_idx in search_matches:
                    line = SGR_REVERSE_YELLOW + ansi_strip(line) + SGR_RESET
                # Highlight visual selection (overrides search highlight)
                elif selection is not None and sel_sr <= buf_line_idx <= sel_er:
                    plain = ansi_strip(line)
                    if sel_line_mode:
                        # Line-mode: entire line highlighted
                        line = SGR_REVERSE + plain + SGR_RESET
                    elif sel_sr == sel_er:
                        # Single-line character selection
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
                        # Middle lines: fully highlighted
                        line = SGR_REVERSE + plain + SGR_RESET

                out.append(line)
            out.append(SGR_RESET)
        sys.stdout.write(''.join(out))

    def render_status(
        self,
        mode: Mode,
        status_text: str,
        search_buf: list[str] | None = None,
        search_direction: int = 1,
        viewport_offset: int = 0,
        total_lines: int = 0,
        cols: int = 80,
        auto_scroll: bool = True,
        new_content_below: bool = False,
        command_buf: list[str] | None = None,
    ) -> None:
        """Render the status bar at the bottom row."""
        out = [cur_move(self.status_row, 1), ERASE_LINE]

        if mode == Mode.COMMAND:
            # Command mode: show ':' prefix + command text on status line
            cmd_text = ''.join(command_buf) if command_buf else ''
            out.append(f':{cmd_text}')
            out.append(SGR_RESET)
            sys.stdout.write(''.join(out))
            # Position cursor right after the typed text
            sys.stdout.write(cur_move(self.status_row, 2 + len(cmd_text)))
            sys.stdout.write(CUR_SHOW)
            return

        if mode == Mode.SEARCH:
            # Show search prompt in status bar
            prefix = '/' if search_direction == 1 else '?'
            pattern = ''.join(search_buf) if search_buf else ''
            out.append(f'{prefix}{pattern}')
            out.append(SGR_RESET)
            sys.stdout.write(''.join(out))
            # Position cursor right after the typed text
            sys.stdout.write(cur_move(self.status_row, 1 + len(prefix) + len(pattern)))
            sys.stdout.write(CUR_SHOW)
            return

        out.append(SGR_AZURE_ON_DGRAY)
        label = _MODE_LABELS.get(mode, '')
        left = f' {label}  {status_text}' if status_text else f' {label}'

        # Right side: scroll position
        if total_lines > 0:
            pct = min(100, int((viewport_offset + self.content_rows) / total_lines * 100)) if total_lines > 0 else 100
            right = f' {viewport_offset + 1}-{min(viewport_offset + self.content_rows, total_lines)}/{total_lines} ({pct}%) '
        else:
            right = ''

        if new_content_below and not auto_scroll:
            right = ' [new content below] ' + right

        # Pad to fill the row
        pad = cols - len(ansi_strip(left)) - len(ansi_strip(right))
        if pad < 0:
            pad = 0
        status_line = left + ' ' * pad + right

        out.append(status_line[:cols])
        out.append(SGR_RESET)
        sys.stdout.write(''.join(out))

    def render_input(
        self,
        input_buf: list,
        cursor_pos: int,
        mode: Mode,
        prompt_prefix: str,
        cont_prefix: str,
        cols: int,
        cmd_overlay: list | None = None,
        overlay_idx: int = -1,
        command_buf: list | None = None,
    ) -> None:
        """Render the input/command area above the status line."""
        cols = max(1, cols)
        input_start = self.input_start_row

        if mode == Mode.COMMAND:
            # Command mode renders on the status line (bottom row), like vim
            # Clear the input area first
            sys.stdout.write(cur_move(input_start, 1) + ERASE_LINE)
            return

        # Normal / Insert / Visual / Search modes
        buf_str = ''.join(input_buf)
        lines = buf_str.split('\n')
        line_vrows = [
            _count_visual_rows(ln, len(prompt_prefix if i == 0 else cont_prefix), cols)
            for i, ln in enumerate(lines)
        ]

        # Render command completion overlay above the input area
        overlay_rows_rendered = 0
        if cmd_overlay and mode == Mode.INSERT:
            max_overlay = min(6, max(0, self.content_rows - 2))
            visible, vis_sel = _window_overlay(cmd_overlay, overlay_idx, max_overlay)
            overlay_rows_rendered = len(visible)
            for i, name in enumerate(visible):
                row = input_start - overlay_rows_rendered + i
                if row < 1:
                    continue
                style = SGR_BOLD_AZURE if i == vis_sel else SGR_DIM_GRAY
                marker = '\u25b6' if i == vis_sel else ' '
                sys.stdout.write(cur_move(row, 1) + ERASE_LINE +
                                 f'{style} {marker} /{name}{SGR_RESET}')

        # Render prompt lines
        vrow_offset = 0
        for i, line in enumerate(lines):
            prefix = prompt_prefix if i == 0 else cont_prefix
            # Each logical line may span multiple visual rows
            for vr in range(line_vrows[i]):
                abs_row = input_start + vrow_offset
                if abs_row >= self.status_row:
                    break
                sys.stdout.write(cur_move(abs_row, 1) + ERASE_LINE)
                if vr == 0:
                    sys.stdout.write(f'{prefix}{line}')
                vrow_offset += 1

        if mode == Mode.INSERT:
            # Position cursor in prompt area
            cursor_vrow, cursor_col = _cursor_visual_pos(
                input_buf, cursor_pos, prompt_prefix, cont_prefix, cols, line_vrows
            )
            abs_cursor_row = input_start + cursor_vrow
            sys.stdout.write(cur_move(abs_cursor_row, cursor_col + 1))
            sys.stdout.write(CUR_SHOW)

    def position_cursor(self, mode: Mode, cursor_row: int, viewport_offset: int) -> None:
        """Final cursor placement after all regions have been rendered.

        Only NORMAL and VISUAL modes need explicit positioning here (block
        cursor in the content area).  INSERT, COMMAND, and SEARCH modes
        already had their cursor placed by render_input / render_status
        respectively — touching the cursor here would undo that work.
        """
        from .ansi import CURSOR_BLOCK
        if mode in (Mode.NORMAL, Mode.VISUAL, Mode.VISUAL_LINE):
            vrow = cursor_row - viewport_offset
            if 0 <= vrow < self.content_rows:
                sys.stdout.write(cur_move(vrow + 1, 1))
            sys.stdout.write(CURSOR_BLOCK + CUR_SHOW)

    def render_all(
        self,
        buffer_lines: list[str],
        viewport_offset: int,
        mode: Mode,
        status_text: str,
        input_buf: list,
        cursor_pos: int,
        prompt_prefix: str,
        cont_prefix: str,
        search_buf: list[str] | None = None,
        search_direction: int = 1,
        search_matches: list[int] | None = None,
        selection: tuple | None = None,
        total_lines: int = 0,
        cmd_overlay: list | None = None,
        overlay_idx: int = -1,
        command_buf: list | None = None,
        auto_scroll: bool = True,
        new_content_below: bool = False,
        cursor_row: int = 0,
    ) -> None:
        """Full screen redraw (used on resize and initial draw)."""
        self.render_content(
            buffer_lines, self.content_rows, self._cols,
            selection=selection,
            search_matches=search_matches,
            viewport_offset=viewport_offset,
        )
        if mode in (Mode.COMMAND, Mode.SEARCH):
            # Render input first (clears prompt area), then status last
            # so cursor ends up on the status line.
            self.render_input(
                input_buf, cursor_pos, mode,
                prompt_prefix, cont_prefix, self._cols,
            )
            self.render_status(
                mode, status_text,
                search_buf=search_buf,
                search_direction=search_direction,
                viewport_offset=viewport_offset,
                total_lines=total_lines,
                cols=self._cols,
                auto_scroll=auto_scroll,
                new_content_below=new_content_below,
                command_buf=command_buf,
            )
        else:
            # Render status first, then input last so cursor ends up
            # in the prompt area (INSERT) or gets repositioned by
            # position_cursor (NORMAL/VISUAL).
            self.render_status(
                mode, status_text,
                search_buf=search_buf,
                search_direction=search_direction,
                viewport_offset=viewport_offset,
                total_lines=total_lines,
                cols=self._cols,
                auto_scroll=auto_scroll,
                new_content_below=new_content_below,
            )
            self.render_input(
                input_buf, cursor_pos, mode,
                prompt_prefix, cont_prefix, self._cols,
                cmd_overlay=cmd_overlay,
                overlay_idx=overlay_idx,
            )
        self.position_cursor(mode, cursor_row, viewport_offset)
        sys.stdout.flush()
