"""Interactive tools toggle overlay (alternate screen, floating box)."""

import os
import re
import select
import shutil
import signal
import sys
import termios
import tty

from ..ansi import (
    ALT_ENTER, ALT_EXIT,
    CUR_SHOW, CUR_HIDE,
    ERASE_SCREEN,
    SGR_RESET, SGR_REVERSE, SGR_YELLOW, SGR_REVERSE_YELLOW,
    cur_move,
    KEY_BACKSPACE, KEY_ESC, KEY_ENTER, KEY_CTRL_C,
    KEY_CTRL_D, KEY_CTRL_U, KEY_UP, KEY_DOWN,
)
from ..input import read_key


# ── Search helpers ────────────────────────────────────────────────────────────

def _find_matches(tool_entries: list, pattern: str) -> list:
    """Search over tool names (first element of each entry tuple)."""
    if not pattern:
        return []
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
    return [i for i, (nm, _origin) in enumerate(tool_entries) if rx.search(nm)]


def _nearest_fwd(matches: list, from_idx: int) -> int:
    for i, m in enumerate(matches):
        if m >= from_idx:
            return i
    return 0


def _nearest_bwd(matches: list, from_idx: int) -> int:
    for i in range(len(matches) - 1, -1, -1):
        if matches[i] <= from_idx:
            return i
    return len(matches) - 1


def _sync_cursor(selected_idx, pre_search_idx, search_direction,
                 search_matches) -> tuple:
    """Return (new_selected_idx, new_search_match_idx)."""
    if not search_matches:
        return pre_search_idx, -1
    if search_direction == 1:
        mi = _nearest_fwd(search_matches, pre_search_idx)
    else:
        mi = _nearest_bwd(search_matches, pre_search_idx)
    return search_matches[mi], mi


# ── Cell styling ──────────────────────────────────────────────────────────────

def _style_cell(cell: str, is_sel: bool, is_match: bool) -> str:
    if is_sel and is_match:
        return f'{SGR_REVERSE_YELLOW}{cell}{SGR_RESET}'
    if is_sel:
        return f'{SGR_REVERSE}{cell}{SGR_RESET}'
    if is_match:
        return f'{SGR_YELLOW}{cell}{SGR_RESET}'
    return cell


# ── Status bar text ───────────────────────────────────────────────────────────

def _build_status(
    search_mode: bool, search_buf: list, search_direction: int,
    search_matches: list, search_match_idx: int,
    enabled_count: int, n: int, search_pattern: str,
    inner_w: int,
) -> str:
    dir_char = '/' if search_direction == 1 else '?'

    if search_mode:
        search_text = ''.join(search_buf)
        if search_matches:
            m_info = f' [{search_match_idx + 1}/{len(search_matches)}]'
        elif search_text:
            m_info = ' [no match]'
        else:
            m_info = ''
        return f' {dir_char}{search_text}{m_info}'

    count_str = f' {enabled_count}/{n} enabled'
    if search_pattern:
        m_label = f' [{search_match_idx + 1}/{len(search_matches)}]' if search_matches else ''
        count_str += f'   {dir_char}{search_pattern}{m_label}'
    hints = '  j/k /:search ESC/↵:close'
    if len(count_str) + len(hints) <= inner_w:
        count_str += hints
    return count_str


# ── Diff emit ─────────────────────────────────────────────────────────────────

def _emit_diff(new_lines: dict, prev_lines: dict, start_c: int, first_draw: bool) -> str:
    """Build the escape string to update only changed rows."""
    out: list[str] = []
    if first_draw:
        sys.stdout.write(ERASE_SCREEN)
        for row_off, (r, text) in new_lines.items():
            out.append(f'{cur_move(r, start_c)}{text}')
    else:
        for row_off, (r, text) in new_lines.items():
            if prev_lines.get(row_off) != text:
                out.append(f'{cur_move(r, start_c)}{text}')
    prev_lines.clear()
    for row_off, (r, text) in new_lines.items():
        prev_lines[row_off] = text
    return ''.join(out)


# ── Renderer ──────────────────────────────────────────────────────────────────

def draw_tools_overlay(
    rows: int,
    cols: int,
    tool_entries: list,
    enabled: set,
    selected_idx: int,
    search_pattern: str,
    search_matches: list,
    search_match_idx: int,
    search_mode: bool,
    search_buf: list,
    search_direction: int,
    prev_lines: dict,
    first_draw: bool,
) -> None:
    """Render the centered floating tools overlay (pure draw, no event loop)."""
    n = len(tool_entries)

    inner_w   = max(20, int(cols * 0.95) - 2)
    box_w     = inner_w + 2

    overhead  = 4
    max_box_h = max(overhead + 1, int(rows * 0.95))
    visible_n = max_box_h - overhead
    box_h     = visible_n + overhead

    start_r = max(1, (rows - box_h) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)

    scroll = 0
    if selected_idx >= visible_n:
        scroll = selected_idx - visible_n + 1
    scroll = max(0, min(scroll, n - visible_n))

    H  = '─'; TL, TR = '┌', '┐'; BL, BR = '└', '┘'
    VL = '│'; ML, MR = '├', '┤'
    h_line = H * inner_w

    new_lines: dict[int, tuple] = {}

    def put(row_off: int, text: str) -> None:
        r = start_r + row_off
        if 1 <= r <= rows:
            new_lines[row_off] = (r, text)

    put(0, f'{TL}{h_line}{TR}')

    for i in range(visible_n):
        ai = i + scroll
        if ai >= n:
            put(1 + i, f'{VL}{" " * inner_w}{VL}')
            continue
        nm, origin = tool_entries[ai]
        check      = '[x]' if nm in enabled else '[ ]'
        origin_tag = f'[{origin}]'
        # prefix "  [x] " = 6 chars, gap between name and tag = 2, tag itself
        max_nm_len = max(1, inner_w - 6 - 2 - len(origin_tag))
        nm_display = nm[:max_nm_len] if len(nm) > max_nm_len else nm
        nm_padded  = nm_display.ljust(max_nm_len)
        raw_line   = f'  {check} {nm_padded}  {origin_tag}'
        cell       = raw_line[:inner_w].ljust(inner_w)
        is_sel     = (ai == selected_idx)
        is_match   = bool(search_matches) and (ai in search_matches)
        put(1 + i, f'{VL}{_style_cell(cell, is_sel, is_match)}{VL}')

    put(1 + visible_n, f'{ML}{h_line}{MR}')

    tool_names = [nm for nm, _ in tool_entries]
    enabled_count = sum(1 for nm in tool_names if nm in enabled)
    raw_status = _build_status(
        search_mode, search_buf, search_direction,
        search_matches, search_match_idx,
        enabled_count, n, search_pattern, inner_w,
    )
    status_cell = raw_status[:inner_w].ljust(inner_w)
    if search_mode:
        put(1 + visible_n + 1, f'{VL}{SGR_REVERSE}{status_cell}{SGR_RESET}{VL}')
    else:
        put(1 + visible_n + 1, f'{VL}{status_cell}{VL}')

    put(1 + visible_n + 2, f'{BL}{h_line}{BR}')

    out = _emit_diff(new_lines, prev_lines, start_c, first_draw)

    if search_mode:
        search_text = ''.join(search_buf)
        cursor_col  = start_c + 1 + 1 + 1 + len(search_text)
        cursor_row  = start_r + 1 + visible_n + 1
        out += f'{CUR_SHOW}{cur_move(cursor_row, cursor_col)}'
    else:
        out += CUR_HIDE

    sys.stdout.write(out)
    sys.stdout.flush()


# ── Event loop ────────────────────────────────────────────────────────────────

def prompt_tools_overlay(screen, tool_entries: list, enabled: set) -> set:
    """
    Interactive tools toggle overlay.

    tool_entries : list of (name, origin_label) tuples
    enabled      : set of currently enabled tool names

    Navigation : j / k / arrows / Ctrl-U / Ctrl-D / gg / G
    Toggle     : Space
    Search fwd : /pattern  then n / N to cycle
    Search bwd : ?pattern  then N / n to cycle
    Close      : ESC or Enter
    """
    if not tool_entries:
        return set(enabled)

    enabled = set(enabled)
    selected_idx     = 0
    search_mode      = False
    search_direction = 1
    search_buf: list = []
    search_pattern   = ''
    search_matches: list = []
    search_match_idx = -1
    pre_search_idx   = 0
    prev_lines: dict = {}
    first_draw       = [True]
    resize_pending   = [False]

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        resize_pending[0] = True
        first_draw[0]     = True

    def _redraw():
        draw_tools_overlay(
            screen._rows, screen._cols,
            tool_entries, enabled, selected_idx,
            search_pattern, search_matches, search_match_idx,
            search_mode, search_buf, search_direction,
            prev_lines, first_draw[0],
        )
        first_draw[0] = False

    old_attrs    = termios.tcgetattr(screen._tty_fd)
    orig_handler = signal.getsignal(signal.SIGWINCH)
    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()

    try:
        signal.signal(signal.SIGWINCH, _on_resize)
        tty.setraw(screen._tty_fd)
        _redraw()

        prev_key = ''
        while True:
            if resize_pending[0]:
                resize_pending[0] = False
                _redraw()

            rlist, _, _ = select.select([screen._tty_fd], [], [], 0.05)
            if not rlist:
                continue
            key = read_key(screen._tty_fd)

            if search_mode:
                search_mode, selected_idx, search_pattern, search_buf, \
                    search_matches, search_match_idx = _handle_search_key(
                        key, tool_entries, search_mode, selected_idx,
                        search_pattern, search_buf, search_matches,
                        search_match_idx, pre_search_idx, search_direction,
                    )
                _redraw()
                continue

            done, selected_idx, search_mode, search_direction, search_buf, \
                search_pattern, search_matches, search_match_idx, \
                pre_search_idx = _handle_normal_key(
                    key, prev_key, tool_entries, enabled,
                    selected_idx, search_matches, search_match_idx,
                    search_direction, search_pattern,
                    screen._rows,
                )
            if done:
                break
            prev_key = key
            _redraw()

    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()

    return enabled


def _handle_search_key(
    key, tool_entries, search_mode, selected_idx,
    search_pattern, search_buf, search_matches,
    search_match_idx, pre_search_idx, search_direction,
) -> tuple:
    """Handle one key in search input mode. Returns updated search state tuple."""
    if key in KEY_ENTER:
        return (False, selected_idx, search_pattern, search_buf,
                search_matches, search_match_idx)

    if key == KEY_ESC:
        return (False, pre_search_idx, '', [],  [], -1)

    if key == KEY_BACKSPACE:
        if search_buf:
            search_buf = search_buf[:-1]
            search_pattern = ''.join(search_buf)
            search_matches = _find_matches(tool_entries, search_pattern)
            selected_idx, search_match_idx = _sync_cursor(
                selected_idx, pre_search_idx,
                search_direction, search_matches,
            )
        else:
            return (False, pre_search_idx, '', [], [], -1)
        return (True, selected_idx, search_pattern, search_buf,
                search_matches, search_match_idx)

    if len(key) == 1 and ord(key) >= 32:
        search_buf = search_buf + [key]
        search_pattern = ''.join(search_buf)
        search_matches = _find_matches(tool_entries, search_pattern)
        selected_idx, search_match_idx = _sync_cursor(
            selected_idx, pre_search_idx,
            search_direction, search_matches,
        )

    return (True, selected_idx, search_pattern, search_buf,
            search_matches, search_match_idx)


def _handle_normal_key(
    key, prev_key, tool_entries, enabled,
    selected_idx, search_matches, search_match_idx,
    search_direction, search_pattern,
    rows,
) -> tuple:
    """Handle one key in normal navigation mode.

    Returns (done, selected_idx, search_mode, search_direction, search_buf,
             search_pattern, search_matches, search_match_idx, pre_search_idx).
    """
    n    = len(tool_entries)
    done = False
    search_mode    = False
    search_buf: list = []
    pre_search_idx = selected_idx

    if key in (KEY_ESC, *KEY_ENTER, KEY_CTRL_C):
        done = True

    elif key in (KEY_UP, 'k'):
        selected_idx = max(0, selected_idx - 1)

    elif key in (KEY_DOWN, 'j'):
        selected_idx = min(n - 1, selected_idx + 1)

    elif key == ' ':
        nm = tool_entries[selected_idx][0]
        if nm in enabled:
            enabled.discard(nm)
        else:
            enabled.add(nm)

    elif key in ('/', '?'):
        search_mode      = True
        search_direction = 1 if key == '/' else -1
        search_buf       = []
        search_pattern   = ''
        search_matches   = []
        search_match_idx = -1
        pre_search_idx   = selected_idx

    elif key == 'n' and search_matches:
        step = 1 if search_direction == 1 else -1
        search_match_idx = (search_match_idx + step) % len(search_matches)
        selected_idx     = search_matches[search_match_idx]

    elif key == 'N' and search_matches:
        step = -1 if search_direction == 1 else 1
        search_match_idx = (search_match_idx + step) % len(search_matches)
        selected_idx     = search_matches[search_match_idx]

    elif key == 'G':
        selected_idx = n - 1

    elif key == 'g' and prev_key == 'g':
        selected_idx = 0

    elif key == KEY_CTRL_U:
        overhead = 4
        vis  = max(5, int(rows * 0.95)) - overhead
        selected_idx = max(0, selected_idx - max(1, vis // 2))

    elif key == KEY_CTRL_D:
        overhead = 4
        vis  = max(5, int(rows * 0.95)) - overhead
        selected_idx = min(n - 1, selected_idx + max(1, vis // 2))

    return (done, selected_idx, search_mode, search_direction, search_buf,
            search_pattern, search_matches, search_match_idx, pre_search_idx)
