"""Interactive model picker overlay (alternate screen, fuzzy finder)."""

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
    SGR_RESET, SGR_REVERSE, SGR_DIM_GRAY, SGR_BOLD_AZURE,
    cur_move,
    KEY_BACKSPACE, KEY_ESC, KEY_ENTER, KEY_CTRL_C,
    KEY_CTRL_D, KEY_CTRL_U, KEY_UP, KEY_DOWN,
)
from ..input import read_key


# ── Fuzzy matching ───────────────────────────────────────────────────────────

def _fuzzy_match(pattern: str, text: str) -> tuple:
    """Simple fuzzy match: every character of pattern must appear in text in order.

    Returns (matched: bool, score: int, match_positions: list[int]).
    Lower score is better.  Score penalises gaps between matched characters.
    """
    if not pattern:
        return True, 0, []

    pat = pattern.lower()
    txt = text.lower()
    positions = []
    ti = 0
    for ch in pat:
        found = txt.find(ch, ti)
        if found == -1:
            return False, 0, []
        positions.append(found)
        ti = found + 1

    # Score: sum of gaps between consecutive matches + start offset
    score = positions[0]
    for i in range(1, len(positions)):
        score += positions[i] - positions[i - 1] - 1
    return True, score, positions


def _filter_and_sort(models: list, pattern: str) -> list:
    """Return models matching pattern, sorted by fuzzy score.

    Each element is (model_name, match_positions).
    """
    if not pattern:
        return [(m, []) for m in models]

    results = []
    for m in models:
        matched, score, positions = _fuzzy_match(pattern, m)
        if matched:
            results.append((score, m, positions))
    results.sort(key=lambda t: t[0])
    return [(m, pos) for _, m, pos in results]


# ── Rendering ────────────────────────────────────────────────────────────────

def _highlight_name(name: str, positions: list, inner_w: int, is_selected: bool) -> str:
    """Render model name with fuzzy-matched characters highlighted."""
    display = name[:inner_w - 4]  # leave room for "  " prefix + padding
    if not positions:
        padded = f'  {display}'.ljust(inner_w)
        if is_selected:
            return f'{SGR_REVERSE}{padded}{SGR_RESET}'
        return padded

    pos_set = set(positions)
    parts = []
    parts.append('  ')
    for i, ch in enumerate(display):
        if i in pos_set:
            if is_selected:
                parts.append(f'{SGR_RESET}{SGR_BOLD_AZURE}{ch}{SGR_RESET}{SGR_REVERSE}')
            else:
                parts.append(f'{SGR_BOLD_AZURE}{ch}{SGR_RESET}')
        else:
            parts.append(ch)
    raw = ''.join(parts)
    # Pad to inner_w using visual length (name chars + 2 prefix)
    pad_needed = inner_w - len(display) - 2
    padding = ' ' * max(0, pad_needed)
    if is_selected:
        return f'{SGR_REVERSE}{raw}{padding}{SGR_RESET}'
    return f'{raw}{padding}'


def _draw_model_overlay(
    rows: int,
    cols: int,
    filtered: list,
    selected_idx: int,
    search_buf: list,
    total_count: int,
    prev_lines: dict,
    first_draw: bool,
) -> None:
    """Render the model picker overlay."""
    n = len(filtered)

    inner_w = max(20, int(cols * 0.95) - 2)
    box_w = inner_w + 2

    overhead = 4  # top border + separator + search line + bottom border
    max_box_h = max(overhead + 1, int(rows * 0.95))
    visible_n = max_box_h - overhead
    box_h = visible_n + overhead

    start_r = max(1, (rows - box_h) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)

    # Scroll so selected item is visible
    scroll = 0
    if selected_idx >= visible_n:
        scroll = selected_idx - visible_n + 1
    scroll = max(0, min(scroll, max(0, n - visible_n)))

    H = '─'
    TL, TR = '┌', '┐'
    BL, BR = '└', '┘'
    VL = '│'
    ML, MR = '├', '┤'
    h_line = H * inner_w

    new_lines: dict[int, tuple] = {}

    def put(row_off: int, text: str) -> None:
        r = start_r + row_off
        if 1 <= r <= rows:
            new_lines[row_off] = (r, text)

    # Top border
    put(0, f'{TL}{h_line}{TR}')

    # Model list
    for i in range(visible_n):
        ai = i + scroll
        if ai >= n:
            put(1 + i, f'{VL}{" " * inner_w}{VL}')
            continue
        model_name, positions = filtered[ai]
        is_sel = (ai == selected_idx)
        cell = _highlight_name(model_name, positions, inner_w, is_sel)
        put(1 + i, f'{VL}{cell}{VL}')

    # Separator
    put(1 + visible_n, f'{ML}{h_line}{MR}')

    # Search / status line
    search_text = ''.join(search_buf)
    match_info = f'{n}/{total_count}' if search_text else f'{total_count} models'
    prompt_str = f' > {search_text}'
    right_str = f'{match_info} '
    gap = inner_w - len(prompt_str) - len(right_str)
    if gap < 0:
        # Truncate search text if needed
        prompt_str = prompt_str[:inner_w - len(right_str) - 1]
        gap = 0
    status_cell = f'{prompt_str}{" " * gap}{SGR_DIM_GRAY}{right_str}{SGR_RESET}'
    put(1 + visible_n + 1, f'{VL}{status_cell}{VL}')

    # Bottom border
    put(1 + visible_n + 2, f'{BL}{h_line}{BR}')

    # Diff draw
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

    # Position cursor in search field
    cursor_col = start_c + 1 + 3 + len(search_text)  # border + " > " + text
    cursor_row = start_r + 1 + visible_n + 1
    out.append(f'{CUR_SHOW}{cur_move(cursor_row, cursor_col)}')

    sys.stdout.write(''.join(out))
    sys.stdout.flush()


# ── Event loop ───────────────────────────────────────────────────────────────

def prompt_model_overlay(screen, models: list) -> str | None:
    """Interactive model picker. Returns selected model name or None on cancel."""
    if not models:
        return None

    models = sorted(models)
    search_buf: list[str] = []
    filtered = _filter_and_sort(models, '')
    selected_idx = 0
    prev_lines: dict = {}
    first_draw = [True]
    resize_pending = [False]

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        resize_pending[0] = True
        first_draw[0] = True

    def _redraw():
        _draw_model_overlay(
            screen._rows, screen._cols,
            filtered, selected_idx, search_buf,
            len(models), prev_lines, first_draw[0],
        )
        first_draw[0] = False

    old_attrs = termios.tcgetattr(screen._tty_fd)
    orig_handler = signal.getsignal(signal.SIGWINCH)
    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()

    result = None
    try:
        signal.signal(signal.SIGWINCH, _on_resize)
        tty.setraw(screen._tty_fd)
        _redraw()

        while True:
            if resize_pending[0]:
                resize_pending[0] = False
                _redraw()

            rlist, _, _ = select.select([screen._tty_fd], [], [], 0.05)
            if not rlist:
                continue
            key = read_key(screen._tty_fd)

            # Cancel
            if key == KEY_ESC or key == KEY_CTRL_C:
                break

            # Confirm selection
            if key in KEY_ENTER:
                if filtered and 0 <= selected_idx < len(filtered):
                    result = filtered[selected_idx][0]
                break

            # Navigation
            if key == KEY_UP:
                if selected_idx > 0:
                    selected_idx -= 1
                _redraw()
                continue

            if key == KEY_DOWN:
                if filtered and selected_idx < len(filtered) - 1:
                    selected_idx += 1
                _redraw()
                continue

            if key == KEY_CTRL_U:
                overhead = 4
                vis = max(5, int(screen._rows * 0.95)) - overhead
                selected_idx = max(0, selected_idx - max(1, vis // 2))
                _redraw()
                continue

            if key == KEY_CTRL_D:
                overhead = 4
                vis = max(5, int(screen._rows * 0.95)) - overhead
                selected_idx = min(max(0, len(filtered) - 1),
                                   selected_idx + max(1, vis // 2))
                _redraw()
                continue

            # Backspace
            if key == KEY_BACKSPACE:
                if search_buf:
                    search_buf.pop()
                    filtered = _filter_and_sort(models, ''.join(search_buf))
                    selected_idx = min(selected_idx, max(0, len(filtered) - 1))
                _redraw()
                continue

            # Printable character — update search
            if len(key) == 1 and ord(key) >= 32:
                search_buf.append(key)
                filtered = _filter_and_sort(models, ''.join(search_buf))
                selected_idx = 0
                _redraw()
                continue

    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()

    return result
