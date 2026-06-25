"""Command palette overlay — fuzzy-searchable commands with one-line help.

Opened with Ctrl-P from NORMAL/INSERT mode (or by submitting an empty ``:``
command). Each item is a ``(name, help)`` pair; typing filters with the same
subsequence fuzzy match used by the model picker. Returns the selected
command name, or None on cancel.
"""

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
from ..input import read_key, parse_mouse


def _fuzzy_match(pattern, text):
    if not pattern:
        return True, 0, []
    pat = pattern.lower()
    txt = text.lower()

    positions = []
    ti = 0
    for ch in pat:
        i = txt.find(ch, ti)
        if i == -1:
            return False, 0, []
        positions.append(i)
        ti = i + 1

    score = positions[0]
    for j in range(1, len(positions)):
        score += positions[j] - positions[j - 1] - 1
    return True, score, positions


def _filter_and_sort(commands, pattern):
    """commands: [(name, help)] -> [(name, help, positions)] filtered/ranked."""
    if not pattern:
        return [(n, h, []) for n, h in commands]
    results = []
    for n, h in commands:
        ok, score, positions = _fuzzy_match(pattern, n)
        if ok:
            results.append((score, n, h, positions))
    results.sort(key=lambda t: (t[0], t[1]))
    return [(n, h, pos) for _, n, h, pos in results]


def _highlight(text, positions, width, selected):
    display = text[:width]
    if not positions:
        padded = f' {display}'.ljust(width)
        if selected:
            return f'{SGR_REVERSE}{padded}{SGR_RESET}'
        return padded

    pos_set = set(positions)
    parts = [' ']
    for i, ch in enumerate(display):
        if i not in pos_set:
            parts.append(ch)
        elif selected:
            parts.append(f'{SGR_RESET}{SGR_BOLD_AZURE}{ch}{SGR_RESET}{SGR_REVERSE}')
        else:
            parts.append(f'{SGR_BOLD_AZURE}{ch}{SGR_RESET}')
    raw = ''.join(parts)
    padding = ' ' * max(0, width - len(display) - 1)
    if selected:
        return f'{SGR_REVERSE}{raw}{padding}{SGR_RESET}'
    return f'{raw}{padding}'


def _draw(rows, cols, filtered, selected, search_buf, total, prev_lines, first_draw):
    n = len(filtered)
    inner_w = max(40, int(cols * 0.7) - 2)
    box_w = inner_w + 2

    overhead = 5  # top + sep + search + hint + bottom
    max_box_h = max(overhead + 1, int(rows * 0.7))
    visible_n = max_box_h - overhead

    start_r = max(1, (rows - (visible_n + overhead)) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)

    scroll = 0
    if selected >= visible_n:
        scroll = selected - visible_n + 1
    scroll = max(0, min(scroll, max(0, n - visible_n)))

    H = '─'
    TL = '┌'
    TR = '┐'
    BL = '└'
    BR = '┘'
    VL = '│'
    ML = '├'
    MR = '┤'
    h_line = H * inner_w

    title = '  Commands  '
    pad_l = max(0, (inner_w - len(title)) // 2)
    pad_r = max(0, inner_w - len(title) - pad_l)

    new_lines = {}

    def put(row_off, text):
        r = start_r + row_off
        if 1 <= r <= rows:
            new_lines[row_off] = (r, text)

    put(0, f'{TL}{H * pad_l}{title}{H * pad_r}{TR}')

    name_col_w = 22
    for i in range(visible_n):
        ai = i + scroll
        if ai >= n:
            put(1 + i, f'{VL}{" " * inner_w}{VL}')
            continue
        name, help_text, positions = filtered[ai]
        is_sel = (ai == selected)
        name_cell = _highlight(name, positions, name_col_w, is_sel)
        desc_w = max(4, inner_w - name_col_w - 2)
        desc_text = (help_text or '')[:desc_w]
        if is_sel:
            line = (f'{name_cell}{SGR_REVERSE} {desc_text}'
                    f'{" " * (desc_w - len(desc_text))} {SGR_RESET}')
        else:
            line = (f'{name_cell} {SGR_DIM_GRAY}{desc_text}{SGR_RESET}'
                    f'{" " * (desc_w - len(desc_text))} ')
        put(1 + i, f'{VL}{line}{VL}')

    put(1 + visible_n, f'{ML}{h_line}{MR}')

    search_text = ''.join(search_buf)
    match_info = f'{total} commands'
    if search_text:
        match_info = f'{n}/{total}'
    prompt_str = f' > {search_text}'
    right_str = f'{match_info} '
    gap = max(0, inner_w - len(prompt_str) - len(right_str))
    if gap == 0 and len(prompt_str) + len(right_str) > inner_w:
        prompt_str = prompt_str[:inner_w - len(right_str) - 1]
    put(1 + visible_n + 1,
        f'{VL}{prompt_str}{" " * gap}{SGR_DIM_GRAY}{right_str}{SGR_RESET}{VL}')

    hint = ' ↵ run  esc cancel '
    hint_pad = max(0, inner_w - len(hint))
    put(1 + visible_n + 2,
        f'{VL}{SGR_DIM_GRAY}{hint}{SGR_RESET}{" " * hint_pad}{VL}')

    put(1 + visible_n + 3, f'{BL}{h_line}{BR}')

    out = []
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

    cursor_col = start_c + 1 + 3 + len(search_text)
    cursor_row = start_r + 1 + visible_n + 1
    out.append(f'{CUR_SHOW}{cur_move(cursor_row, cursor_col)}')

    sys.stdout.write(''.join(out))
    sys.stdout.flush()


def prompt_palette_overlay(screen, commands):
    """pick a command from commands ([(name, help)]). None on cancel."""
    if not commands:
        return None

    search_buf = []
    filtered = _filter_and_sort(commands, '')
    selected = 0
    prev_lines = {}
    first_draw = [True]
    resize_pending = [False]

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        resize_pending[0] = True
        first_draw[0] = True

    def _redraw():
        _draw(screen._rows, screen._cols, filtered, selected, search_buf,
              len(commands), prev_lines, first_draw[0])
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

            mouse = parse_mouse(key)
            if mouse is not None:
                if mouse[0] == 'wheel_up':
                    selected = max(0, selected - 1)
                elif mouse[0] == 'wheel_down':
                    selected = min(max(0, len(filtered) - 1), selected + 1)
                _redraw()
                continue

            if key == KEY_ESC or key == KEY_CTRL_C:
                break
            if key in KEY_ENTER:
                if filtered and 0 <= selected < len(filtered):
                    result = filtered[selected][0]
                break
            if key in (KEY_UP, '\x0b'):       # up / ctrl-k
                selected = max(0, selected - 1)
                _redraw()
                continue
            if key in (KEY_DOWN, '\x0e'):     # down / ctrl-n
                selected = min(max(0, len(filtered) - 1), selected + 1)
                _redraw()
                continue
            if key in (KEY_CTRL_U, KEY_CTRL_D):
                overhead = 5
                vis = max(5, int(screen._rows * 0.7)) - overhead
                step = max(1, vis // 2)
                if key == KEY_CTRL_U:
                    selected = max(0, selected - step)
                else:
                    selected = min(max(0, len(filtered) - 1), selected + step)
                _redraw()
                continue
            if key == KEY_BACKSPACE:
                if search_buf:
                    search_buf.pop()
                    filtered = _filter_and_sort(commands, ''.join(search_buf))
                    selected = min(selected, max(0, len(filtered) - 1))
                _redraw()
                continue
            if len(key) == 1 and ord(key) >= 32:
                search_buf.append(key)
                filtered = _filter_and_sort(commands, ''.join(search_buf))
                selected = 0
                _redraw()
                continue
    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()

    return result
