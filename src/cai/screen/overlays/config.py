"""Interactive session-config overlay (alternate screen, floating box).

a single vim-navigated menu for changing the live session's settings. the
overlay itself is generic: the caller hands it a list of Setting descriptors
(see cli.py) and the overlay renders/edits each by its kind. nothing here knows
where a value actually lives - reads go through setting.get(), writes through
setting.set(), so the same widget edits args, the llm config dict and the api
client alike.

kinds:
  BOOL    - on/off, toggled with space (or h/l)
  CHOICE  - a small fixed option list, cycled in place with h/l
  ENUM    - a large option list, picked via the fuzzy model overlay (Enter)
  INT     - integer, edited inline (Enter); empty input clears to None
  FLOAT   - float, edited inline (Enter); empty input clears to None
  STRING  - free text, edited inline (Enter)

a setting may carry fmt(value) to render its current value and parse(text) is
folded into set() for the editable kinds (set receives the raw input string and
returns an error message or None). restore(value) is the undo path: it writes a
previously-read native value back, bypassing set()'s string parsing.

navigation : j / k / arrows / Ctrl-U / Ctrl-D / gg / G
search     : /pattern (forward) or ?pattern (backward), then n / N to cycle
undo       : u reverts the last change
close      : esc / q"""

import re
import select
import shutil
import signal
import sys
import termios
import tty
from dataclasses import dataclass
from dataclasses import field

from ..ansi import (
    ALT_ENTER, ALT_EXIT,
    CUR_SHOW, CUR_HIDE,
    ERASE_SCREEN,
    SGR_RESET, SGR_REVERSE, SGR_DIM_GRAY, SGR_BOLD_AZURE,
    SGR_YELLOW, SGR_REVERSE_YELLOW,
    cur_move,
    KEY_BACKSPACE, KEY_ESC, KEY_ENTER, KEY_CTRL_C,
    KEY_CTRL_D, KEY_CTRL_U, KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
)
from ..input import read_key, parse_mouse


BOOL = 'bool'
CHOICE = 'choice'
ENUM = 'enum'
INT = 'int'
FLOAT = 'float'
STRING = 'string'

_EDITABLE = (INT, FLOAT, STRING)


@dataclass
class Setting:
    label: str
    kind: str
    get: object            # callable() -> current value
    set: object            # callable(value) -> error string or None
    options: list = field(default_factory=list)   # CHOICE / ENUM
    fmt: object = None     # callable(value) -> display string
    restore: object = None  # callable(value) -> None, the undo writer


def _display_value(setting):
    value = setting.get()
    if setting.fmt is not None:
        return setting.fmt(value)
    if setting.kind == BOOL:
        if value:
            return 'on'
        return 'off'
    if value is None:
        return ''
    return str(value)


def _revert(setting, value):
    """write a previously-read native value back (undo)."""
    if setting.restore is not None:
        setting.restore(value)
        return
    setting.set(value)


def _hints(setting, editing):
    if editing:
        return '↵ save  esc cancel'
    if setting.kind == BOOL:
        return 'j/k ␣ toggle  / search  u undo  esc'
    if setting.kind == CHOICE:
        return 'j/k  h/l change  / search  u undo  esc'
    if setting.kind == ENUM:
        return 'j/k  ↵ choose  / search  u undo  esc'
    return 'j/k  ↵ edit  / search  u undo  esc'


def _find_matches(settings, pattern):
    """indices of settings whose label matches pattern (regex, ci)."""
    if not pattern:
        return []
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
    matches = []
    for i, setting in enumerate(settings):
        if rx.search(setting.label):
            matches.append(i)
    return matches


def _nearest_fwd(matches, from_idx):
    for i, m in enumerate(matches):
        if m >= from_idx:
            return i
    return 0


def _nearest_bwd(matches, from_idx):
    for i in range(len(matches) - 1, -1, -1):
        if matches[i] <= from_idx:
            return i
    return len(matches) - 1


def _sync_cursor(from_idx, direction, matches):
    """return (selected_idx, search_match_idx) for the nearest live match."""
    if not matches:
        return from_idx, -1
    if direction == 1:
        mi = _nearest_fwd(matches, from_idx)
    else:
        mi = _nearest_bwd(matches, from_idx)
    return matches[mi], mi


def _emit_diff(new_lines, prev_lines, start_c, first_draw):
    """build the escape string to update only changed rows."""
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
    return ''.join(out)


def _label_width(settings):
    width = 0
    for s in settings:
        if len(s.label) > width:
            width = len(s.label)
    return min(width, 28)


def _draw(rows,
          cols,
          settings,
          selected_idx,
          editing,
          edit_buf,
          edit_error,
          search_mode,
          search_buf,
          search_direction,
          search_matches,
          search_match_idx,
          prev_lines,
          first_draw):
    n = len(settings)
    label_w = _label_width(settings)

    inner_w = max(44, int(cols * 0.6) - 2)
    box_w = inner_w + 2

    overhead = 4   # top + sep + status + bottom
    max_box_h = max(overhead + 1, int(rows * 0.8))
    visible_n = min(n, max_box_h - overhead)
    box_h = visible_n + overhead

    start_r = max(1, (rows - box_h) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)

    scroll = 0
    if selected_idx >= visible_n:
        scroll = selected_idx - visible_n + 1
    scroll = max(0, min(scroll, n - visible_n))

    H = '─'
    TL = '┌'
    TR = '┐'
    BL = '└'
    BR = '┘'
    VL = '│'
    ML = '├'
    MR = '┤'
    h_line = H * inner_w

    title = '  Settings  '
    pad_l = max(0, (inner_w - len(title)) // 2)
    pad_r = max(0, inner_w - len(title) - pad_l)

    new_lines = {}

    def put(row_off, text):
        r = start_r + row_off
        if 1 <= r <= rows:
            new_lines[row_off] = (r, text)

    put(0, f'{TL}{H * pad_l}{title}{H * pad_r}{TR}')

    value_w = max(4, inner_w - label_w - 4)
    for i in range(visible_n):
        ai = i + scroll
        if ai >= n:
            put(1 + i, f'{VL}{" " * inner_w}{VL}')
            continue
        setting = settings[ai]
        is_sel = (ai == selected_idx)
        is_edit = is_sel and editing
        is_match = bool(search_matches) and (ai in search_matches)

        if is_edit:
            value_text = ''.join(edit_buf)
        else:
            value_text = _display_value(setting)
        value_text = value_text[:value_w]

        label_cell = setting.label[:label_w].ljust(label_w)
        raw = f'  {label_cell}  {value_text}'
        cell = raw[:inner_w].ljust(inner_w)

        if is_edit:
            put(1 + i, f'{VL}{cell}{VL}')
        elif is_sel and is_match:
            put(1 + i, f'{VL}{SGR_REVERSE_YELLOW}{cell}{SGR_RESET}{VL}')
        elif is_sel:
            put(1 + i, f'{VL}{SGR_REVERSE}{cell}{SGR_RESET}{VL}')
        elif is_match:
            put(1 + i, f'{VL}{SGR_YELLOW}{cell}{SGR_RESET}{VL}')
        else:
            tinted = f'  {label_cell}  {SGR_BOLD_AZURE}{value_text}{SGR_RESET}'
            pad = ' ' * max(0, inner_w - len(raw))
            put(1 + i, f'{VL}{tinted}{pad}{VL}')

    put(1 + visible_n, f'{ML}{h_line}{MR}')

    if search_mode:
        dir_char = '/'
        if search_direction == -1:
            dir_char = '?'
        search_text = ''.join(search_buf)
        info = ''
        if search_matches:
            info = f' [{search_match_idx + 1}/{len(search_matches)}]'
        elif search_text:
            info = ' [no match]'
        status = f' {dir_char}{search_text}{info}'
        status_cell = status[:inner_w].ljust(inner_w)
        put(1 + visible_n + 1, f'{VL}{SGR_REVERSE}{status_cell}{SGR_RESET}{VL}')
    else:
        status = _hints(settings[selected_idx], editing)
        if edit_error:
            status = f'{status}   {edit_error}'
        status_cell = status[:inner_w].ljust(inner_w)
        put(1 + visible_n + 1, f'{VL}{SGR_DIM_GRAY}{status_cell}{SGR_RESET}{VL}')

    put(1 + visible_n + 2, f'{BL}{h_line}{BR}')

    out = _emit_diff(new_lines, prev_lines, start_c, first_draw)

    if editing:
        cursor_col = start_c + 1 + 2 + label_w + 2 + len(edit_buf)
        cursor_row = start_r + 1 + (selected_idx - scroll)
        out += f'{CUR_SHOW}{cur_move(cursor_row, cursor_col)}'
    elif search_mode:
        cursor_col = start_c + 1 + 2 + len(''.join(search_buf))
        cursor_row = start_r + 1 + visible_n + 1
        out += f'{CUR_SHOW}{cur_move(cursor_row, cursor_col)}'
    else:
        out += CUR_HIDE

    sys.stdout.write(out)
    sys.stdout.flush()


def _cycle(setting, step):
    """move a CHOICE setting step places through its options (clamped)."""
    if not setting.options:
        return
    current = _display_value(setting)
    idx = 0
    if current in setting.options:
        idx = setting.options.index(current)
    idx = max(0, min(len(setting.options) - 1, idx + step))
    setting.set(setting.options[idx])


def _pick_enum(screen, setting, first_draw):
    """open the fuzzy picker for an ENUM setting. the picker exits the
    alternate screen on close, so re-enter it and force a full redraw."""
    if not setting.options:
        return
    picked = screen.prompt_model_overlay(setting.options)
    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()
    first_draw[0] = True
    if picked is None:
        return
    setting.set(picked)


def prompt_config_overlay(screen, settings):
    """interactive session-config menu. mutates settings in place via their
    set() callbacks; returns True if anything changed."""
    if not settings:
        return False

    selected_idx = 0
    editing = False
    edit_buf = []
    edit_error = ''
    changed = False
    undo_stack = []

    search_mode = False
    search_direction = 1
    search_buf = []
    search_pattern = ''
    search_matches = []
    search_match_idx = -1
    pre_search_idx = 0

    prev_lines = {}
    first_draw = [True]
    resize_pending = [False]

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        resize_pending[0] = True
        first_draw[0] = True

    def _redraw():
        _draw(screen._rows, screen._cols,
              settings, selected_idx,
              editing, edit_buf, edit_error,
              search_mode, search_buf, search_direction,
              search_matches, search_match_idx,
              prev_lines, first_draw[0])
        first_draw[0] = False

    old_attrs = termios.tcgetattr(screen._tty_fd)
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

            mouse = parse_mouse(key)
            if mouse is not None:
                if not editing:
                    n = len(settings)
                    if mouse[0] == 'wheel_up':
                        selected_idx = max(0, selected_idx - 1)
                    elif mouse[0] == 'wheel_down':
                        selected_idx = min(max(0, n - 1), selected_idx + 1)
                    _redraw()
                continue

            if editing:
                setting = settings[selected_idx]
                if key in KEY_ENTER:
                    before = setting.get()
                    err = setting.set(''.join(edit_buf))
                    if err:
                        edit_error = err
                    else:
                        editing = False
                        edit_error = ''
                        if setting.get() != before:
                            undo_stack.append((setting, before))
                            changed = True
                    _redraw()
                    continue
                if key == KEY_ESC:
                    editing = False
                    edit_error = ''
                    _redraw()
                    continue
                if key == KEY_BACKSPACE:
                    if edit_buf:
                        edit_buf.pop()
                    _redraw()
                    continue
                if len(key) == 1 and ord(key) >= 32:
                    edit_buf.append(key)
                    _redraw()
                    continue
                continue

            if search_mode:
                if key in KEY_ENTER:
                    search_mode = False
                    _redraw()
                    continue
                if key == KEY_ESC:
                    search_mode = False
                    selected_idx = pre_search_idx
                    search_pattern = ''
                    search_buf = []
                    search_matches = []
                    search_match_idx = -1
                    _redraw()
                    continue
                if key == KEY_BACKSPACE:
                    if not search_buf:
                        search_mode = False
                        selected_idx = pre_search_idx
                        search_pattern = ''
                        search_matches = []
                        search_match_idx = -1
                        _redraw()
                        continue
                    search_buf.pop()
                    search_pattern = ''.join(search_buf)
                    search_matches = _find_matches(settings, search_pattern)
                    selected_idx, search_match_idx = _sync_cursor(
                        pre_search_idx, search_direction, search_matches)
                    _redraw()
                    continue
                if len(key) == 1 and ord(key) >= 32:
                    search_buf.append(key)
                    search_pattern = ''.join(search_buf)
                    search_matches = _find_matches(settings, search_pattern)
                    selected_idx, search_match_idx = _sync_cursor(
                        pre_search_idx, search_direction, search_matches)
                    _redraw()
                    continue
                continue

            setting = settings[selected_idx]
            last_key = prev_key
            prev_key = key

            if key in (KEY_ESC, KEY_CTRL_C) or key == 'q':
                break

            if key in (KEY_UP, 'k'):
                selected_idx = (selected_idx - 1) % len(settings)
                _redraw()
                continue

            if key in (KEY_DOWN, 'j'):
                selected_idx = (selected_idx + 1) % len(settings)
                _redraw()
                continue

            if key == 'G':
                selected_idx = len(settings) - 1
                _redraw()
                continue

            if key == 'g' and last_key == 'g':
                selected_idx = 0
                _redraw()
                continue

            if key in (KEY_CTRL_U, KEY_CTRL_D):
                step = max(1, len(settings) // 2)
                if key == KEY_CTRL_U:
                    selected_idx = max(0, selected_idx - step)
                else:
                    selected_idx = min(len(settings) - 1, selected_idx + step)
                _redraw()
                continue

            if key in ('/', '?'):
                search_mode = True
                search_direction = 1
                if key == '?':
                    search_direction = -1
                search_buf = []
                search_pattern = ''
                search_matches = []
                search_match_idx = -1
                pre_search_idx = selected_idx
                _redraw()
                continue

            if key == 'n' and search_matches:
                search_match_idx = (search_match_idx + search_direction) % len(search_matches)
                selected_idx = search_matches[search_match_idx]
                _redraw()
                continue

            if key == 'N' and search_matches:
                search_match_idx = (search_match_idx - search_direction) % len(search_matches)
                selected_idx = search_matches[search_match_idx]
                _redraw()
                continue

            if key == 'u':
                if undo_stack:
                    target, before = undo_stack.pop()
                    _revert(target, before)
                    selected_idx = settings.index(target)
                    changed = True
                _redraw()
                continue

            if setting.kind == BOOL:
                before = setting.get()
                if key in (' ', *KEY_ENTER):
                    setting.set(not setting.get())
                elif key in ('l', KEY_RIGHT):
                    setting.set(True)
                elif key in ('h', KEY_LEFT):
                    setting.set(False)
                if setting.get() != before:
                    undo_stack.append((setting, before))
                    changed = True
                _redraw()
                continue

            if setting.kind == CHOICE:
                before = setting.get()
                if key in ('l', KEY_RIGHT, ' ', *KEY_ENTER):
                    _cycle(setting, 1)
                elif key in ('h', KEY_LEFT):
                    _cycle(setting, -1)
                if setting.get() != before:
                    undo_stack.append((setting, before))
                    changed = True
                _redraw()
                continue

            if setting.kind == ENUM:
                if key in KEY_ENTER:
                    before = setting.get()
                    _pick_enum(screen, setting, first_draw)
                    if setting.get() != before:
                        undo_stack.append((setting, before))
                        changed = True
                _redraw()
                continue

            if setting.kind in _EDITABLE:
                if key in KEY_ENTER or key == 'i':
                    editing = True
                    edit_error = ''
                    edit_buf = list(_display_value(setting))
                _redraw()
                continue

    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()

    return changed
