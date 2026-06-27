"""Single-line text-input overlay for a hook's ctx.ui.text(...).

A centered box with a title and one editable line. Driven on the main thread via
Screen._service_requests when the served LLM worker calls Screen.submit_request.
Enter submits (an empty line yields the supplied default); Esc / Ctrl-C cancels
and returns None; secret=True masks the typed characters.
"""

import select
import shutil
import signal
import sys
import termios
import tty

from ..ansi import (
    ALT_ENTER, ALT_EXIT,
    CUR_HIDE,
    ERASE_SCREEN,
    SGR_RESET, SGR_DIM, SGR_BOLD, SGR_REVERSE,
    cur_move,
    KEY_ENTER, KEY_ESC, KEY_CTRL_C, KEY_BACKSPACE,
)
from ..input import read_key


def prompt_text_overlay(screen, title, default="", secret=False):
    """show a centered single-line input box and return the entered string. an
    empty entry returns `default`; Esc / Ctrl-C returns None."""
    buf = []
    resize_pending = [False]

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        resize_pending[0] = True

    def _redraw():
        _draw(screen._rows, screen._cols, title, buf, secret)

    old_attrs = termios.tcgetattr(screen._tty_fd)
    orig_handler = signal.getsignal(signal.SIGWINCH)
    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()
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
            if key in KEY_ENTER:
                text = ''.join(buf)
                if text == '':
                    return default
                return text
            if key in (KEY_ESC, KEY_CTRL_C):
                return None
            if key == KEY_BACKSPACE:
                if buf:
                    buf.pop()
                _redraw()
                continue
            if len(key) == 1 and key >= ' ':
                buf.append(key)
                _redraw()
    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()


def _draw(rows, cols, title, buf, secret):
    """render the centered input box (full redraw, no diffing)."""
    H = '─'
    TL, TR, BL, BR, VL, ML, MR = '┌', '┐', '└', '┘', '│', '├', '┤'

    inner_w = max(30, min(int(cols * 0.8), 100))
    box_w = inner_w + 2
    h_line = H * inner_w

    if secret:
        shown = '*' * len(buf)
    else:
        shown = ''.join(buf)
    field = '> ' + shown
    caret = f'{SGR_REVERSE} {SGR_RESET}'
    visible = field[-(inner_w - 3):]
    pad = ' ' * max(0, inner_w - 1 - len(visible) - 1)

    box_h = 7
    start_r = max(1, (rows - box_h) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)

    out = [ERASE_SCREEN]

    def put(row_off, text):
        r = start_r + row_off
        if 1 <= r <= rows:
            out.append(f'{cur_move(r, start_c)}{text}')

    t = title[:inner_w - 1]
    put(0, f'{TL}{h_line}{TR}')
    put(1, f'{VL} {SGR_BOLD}{t}{SGR_RESET}{" " * (inner_w - 1 - len(t))}{VL}')
    put(2, f'{ML}{h_line}{MR}')
    put(3, f'{VL} {visible}{caret}{pad}{VL}')
    put(4, f'{ML}{h_line}{MR}')
    hint = ' enter submit · esc cancel'[:inner_w]
    put(5, f'{VL}{SGR_DIM}{hint}{SGR_RESET}{" " * (inner_w - len(hint))}{VL}')
    put(6, f'{BL}{h_line}{BR}')

    out.append(CUR_HIDE)
    sys.stdout.write(''.join(out))
    sys.stdout.flush()
