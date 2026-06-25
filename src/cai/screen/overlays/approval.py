"""Interactive allow/deny confirmation overlay for unsafe tool calls.

A small centered box that shows the tool/command about to run and asks the
user to Allow or Deny. Defaults to Deny (fail-safe). Driven on the main thread
via Screen._service_requests when the LLM worker calls Screen.submit_request.
"""

import select
import shutil
import signal
import sys
import termios
import textwrap
import tty

from ..ansi import (
    ALT_ENTER, ALT_EXIT,
    CUR_HIDE,
    ERASE_SCREEN,
    SGR_RESET, SGR_DIM, SGR_GREEN, SGR_BOLD_RED, SGR_REVERSE,
    cur_move,
    KEY_ENTER, KEY_ESC, KEY_CTRL_C, KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, KEY_TAB,
)
from ..input import read_key


def prompt_approval_overlay(screen, title, body):
    """show a centered allow/deny box and return the decision.
    True to allow, False to deny. h/l/j/k/arrows/Tab move between Allow and
    Deny; Enter confirms the highlighted option; a/y allow outright;
    d/n/Esc/Ctrl-C deny. defaults to Deny."""
    selected = [1]  # 0 = Allow, 1 = Deny — start on the safe choice
    resize_pending = [False]

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        resize_pending[0] = True

    def _redraw():
        _draw(screen._rows, screen._cols, title, body, selected[0])

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
            if key in (KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, KEY_TAB,
                       'h', 'l', 'j', 'k'):
                selected[0] ^= 1
                _redraw()
            elif key in ('a', 'y', 'Y'):
                return True
            elif key in ('d', 'n', 'N'):
                return False
            elif key in KEY_ENTER:
                return selected[0] == 0
            elif key in (KEY_ESC, KEY_CTRL_C):
                return False
    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()


def _draw(rows, cols, title, body, selected):
    """render the centered allow/deny box (full redraw, no diffing)."""
    H = '─'
    TL = '┌'
    TR = '┐'
    BL = '└'
    BR = '┘'
    VL = '│'
    ML = '├'
    MR = '┤'

    inner_w = max(30, min(int(cols * 0.8), 100))
    box_w = inner_w + 2
    h_line = H * inner_w

    # wrap the command body to the inner width (1-col left padding).
    wrapped = []
    for line in (body.splitlines() or ['']):
        wrapped.extend(textwrap.wrap(line, inner_w - 2) or [''])
    max_body = max(3, rows - 11)
    if len(wrapped) > max_body:
        wrapped = wrapped[:max_body] + ['  … (truncated)']

    body_h = len(wrapped)
    box_h = body_h + 7  # top, title, sep, body, sep, buttons, hint, bottom
    start_r = max(1, (rows - box_h) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)

    out = [ERASE_SCREEN]

    def put(row_off, text):
        r = start_r + row_off
        if 1 <= r <= rows:
            out.append(f'{cur_move(r, start_c)}{text}')

    # top border + title
    put(0, f'{TL}{h_line}{TR}')
    t = title[:inner_w - 1]
    put(1, f'{VL} {SGR_BOLD_RED}{t}{SGR_RESET}{" " * (inner_w - 1 - len(t))}{VL}')
    put(2, f'{ML}{h_line}{MR}')

    # Command body
    for i, line in enumerate(wrapped):
        cell = (' ' + line)[:inner_w].ljust(inner_w)
        put(3 + i, f'{VL}{cell}{VL}')

    sep = 3 + body_h
    put(sep, f'{ML}{h_line}{MR}')

    # Buttons
    allow = ' Allow (a) '
    deny = ' Deny (d) '
    if selected == 0:
        allow_s = f'{SGR_REVERSE}{SGR_GREEN}{allow}{SGR_RESET}'
        deny_s = deny
    else:
        allow_s = allow
        deny_s = f'{SGR_REVERSE}{SGR_BOLD_RED}{deny}{SGR_RESET}'
    gap = 4
    plain_w = 2 + len(allow) + gap + len(deny)
    btn_pad = max(0, inner_w - plain_w)
    put(sep + 1, f'{VL}  {allow_s}{" " * gap}{deny_s}{" " * btn_pad}{VL}')

    # Hint
    hint = ' ←/→ move · enter confirm · esc/ctrl-c deny'
    hint = hint[:inner_w]
    put(sep + 2, f'{VL}{SGR_DIM}{hint}{SGR_RESET}{" " * (inner_w - len(hint))}{VL}')
    put(sep + 3, f'{BL}{h_line}{BR}')

    out.append(CUR_HIDE)
    sys.stdout.write(''.join(out))
    sys.stdout.flush()
