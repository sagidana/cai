"""Live attach view: a full-screen, read-only mirror of another agent's
conversation.

The caller (tui._attach_agent) owns all wire traffic; this module is pure
presentation. It receives `view` - the already-rendered conversation
(tui._AttachView, the same write(text, kind=, block=) surface Screen offers,
so the shared renderers paint into it) - plus `watch`, the socket carrying
the watched agent's EVENT broadcast, and `drain_fn`, called whenever that
socket is readable (it decodes and appends to the view; False means EOF -
the agent tore down). Nothing here ever writes to the socket, so the watched
agent cannot be driven from this view - read-only by construction.

Keys: j/k scroll (wheel too), gg/G jump (G re-follows the tail), Ctrl-U/D
half-page, Ctrl-K kills the watched agent, ESC/q detaches (it runs on)."""

import select
import shutil
import signal
import sys
import termios
import time
import tty

from ..ansi import (
    ALT_ENTER, ALT_EXIT,
    CUR_HIDE,
    ERASE_SCREEN,
    SGR_AZURE_ON_DGRAY, SGR_RESET,
    SYNC_START, SYNC_END,
    KEY_ESC, KEY_CTRL_C, KEY_CTRL_D, KEY_CTRL_K, KEY_CTRL_U,
    KEY_UP, KEY_DOWN,
)
from ..input import read_key, parse_mouse

# minimum time between streaming repaints, so a fast event burst does not
# repaint the whole viewport once per chunk.
_PAINT_INTERVAL = 0.03


def prompt_attach_overlay(screen, view, *, title, watch=None, drain_fn=None,
                          kill_fn=None):
    """full-screen read-only viewport over `view`, following its tail while
    drain_fn appends. watch is the socket select()ed next to the tty (None
    for an agent that already finished: pure scrollback); kill_fn()
    interrupts the watched agent on Ctrl-K. returns on ESC/q."""
    alive = watch is not None and drain_fn is not None
    follow = True
    offset = 0
    prev_key = ''
    painted = None       # the (version, offset, rows, cols, alive) last drawn
    resize_pending = [False]

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        resize_pending[0] = True

    def _status_row():
        state = 'finished'
        if alive:
            state = 'running'
        left = f' {title} · read-only · {state}'
        hints = 'j/k:scroll G:tail ^K:kill ESC:back '
        pad = max(1, screen._cols - len(left) - len(hints))
        bar = (left + ' ' * pad + hints)[:screen._cols]
        return f'{SGR_AZURE_ON_DGRAY}{bar}{SGR_RESET}'

    def _redraw():
        nonlocal offset, painted
        rows = screen._rows
        cols = screen._cols
        content_rows = max(1, rows - 1)
        total = view.line_count()
        max_offset = max(0, total - content_rows)
        if follow:
            offset = max_offset
        offset = min(max_offset, max(0, offset))
        lines = view.get_lines(offset, content_rows)
        out = [SYNC_START]
        for i in range(content_rows):
            line = ''
            if i < len(lines):
                line = lines[i]
            out.append(f'\033[{i + 1};1H\033[2K{line}{SGR_RESET}')
        out.append(f'\033[{rows};1H\033[2K{_status_row()}')
        out.append(SYNC_END)
        sys.stdout.write(''.join(out))
        sys.stdout.flush()
        painted = (view.version, offset, rows, cols, alive)

    def _stale():
        return painted != (view.version, offset, screen._rows, screen._cols, alive)

    old_attrs = termios.tcgetattr(screen._tty_fd)
    orig_handler = signal.getsignal(signal.SIGWINCH)
    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}{CUR_HIDE}')
    sys.stdout.flush()
    try:
        signal.signal(signal.SIGWINCH, _on_resize)
        tty.setraw(screen._tty_fd)
        _redraw()
        last_paint = time.monotonic()

        while True:
            if resize_pending[0]:
                resize_pending[0] = False
                view.rewrap(screen._cols)

            now = time.monotonic()
            if _stale() and now - last_paint >= _PAINT_INTERVAL:
                _redraw()
                last_paint = now

            channels = [screen._tty_fd]
            if alive:
                channels.append(watch)
            try:
                readable, _, _ = select.select(channels, [], [], 0.05)
            except (OSError, ValueError):
                continue                 # the socket closed under us; re-select

            if alive and watch in readable:
                alive = drain_fn()

            if screen._tty_fd not in readable:
                continue
            key = read_key(screen._tty_fd)

            content_rows = max(1, screen._rows - 1)
            max_offset = max(0, view.line_count() - content_rows)

            mouse = parse_mouse(key)
            if mouse is not None:
                action = mouse[0]
                if action == 'wheel_up':
                    follow = False
                    offset = max(0, offset - 3)
                elif action == 'wheel_down':
                    offset = min(max_offset, offset + 3)
                    follow = offset >= max_offset
                prev_key = ''
                continue

            if key == KEY_ESC or key == 'q' or key == KEY_CTRL_C:
                return
            if key == KEY_CTRL_K:
                if alive and kill_fn is not None:
                    kill_fn()
                prev_key = ''
                continue

            if key in (KEY_DOWN, 'j'):
                offset = min(max_offset, offset + 1)
                follow = offset >= max_offset
            elif key in (KEY_UP, 'k'):
                follow = False
                offset = max(0, offset - 1)
            elif key == 'G':
                follow = True
            elif key == 'g' and prev_key == 'g':
                follow = False
                offset = 0
            elif key == KEY_CTRL_U:
                follow = False
                offset = max(0, offset - max(1, content_rows // 2))
            elif key == KEY_CTRL_D:
                offset = min(max_offset, offset + max(1, content_rows // 2))
                follow = offset >= max_offset
            prev_key = key

    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()
