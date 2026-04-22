"""Vim ``:undotree``-style history browser for MessageHistoryTracker.

Left pane: ASCII tree of all recorded snapshots (pre-order walk).
Right pane: preview of the selected node's last few messages.

Keys:
  j/k         move through the tree
  Enter       jump to the selected snapshot (rewrites messages[])
  u / ^R      undo/redo along the current branch
  d           drop the selected node and its descendants
  ESC / q     close
"""

import select
import shutil
import signal
import sys
import termios
import time
import tty

from ..ansi import (
    ALT_ENTER, ALT_EXIT,
    CUR_SHOW, CUR_HIDE,
    ERASE_SCREEN,
    SGR_RESET, SGR_REVERSE, SGR_YELLOW, SGR_DIM_GRAY,
    SGR_GREEN, SGR_CYAN, SGR_MAGENTA, SGR_BOLD,
    ansi_strip, ansi_pad,
    cur_move,
    KEY_BACKSPACE, KEY_ESC, KEY_ENTER, KEY_CTRL_C,
    KEY_CTRL_D, KEY_CTRL_U, KEY_UP, KEY_DOWN,
)
from ..input import read_key
from ..state import _overlay_msg_text


_ROLE_COLOR = {
    'system':    SGR_MAGENTA,
    'user':      SGR_GREEN,
    'assistant': SGR_CYAN,
    'tool':      SGR_YELLOW,
}


def _format_age(ts_delta: float) -> str:
    """Short human-readable age (seconds ago, m, h)."""
    if ts_delta < 60:
        return f'{int(ts_delta)}s'
    if ts_delta < 3600:
        return f'{int(ts_delta // 60)}m'
    return f'{int(ts_delta // 3600)}h'


def _preview_lines(snapshot: list[dict], width: int, max_lines: int) -> list[str]:
    """One-line-per-message preview, colored by role."""
    lines = []
    # Show the tail of the conversation (last max_lines messages).
    tail = snapshot[-max_lines:] if len(snapshot) > max_lines else snapshot
    for i, msg in enumerate(tail):
        abs_i = (len(snapshot) - len(tail)) + i
        role = msg.get('role', '?')
        color = _ROLE_COLOR.get(role, '')
        text = ansi_strip(_overlay_msg_text(msg).replace('\n', ' ').replace('\r', ' '))
        raw = f'#{abs_i} {role[:9].ljust(9)} {text}'
        raw = raw[:width]
        lines.append(f'{color}{raw}{SGR_RESET}' if color else raw)
    return lines


def _diff_summary(parent_snap: list[dict] | None, snap: list[dict]) -> str:
    """Short diff indicator against the parent snapshot: +N/-M."""
    if parent_snap is None:
        return ''
    a, b = len(parent_snap), len(snap)
    if a == b:
        # Same length → count content differences
        diffs = sum(1 for x, y in zip(parent_snap, snap) if x != y)
        if diffs:
            return f'~{diffs}'
        return ''
    if a < b:
        return f'+{b - a}'
    return f'-{a - b}'


def _draw(tracker, screen, state) -> None:
    rows, cols = screen._rows, screen._cols

    overhead = 4  # title + middle separator + status + bottom
    max_box_h = max(overhead + 1, int(rows * 0.9))
    visible_n = max_box_h - overhead
    box_h = visible_n + overhead

    box_w = max(50, int(cols * 0.95))
    # inner_w is the width between the outer │ │ frame.
    inner_w = box_w - 2
    # Split inner into left (tree), 1 char separator, right (preview).
    left_w = max(28, inner_w // 2 - 1)
    right_w = max(10, inner_w - left_w - 1)
    # Re-normalise box_w so left_w + 1 + right_w == inner_w exactly.
    inner_w = left_w + 1 + right_w
    box_w = inner_w + 2

    start_r = max(1, (rows - box_h) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)

    walk = tracker.walk()
    nodes = [n for _, n in walk]
    if not nodes:
        return
    if state['selected_i'] >= len(nodes):
        state['selected_i'] = len(nodes) - 1

    if state['selected_i'] >= state['scroll'] + visible_n:
        state['scroll'] = state['selected_i'] - visible_n + 1
    if state['selected_i'] < state['scroll']:
        state['scroll'] = state['selected_i']
    state['scroll'] = max(0, min(state['scroll'], max(0, len(walk) - visible_n)))

    H = '─'
    TL, TR, BL, BR = '┌', '┐', '└', '┘'
    VL = '│'
    TD, TU = '┬', '┴'

    # Top border: title centered in the *left* pane, preview label centered
    # in the right pane, split T between them. Centering over the full inner
    # width would run straight through the split column and clobber a
    # character of the title.
    left_title = '  History  '[:left_w]
    l_pad_l = max(0, (left_w - len(left_title)) // 2)
    l_pad_r = max(0, left_w - len(left_title) - l_pad_l)
    right_title = '  preview  '[:right_w]
    r_pad_l = max(0, (right_w - len(right_title)) // 2)
    r_pad_r = max(0, right_w - len(right_title) - r_pad_l)
    title_border = (
        TL
        + H * l_pad_l + left_title + H * l_pad_r
        + TD
        + H * r_pad_l + right_title + H * r_pad_r
        + TR
    )

    mid_chars = ['├']
    for i in range(inner_w):
        mid_chars.append(TU if i == left_w else H)
    mid_chars.append('┤')
    mid_border = ''.join(mid_chars)

    bottom_chars = [BL]
    for i in range(inner_w):
        bottom_chars.append(H)
    bottom_chars.append(BR)
    bottom_border = ''.join(bottom_chars)

    sel_node = nodes[state['selected_i']]
    preview_rows = _preview_lines(sel_node.snapshot, right_w - 1, visible_n)

    new_lines: dict[int, tuple] = {}
    now = time.monotonic()

    def put(row_off: int, text: str) -> None:
        r = start_r + row_off
        if 1 <= r <= rows:
            new_lines[row_off] = (r, text)

    put(0, title_border)

    for i in range(visible_n):
        src = i + state['scroll']

        # Left (tree) cell
        if src >= len(walk):
            left_styled = ' ' * left_w
        else:
            depth, node = walk[src]
            is_head = (node.id == tracker.head())
            is_cursor = (src == state['selected_i'])
            gutter = '  ' * depth + '● '
            age = _format_age(max(0.0, now - node.ts))
            label = node.label[:30]
            head_tag = ' ← HEAD' if is_head else ''
            left_text = f'{gutter}#{node.id} {label}  {age}{head_tag}'
            left_padded = ansi_pad(left_text, left_w)
            if is_cursor:
                left_styled = f'{SGR_REVERSE}{left_padded}{SGR_RESET}'
            elif is_head:
                left_styled = f'{SGR_BOLD}{left_padded}{SGR_RESET}'
            else:
                left_styled = left_padded

        # Right (preview) cell
        preview_line = preview_rows[i] if i < len(preview_rows) else ''
        right_padded = ansi_pad(' ' + preview_line, right_w)

        put(1 + i, f'{VL}{left_styled}{VL}{right_padded}{VL}')

    # Middle separator (split T-piece pointing up so the bottom is unified).
    put(1 + visible_n, mid_border)

    # Status spans the full inner width.
    parent_snap = (tracker.node(sel_node.parent).snapshot
                   if sel_node.parent is not None else None)
    diff = _diff_summary(parent_snap, sel_node.snapshot)
    status_left = (f' #{sel_node.id} {sel_node.label}  '
                   f'msgs={len(sel_node.snapshot)}  diff={diff or "·"}')
    hints = (f'  {SGR_DIM_GRAY}j/k  Enter:jump  u/^R:undo/redo  '
             f'd:drop  ESC{SGR_RESET}')
    status = status_left + hints
    put(1 + visible_n + 1, f'{VL}{ansi_pad(status, inner_w)}{VL}')

    put(1 + visible_n + 2, bottom_border)

    out: list[str] = []
    if state['first_draw']:
        sys.stdout.write(ERASE_SCREEN)
        for row_off, (r, text) in new_lines.items():
            out.append(f'{cur_move(r, start_c)}{text}')
        state['first_draw'] = False
    else:
        prev = state['prev_lines']
        for row_off, (r, text) in new_lines.items():
            if prev.get(row_off) != text:
                out.append(f'{cur_move(r, start_c)}{text}')
    state['prev_lines'] = {k: v[1] for k, v in new_lines.items()}

    out.append(CUR_HIDE)
    sys.stdout.write(''.join(out))
    sys.stdout.flush()


def prompt_history_overlay(screen, tracker) -> None:
    """Interactive undo-tree viewer. Mutates tracker/messages in place on jump."""
    if tracker is None:
        return

    state = {
        'selected_i': 0,
        'scroll': 0,
        'prev_lines': {},
        'first_draw': True,
        'resize_pending': False,
    }
    # Start with cursor on the current HEAD node.
    walk = tracker.walk()
    for i, (_d, n) in enumerate(walk):
        if n.id == tracker.head():
            state['selected_i'] = i
            break

    old_attrs = termios.tcgetattr(screen._tty_fd)
    orig_handler = signal.getsignal(signal.SIGWINCH)

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        state['resize_pending'] = True
        state['first_draw'] = True

    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()

    try:
        signal.signal(signal.SIGWINCH, _on_resize)
        tty.setraw(screen._tty_fd)
        _draw(tracker, screen, state)

        while True:
            if state['resize_pending']:
                state['resize_pending'] = False
                _draw(tracker, screen, state)

            rlist, _, _ = select.select([screen._tty_fd], [], [], 0.05)
            if not rlist:
                continue
            key = read_key(screen._tty_fd)

            if key in (KEY_ESC, 'q', KEY_CTRL_C):
                break

            walk = tracker.walk()
            if not walk:
                break

            if key in (KEY_DOWN, 'j'):
                state['selected_i'] = min(len(walk) - 1, state['selected_i'] + 1)
            elif key in (KEY_UP, 'k'):
                state['selected_i'] = max(0, state['selected_i'] - 1)
            elif key == 'G':
                state['selected_i'] = len(walk) - 1
            elif key == 'g':
                state['selected_i'] = 0
            elif key == KEY_CTRL_D:
                state['selected_i'] = min(len(walk) - 1, state['selected_i'] + 10)
            elif key == KEY_CTRL_U:
                state['selected_i'] = max(0, state['selected_i'] - 10)
            elif key in KEY_ENTER:
                _d, node = walk[state['selected_i']]
                tracker.jump(node.id)
            elif key == 'u':
                if tracker.undo():
                    # Reposition cursor on new HEAD
                    walk = tracker.walk()
                    for i, (_d, n) in enumerate(walk):
                        if n.id == tracker.head():
                            state['selected_i'] = i
                            break
            elif key == '\x12':  # Ctrl-R
                if tracker.redo():
                    walk = tracker.walk()
                    for i, (_d, n) in enumerate(walk):
                        if n.id == tracker.head():
                            state['selected_i'] = i
                            break
            elif key == 'd':
                _d, node = walk[state['selected_i']]
                # Can't drop root; can't drop HEAD (confuses the current-list
                # invariant). Refuse silently for both.
                if node.parent is not None and node.id != tracker.head():
                    # Collect subtree ids and drop them. We reach into the
                    # tracker internals for this — it's the only caller.
                    stack = [node.id]
                    victims = []
                    while stack:
                        nid = stack.pop()
                        cur = tracker._nodes.get(nid)
                        if cur is None:
                            continue
                        victims.append(nid)
                        stack.extend(cur.children)
                    for nid in victims:
                        tracker._drop(nid)
                    # Clamp cursor if it pointed into the dropped subtree.
                    walk = tracker.walk()
                    if state['selected_i'] >= len(walk):
                        state['selected_i'] = max(0, len(walk) - 1)

            _draw(tracker, screen, state)

    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()
