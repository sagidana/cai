"""Vim ``:undotree``-style history browser for MessageHistoryTracker.

Left pane: 2D grid of all recorded snapshots (rows = chronological
depth, columns = branches). Forks render as a new column to the right
of the spine, joined by a ``──.`` connector at the fork point.
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
    KEY_CTRL_D, KEY_CTRL_U, KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
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


def _ordered_ids(tracker) -> list[int]:
    """Pre-order DFS of node ids — used for absolute clamps and HEAD lookup."""
    _coords, _forks, ordered, _max_col = tracker.layout()
    return ordered


def _move_vertical(display_rows: list, rows_to_cells: dict, coords: dict,
                   current_nid: int, direction: int) -> int:
    """j/k: step to the next/prev tree-row, preferring the same column."""
    cur_row, cur_col = coords[current_nid]
    cur_idx = display_rows.index(cur_row)
    new_idx = cur_idx + direction
    if new_idx < 0 or new_idx >= len(display_rows):
        return current_nid
    cells = rows_to_cells[display_rows[new_idx]]
    for c, n in cells:
        if c == cur_col:
            return n
    return min(cells, key=lambda x: abs(x[0] - cur_col))[1]


def _move_horizontal(rows_to_cells: dict, coords: dict,
                     current_nid: int, direction: int) -> int:
    """h/l: step to the nearest node to the left/right at the same tree-row."""
    cur_row, cur_col = coords[current_nid]
    cells = sorted(rows_to_cells[cur_row])
    if direction > 0:
        for c, n in cells:
            if c > cur_col:
                return n
    else:
        for c, n in reversed(cells):
            if c < cur_col:
                return n
    return current_nid


def _render_grid_row(tree_row: int,
                     rows_to_cells: dict,
                     forks: dict,
                     selected_nid: int,
                     head_nid: int,
                     max_col: int,
                     tracker,
                     cell_w: int) -> str:
    """Render one row of the grid layout.

    Each branch column occupies ``cell_w`` chars. Cells show
    ``●N label`` (``> N label`` for HEAD), with the label truncated to
    fit. Parents that fork emit ``─╮`` connectors from the end of
    their cell text into the new column(s) so the branch visually drops
    down to the next row. Selection gets SGR_REVERSE on the cell text;
    HEAD gets SGR_BOLD. Connectors and padding stay unstyled.
    """
    total_w = (max_col + 1) * cell_w
    chars = [' '] * total_w
    spans: list[tuple[int, int, str]] = []
    text_lens: dict[int, int] = {}

    cells = rows_to_cells.get(tree_row, [])

    for col, nid in cells:
        node = tracker.node(nid)
        marker = '>' if nid == head_nid else '●'
        prefix = f'{marker}{nid}'
        # Reserve 1 char gap before the next column; whatever space is
        # left after marker+id+space goes to the label.
        label_room = max(0, cell_w - 1 - len(prefix) - 1)
        label = (node.label or '')[:label_room]
        text = f'{prefix} {label}'.rstrip() if label else prefix
        text = text[:cell_w - 1]
        cs = col * cell_w
        for i, ch in enumerate(text):
            if cs + i < total_w:
                chars[cs + i] = ch
        text_lens[col] = len(text)
        end = min(cs + len(text), total_w)
        if nid == selected_nid:
            spans.append((cs, end, 'sel'))
        elif nid == head_nid:
            spans.append((cs, end, 'head'))

    for col, nid in cells:
        if nid not in forks:
            continue
        p_end = col * cell_w + text_lens.get(col, 0)
        for target_col in sorted(forks[nid]):
            t_start = target_col * cell_w
            for i in range(p_end, min(t_start, total_w)):
                if chars[i] == ' ':
                    chars[i] = '─'
            if t_start < total_w:
                chars[t_start] = '╮'
            p_end = t_start + 1

    plain = ''.join(chars)
    if not spans:
        return plain
    spans.sort(key=lambda s: -s[0])
    out = plain
    for start, end, style in spans:
        if style == 'sel':
            prefix, suffix = SGR_REVERSE, SGR_RESET
        else:
            prefix, suffix = SGR_BOLD, SGR_RESET
        out = out[:start] + prefix + out[start:end] + suffix + out[end:]
    return out


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

    coords, forks, ordered, max_col = tracker.layout()
    if not ordered:
        return
    if state['selected_i'] >= len(ordered):
        state['selected_i'] = len(ordered) - 1

    # Cell width adapts to the left pane and how many branch columns we
    # need to fit. Wider when there are few branches (so labels breathe);
    # narrower when many branches need to share the pane.
    cell_w = max(7, min(16, left_w // (max_col + 1)))

    # Group nodes by tree-row so each display row can place every node
    # at that depth across its branch columns.
    rows_to_cells: dict[int, list[tuple[int, int]]] = {}
    for nid in ordered:
        r, c = coords[nid]
        rows_to_cells.setdefault(r, []).append((c, nid))
    display_rows = sorted(rows_to_cells.keys())

    selected_nid = ordered[state['selected_i']]
    selected_disp_row = display_rows.index(coords[selected_nid][0])

    if selected_disp_row >= state['scroll'] + visible_n:
        state['scroll'] = selected_disp_row - visible_n + 1
    if selected_disp_row < state['scroll']:
        state['scroll'] = selected_disp_row
    state['scroll'] = max(0, min(state['scroll'], max(0, len(display_rows) - visible_n)))

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

    sel_node = tracker.node(selected_nid)
    head_nid = tracker.head()
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

        # Left (tree) cell — one tree-row, all columns at this depth.
        if src >= len(display_rows):
            left_styled = ' ' * left_w
        else:
            tree_row = display_rows[src]
            left_text = _render_grid_row(
                tree_row, rows_to_cells, forks,
                selected_nid, head_nid, max_col,
                tracker, cell_w,
            )
            left_styled = ansi_pad(left_text, left_w)

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
    age = _format_age(max(0.0, now - sel_node.ts))
    ctx_str = ''
    ctx_size = state.get('context_size') or 0
    if ctx_size:
        chars = sum(len(str(m.get('content', ''))) for m in sel_node.snapshot)
        rough_tok = chars // 4
        ctx_str = f'  ctx {rough_tok / ctx_size:.0%} (~{rough_tok}/{ctx_size})'
    status_left = (f' #{sel_node.id} {sel_node.label}  {age}  '
                   f'msgs={len(sel_node.snapshot)}  diff={diff or "·"}'
                   f'{ctx_str}')
    hints = (f'  {SGR_DIM_GRAY}h/j/k/l  Enter:jump  F:fork  u/^R:undo/redo  '
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


def prompt_history_overlay(screen, tracker, *,
                           context_size: int = 0) -> bool:
    """Interactive undo-tree viewer. Mutates tracker/messages in place on jump.

    Returns ``True`` when the user pressed ``F`` to fork at the selected
    node (tracker has already been jumped). The caller should cascade any
    wrapping overlay closed and let the CLI auto-continue the agentic
    loop if appropriate. Returns ``False`` on a normal ESC/q exit.

    ``context_size`` (in tokens) lets the status bar surface a live
    ``ctx N% (~tokens/limit)`` indicator for whichever snapshot is
    selected, computed by the same chars/4 heuristic used elsewhere.
    Pass 0 to suppress.
    """
    if tracker is None:
        return False

    state = {
        'selected_i': 0,
        'scroll': 0,
        'prev_lines': {},
        'first_draw': True,
        'resize_pending': False,
        'fork_requested': False,
        'context_size': context_size,
    }
    # Start with cursor on the current HEAD node.
    ordered = _ordered_ids(tracker)
    for i, nid in enumerate(ordered):
        if nid == tracker.head():
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

            coords, _forks_nav, ordered, _max_col_nav = tracker.layout()
            if not ordered:
                break
            rows_to_cells_nav: dict = {}
            for nid in ordered:
                r, c = coords[nid]
                rows_to_cells_nav.setdefault(r, []).append((c, nid))
            display_rows_nav = sorted(rows_to_cells_nav.keys())

            def _select(nid: int) -> None:
                state['selected_i'] = ordered.index(nid)

            cur_nid = ordered[state['selected_i']]

            if key in (KEY_DOWN, 'j'):
                _select(_move_vertical(display_rows_nav, rows_to_cells_nav,
                                        coords, cur_nid, +1))
            elif key in (KEY_UP, 'k'):
                _select(_move_vertical(display_rows_nav, rows_to_cells_nav,
                                        coords, cur_nid, -1))
            elif key in (KEY_RIGHT, 'l'):
                _select(_move_horizontal(rows_to_cells_nav, coords, cur_nid, +1))
            elif key in (KEY_LEFT, 'h'):
                _select(_move_horizontal(rows_to_cells_nav, coords, cur_nid, -1))
            elif key == 'G':
                state['selected_i'] = len(ordered) - 1
            elif key == 'g':
                state['selected_i'] = 0
            elif key == KEY_CTRL_D:
                state['selected_i'] = min(len(ordered) - 1, state['selected_i'] + 10)
            elif key == KEY_CTRL_U:
                state['selected_i'] = max(0, state['selected_i'] - 10)
            elif key in KEY_ENTER:
                node = tracker.node(ordered[state['selected_i']])
                tracker.jump(node.id)
            elif key == 'F':
                # Fork: jump to the selected snapshot AND cascade the
                # overlay exit so the CLI lands in the main view and
                # continues the conversation from there.
                node = tracker.node(ordered[state['selected_i']])
                tracker.jump(node.id)
                state['fork_requested'] = True
                break
            elif key == 'u':
                if tracker.undo():
                    ordered = _ordered_ids(tracker)
                    for i, nid in enumerate(ordered):
                        if nid == tracker.head():
                            state['selected_i'] = i
                            break
            elif key == '\x12':  # Ctrl-R
                if tracker.redo():
                    ordered = _ordered_ids(tracker)
                    for i, nid in enumerate(ordered):
                        if nid == tracker.head():
                            state['selected_i'] = i
                            break
            elif key == 'd':
                node = tracker.node(ordered[state['selected_i']])
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
                    ordered = _ordered_ids(tracker)
                    if state['selected_i'] >= len(ordered):
                        state['selected_i'] = max(0, len(ordered) - 1)

            _draw(tracker, screen, state)

    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()

    return state['fork_requested']
