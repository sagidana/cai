"""Interactive context message viewer/editor overlay."""

import json as _json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import termios
import tty

from ..ansi import (
    ALT_ENTER, ALT_EXIT,
    CUR_SHOW, CUR_HIDE,
    ERASE_SCREEN,
    SGR_RESET, SGR_REVERSE, SGR_YELLOW, SGR_REVERSE_YELLOW,
    SGR_GREEN, SGR_CYAN, SGR_MAGENTA,
    cur_move,
)
from ..state import (
    _OverlayCtx,
    _overlay_msg_text, _overlay_visible_n,
    _overlay_find_matches, _overlay_sync_search_cursor,
    _overlay_apply_filter, _overlay_recompute_tokens,
)
from ..ansi import ansi_strip
from ..input import read_key


# ── Role color map ────────────────────────────────────────────────────────────
_ROLE_COLOR = {
    'system':    SGR_MAGENTA,
    'user':      SGR_GREEN,
    'assistant': SGR_CYAN,
    'tool':      SGR_YELLOW,
}


# ── Cell / status styling ─────────────────────────────────────────────────────

def _style_context_cell(cell: str, role: str, is_sel: bool, is_match: bool) -> str:
    if is_sel and is_match:
        return f'{SGR_REVERSE_YELLOW}{cell}{SGR_RESET}'
    if is_sel:
        return f'{SGR_REVERSE}{cell}{SGR_RESET}'
    if is_match:
        return f'{SGR_YELLOW}{cell}{SGR_RESET}'
    if role in _ROLE_COLOR:
        return f'{_ROLE_COLOR[role]}{cell}{SGR_RESET}'
    return cell


def _build_context_status(ctx: _OverlayCtx, nv: int, nm: int, inner_w: int) -> tuple:
    """Return (raw_status_str, use_reverse_style)."""
    dir_char = '/' if ctx.search_direction == 1 else '?'

    if ctx.filter_mode:
        filter_text = ''.join(ctx.filter_buf)
        return f' filter: {filter_text}', True

    if ctx.search_mode:
        search_text = ''.join(ctx.search_buf)
        if ctx.search_matches:
            m_info = f' [{ctx.search_match_idx + 1}/{len(ctx.search_matches)}]'
        elif search_text:
            m_info = ' [no match]'
        else:
            m_info = ''
        return f' {dir_char}{search_text}{m_info}', True

    pos_str = f' {ctx.selected_idx + 1}/{nv}' if nv > 0 else ' 0/0'
    if ctx.filter_pattern:
        pos_str += f'  \033[33m[filter: {ctx.filter_pattern}]\033[m  ({nv}/{nm})'
    if ctx.search_pattern:
        m_label = (
            f' [{ctx.search_match_idx + 1}/{len(ctx.search_matches)}]'
            if ctx.search_matches else ''
        )
        pos_str += f'   {dir_char}{ctx.search_pattern}{m_label}'
    hints = '  j/k  /:search  f:filter  p:prune  ↵:nvim  ESC:close'
    if len(ansi_strip(pos_str)) + len(hints) <= inner_w:
        pos_str += hints
    return pos_str, False


# ── Renderer ──────────────────────────────────────────────────────────────────

def draw_context_overlay(ctx: _OverlayCtx, rows: int, cols: int) -> None:
    """Render the centered floating context overlay."""
    messages = ctx.messages
    view     = ctx.view
    nv       = len(view)
    nm       = len(messages)

    idx_w    = max(2, len(str(nm)))
    role_w   = 9
    pct_w    = 6
    prefix_w = 1 + idx_w + 2 + role_w + 2 + pct_w + 2

    overhead  = 4
    max_box_h = max(overhead + 1, int(rows * 0.85))
    visible_n = max(1, min(nm, max_box_h - overhead))
    box_h     = visible_n + overhead

    max_inner_w = max(prefix_w + 10, int(cols * 0.95) - 2)
    inner_w     = max(prefix_w + 10, min(max_inner_w, cols - 4))
    box_w       = inner_w + 2

    start_r = max(1, (rows - box_h) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)

    # Scroll window.
    # forced_scroll is set by zz/zt/zb and kept until the cursor navigates
    # outside the visible window it defines — at that point we fall back to
    # the default "keep selection visible" logic and clear the override.
    if ctx.forced_scroll is not None:
        scroll = max(0, min(ctx.forced_scroll, nv - visible_n))
        if not (scroll <= ctx.selected_idx < scroll + visible_n):
            ctx.forced_scroll = None
    if ctx.forced_scroll is None:
        scroll = max(0, min(
            ctx.selected_idx - visible_n + 1 if ctx.selected_idx >= visible_n else 0,
            nv - visible_n,
        ))

    total_chars = sum(len(_overlay_msg_text(m)) for m in messages) or 1
    content_w   = max(4, inner_w - prefix_w)

    H  = '─'; TL, TR = '┌', '┐'; BL, BR = '└', '┘'
    VL = '│'; ML, MR = '├', '┤'
    h_line = H * inner_w

    if ctx.context_size and ctx.tokens_est:
        pct      = ctx.tokens_est / ctx.context_size * 100
        ctx_info = f'{pct:.0f}% ({ctx.tokens_est}/{ctx.context_size})'
        title    = f'  Context  {ctx_info}  '
    else:
        title = '  Context  '
    title  = title[:inner_w]
    pad_l  = (inner_w - len(title)) // 2
    pad_r  = inner_w - len(title) - pad_l
    title_border = f'{TL}{H * pad_l}{title}{H * pad_r}{TR}'

    new_lines: dict[int, tuple] = {}

    def put(row_off: int, text: str) -> None:
        r = start_r + row_off
        if 1 <= r <= rows:
            new_lines[row_off] = (r, text)

    put(0, title_border)

    for i in range(visible_n):
        pos = i + scroll
        if pos >= nv:
            put(1 + i, f'{VL}{" " * inner_w}{VL}')
            continue
        ai      = view[pos]
        msg     = messages[ai]
        role    = msg.get('role', '?')
        raw     = _overlay_msg_text(msg)
        pct     = len(raw) / total_chars * 100
        idx_str  = str(ai + 1).rjust(idx_w)
        role_str = role[:role_w].ljust(role_w)
        pct_str  = f'{pct:5.1f}%'
        preview  = ansi_strip(raw.replace('\n', ' ').replace('\r', ' '))[:content_w]
        cell     = f' {idx_str}  {role_str}  {pct_str}  {preview}'[:inner_w].ljust(inner_w)
        is_sel   = (pos == ctx.selected_idx)
        is_match = bool(ctx.search_matches) and (pos in ctx.search_matches)
        put(1 + i, f'{VL}{_style_context_cell(cell, role, is_sel, is_match)}{VL}')

    put(1 + visible_n, f'{ML}{h_line}{MR}')

    status_row  = start_r + 1 + visible_n + 1
    raw_status, use_reverse = _build_context_status(ctx, nv, nm, inner_w)

    if use_reverse:
        put(1 + visible_n + 1,
            f'{VL}{SGR_REVERSE}{raw_status[:inner_w].ljust(inner_w)}{SGR_RESET}{VL}')
    else:
        # +30 headroom for embedded ANSI codes that don't count visually
        put(1 + visible_n + 1, f'{VL}{raw_status[:inner_w + 30].ljust(inner_w)}{VL}')

    put(1 + visible_n + 2, f'{BL}{h_line}{BR}')

    out: list[str] = []
    if ctx.first_draw:
        sys.stdout.write(ERASE_SCREEN)
        for row_off, (r, text) in new_lines.items():
            out.append(f'{cur_move(r, start_c)}{text}')
    else:
        for row_off, (r, text) in new_lines.items():
            if ctx.prev_lines.get(row_off) != text:
                out.append(f'{cur_move(r, start_c)}{text}')

    if ctx.filter_mode:
        filter_text = ''.join(ctx.filter_buf)
        cursor_col  = start_c + 1 + len(' filter: ') + len(filter_text)
        out.append(f'{CUR_SHOW}{cur_move(status_row, cursor_col)}')
    elif ctx.search_mode:
        search_text = ''.join(ctx.search_buf)
        cursor_col  = start_c + 1 + 1 + 1 + len(search_text)
        out.append(f'{CUR_SHOW}{cur_move(status_row, cursor_col)}')
    else:
        out.append(CUR_HIDE)

    ctx.prev_lines.clear()
    for row_off, (r, text) in new_lines.items():
        ctx.prev_lines[row_off] = text

    sys.stdout.write(''.join(out))
    sys.stdout.flush()


# ── Overlay redraw wrapper ────────────────────────────────────────────────────

def overlay_redraw(ctx: _OverlayCtx, rows: int, cols: int) -> None:
    draw_context_overlay(ctx, rows, cols)
    ctx.first_draw = False


# ── Key handlers ──────────────────────────────────────────────────────────────

def overlay_filter_key(ctx: _OverlayCtx, key: str) -> None:
    """Handle one keypress while the filter prompt is open."""
    if key in ('\r', '\n'):
        pat = ''.join(ctx.filter_buf)
        if pat and (not ctx.filter_history or ctx.filter_history[0] != pat):
            ctx.filter_history.insert(0, pat)
        _overlay_apply_filter(ctx, pat)
        ctx.filter_mode        = False
        ctx.filter_history_pos = -1
        ctx.first_draw         = True
        return

    if key == '\033':
        ctx.filter_mode        = False
        ctx.filter_history_pos = -1
        ctx.filter_buf[:]      = list(ctx.filter_pattern)
        return

    if key == '\x7f':
        if ctx.filter_buf:
            ctx.filter_buf.pop()
            ctx.filter_history_pos = -1
        else:
            ctx.filter_mode        = False
            ctx.filter_history_pos = -1
            ctx.filter_buf[:]      = list(ctx.filter_pattern)
        return

    if key == '\033[A' and ctx.filter_history:   # up → older history entry
        ctx.filter_history_pos = min(
            ctx.filter_history_pos + 1, len(ctx.filter_history) - 1)
        ctx.filter_buf[:] = list(ctx.filter_history[ctx.filter_history_pos])
        return

    if key == '\033[B':   # down → newer history entry
        if ctx.filter_history_pos > 0:
            ctx.filter_history_pos -= 1
            ctx.filter_buf[:] = list(ctx.filter_history[ctx.filter_history_pos])
        else:
            ctx.filter_history_pos = -1
            ctx.filter_buf[:] = []
        return

    if len(key) == 1 and ord(key) >= 32:
        ctx.filter_buf.append(key)
        ctx.filter_history_pos = -1


def overlay_search_key(ctx: _OverlayCtx, key: str) -> None:
    """Handle one keypress while the search prompt is open."""
    if key in ('\r', '\n'):
        ctx.search_mode = False
        return

    if key == '\033':
        ctx.search_mode      = False
        ctx.selected_idx     = ctx.pre_search_idx
        ctx.search_pattern   = ''
        ctx.search_buf       = []
        ctx.search_matches   = []
        ctx.search_match_idx = -1
        return

    if key == '\x7f':
        if ctx.search_buf:
            ctx.search_buf.pop()
            ctx.search_pattern = ''.join(ctx.search_buf)
            ctx.search_matches = _overlay_find_matches(ctx)
            _overlay_sync_search_cursor(ctx)
        else:
            ctx.search_mode      = False
            ctx.selected_idx     = ctx.pre_search_idx
            ctx.search_pattern   = ''
            ctx.search_matches   = []
            ctx.search_match_idx = -1
        return

    if len(key) == 1 and ord(key) >= 32:
        ctx.search_buf.append(key)
        ctx.search_pattern = ''.join(ctx.search_buf)
        ctx.search_matches = _overlay_find_matches(ctx)
        _overlay_sync_search_cursor(ctx)


def overlay_nav_key(
    ctx: _OverlayCtx, key: str, rows: int, tty_fd: int, cooked_attrs
) -> None:
    """Handle one keypress in normal navigation mode."""
    nv = len(ctx.view)
    if nv > 0:
        ctx.selected_idx = max(0, min(nv - 1, ctx.selected_idx))
    if nv == 0:
        return

    vis  = _overlay_visible_n(ctx, rows)
    half = max(1, vis // 2)
    sel  = ctx.selected_idx
    pk   = ctx.prev_key

    if key in ('\033[A', 'k'):
        ctx.selected_idx = max(0, sel - 1)
    elif key in ('\033[B', 'j'):
        ctx.selected_idx = min(nv - 1, sel + 1)
    elif key == 'G':
        ctx.selected_idx = nv - 1
    elif key == 'g' and pk == 'g':
        ctx.selected_idx = 0
    elif key == '\x15':
        ctx.selected_idx = max(0, sel - half)
    elif key == '\x04':
        ctx.selected_idx = min(nv - 1, sel + half)
    # vim scroll-align — no-op when already at first entry
    elif key == 'z' and pk == 'z' and sel > 0:
        ctx.forced_scroll = max(0, sel - vis // 2)
    elif key == 't' and pk == 'z' and sel > 0:
        ctx.forced_scroll = sel
    elif key == 'b' and pk == 'z' and sel > 0:
        ctx.forced_scroll = max(0, sel - vis + 1)
    elif key == 'f':
        ctx.filter_mode        = True
        ctx.filter_history_pos = -1
        ctx.filter_buf[:]      = list(ctx.filter_pattern)
    elif key in ('/', '?'):
        ctx.search_mode      = True
        ctx.search_direction = 1 if key == '/' else -1
        ctx.search_buf       = []
        ctx.search_pattern   = ''
        ctx.search_matches   = []
        ctx.search_match_idx = -1
        ctx.pre_search_idx   = sel
    elif key == 'n' and ctx.search_matches:
        step = 1 if ctx.search_direction == 1 else -1
        ctx.search_match_idx = (ctx.search_match_idx + step) % len(ctx.search_matches)
        ctx.selected_idx     = ctx.search_matches[ctx.search_match_idx]
    elif key == 'N' and ctx.search_matches:
        step = -1 if ctx.search_direction == 1 else 1
        ctx.search_match_idx = (ctx.search_match_idx + step) % len(ctx.search_matches)
        ctx.selected_idx     = ctx.search_matches[ctx.search_match_idx]
    elif key == 'p':
        ctx.messages[ctx.view[sel]]['content'] = '[pruned by user]'
        _overlay_recompute_tokens(ctx)
    elif key in ('\r', '\n'):
        overlay_edit_in_nvim(ctx, sel, tty_fd, cooked_attrs)


def overlay_edit_in_nvim(ctx: _OverlayCtx, pos: int, tty_fd: int, cooked_attrs) -> None:
    """Open messages[view[pos]] in nvim; write changes back on exit."""
    ai      = ctx.view[pos]
    msg     = ctx.messages[ai]
    content = msg.get('content', '')
    is_json = not isinstance(content, str)
    text    = _json.dumps(content, indent=2) if is_json else (content or '')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        sys.stdout.write(f'{ALT_EXIT}{CUR_SHOW}')
        sys.stdout.flush()
        termios.tcsetattr(tty_fd, termios.TCSADRAIN, cooked_attrs)
        subprocess.run(['nvim', tmp])
        with open(tmp, 'r') as f:
            new_text = f.read()
        if is_json:
            try:
                ctx.messages[ai]['content'] = _json.loads(new_text)
            except Exception:
                ctx.messages[ai]['content'] = new_text
        else:
            ctx.messages[ai]['content'] = new_text
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
        sys.stdout.flush()
        tty.setraw(tty_fd)
        _overlay_recompute_tokens(ctx)


# ── Event loop ────────────────────────────────────────────────────────────────

def prompt_context_overlay(
    screen,
    messages: list,
    context_size: int = 0,
    prompt_tokens: int = 0,
) -> tuple:
    """
    Interactive context viewer/editor.

    Navigation   j/k/↑↓  Ctrl-U/Ctrl-D  gg/G
    Search       /pat  ?pat  then n/N
    Filter       f → enter regex  (flags: ~t type, ~c content, default all)
    Scroll-align zz/zt/zb
    Prune        p — replace selected content with placeholder
    Edit         Enter — open in nvim; changes written back
    Close        ESC / q

    Returns (messages, new_prompt_tokens_estimate).
    """
    if not messages:
        return messages, 0

    ctx = _OverlayCtx(messages, context_size, prompt_tokens)

    old_attrs    = termios.tcgetattr(screen._tty_fd)
    orig_handler = signal.getsignal(signal.SIGWINCH)

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        ctx.resize_pending = True
        ctx.first_draw     = True

    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()
    try:
        signal.signal(signal.SIGWINCH, _on_resize)
        tty.setraw(screen._tty_fd)
        overlay_redraw(ctx, screen._rows, screen._cols)

        while True:
            if ctx.resize_pending:
                ctx.resize_pending = False
                overlay_redraw(ctx, screen._rows, screen._cols)

            key = read_key(screen._tty_fd)

            if ctx.filter_mode:
                overlay_filter_key(ctx, key)
                overlay_redraw(ctx, screen._rows, screen._cols)
                continue

            if ctx.search_mode:
                overlay_search_key(ctx, key)
                overlay_redraw(ctx, screen._rows, screen._cols)
                continue

            if key in ('\033', 'q', '\x03'):
                break

            overlay_nav_key(ctx, key, screen._rows, screen._tty_fd, screen._cooked_attrs)
            ctx.prev_key = key
            overlay_redraw(ctx, screen._rows, screen._cols)

    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()

    return messages, ctx.tokens_est
