"""Interactive messages overlay — vim-style viewer/editor for messages[].

Adds folding (per-message + assistant/tool pair), multi-message selection
(marks + visual-line), pipe-through-transform (``>``), ad-hoc LLM transform
(``!``), and in-overlay undo/redo that rides on the same MessageHistoryTracker
that backs the :history overlay. Sibling to :context; do not remove the
context overlay.
"""

import ast
import copy
import json as _json
import os
import select
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
    SGR_GREEN, SGR_CYAN, SGR_MAGENTA, SGR_DIM_GRAY, SGR_BOLD,
    ansi_strip, ansi_pad,
    cur_move,
    KEY_BACKSPACE, KEY_ESC, KEY_ENTER, KEY_CTRL_C,
    KEY_CTRL_D, KEY_CTRL_U, KEY_UP, KEY_DOWN,
)
from ..state import (
    _MsgOverlayCtx,
    _msg_header_preview, _msg_body_lines,
    _msg_is_folded, _msg_effective_selection,
    _overlay_msg_text, _overlay_find_matches, _overlay_sync_search_cursor,
    _overlay_apply_filter, _overlay_recompute_tokens,
)
from ..input import read_key


_ROLE_COLOR = {
    'system':    SGR_MAGENTA,
    'user':      SGR_GREEN,
    'assistant': SGR_CYAN,
    'tool':      SGR_YELLOW,
}


# ── Row building ──────────────────────────────────────────────────────────────

def _build_rows(ctx: _MsgOverlayCtx, inner_w: int, prefix_w: int) -> list:
    """Flatten ctx.view into display rows with folding applied.

    One message = one header row. Expanded messages additionally emit one
    body row per wrapped content line.
    """
    rows = []
    content_w = max(4, inner_w - prefix_w)
    for pos, ai in enumerate(ctx.view):
        msg = ctx.messages[ai]
        folded = _msg_is_folded(ctx, ai)
        glyph = '▶' if folded else '▼'
        preview = _msg_header_preview(msg, content_w - 2)
        rows.append((pos, ai, 'header', f'{glyph} {preview}'))
        if not folded:
            for line in _msg_body_lines(msg, content_w - 2):  # 2 for indent
                rows.append((pos, ai, 'body', line))
    return rows


# ── Status line ───────────────────────────────────────────────────────────────

def _build_status(ctx: _MsgOverlayCtx, nv: int, nm: int, inner_w: int,
                  sel_count: int) -> tuple:
    """Return (raw_status, use_reverse)."""
    if ctx.filter_mode:
        return f' filter: {"".join(ctx.filter_buf)}', True
    if ctx.search_mode:
        dir_char = '/' if ctx.search_direction == 1 else '?'
        text = ''.join(ctx.search_buf)
        if ctx.search_matches:
            info = f' [{ctx.search_match_idx + 1}/{len(ctx.search_matches)}]'
        elif text:
            info = ' [no match]'
        else:
            info = ''
        return f' {dir_char}{text}{info}', True
    if ctx.instruction_mode:
        return f' !> {"".join(ctx.instruction_buf)}', True
    if ctx.status_flash:
        return f' {ctx.status_flash}', True

    pos_str = f' {ctx.selected_idx + 1}/{nv}' if nv > 0 else ' 0/0'
    if sel_count:
        vis_tag = ' V' if ctx.visual_mode else ''
        pos_str += f'  {SGR_YELLOW}[sel:{sel_count}{vis_tag}]{SGR_RESET}'
    if ctx.filter_pattern:
        pos_str += f'  {SGR_YELLOW}[filter: {ctx.filter_pattern}]{SGR_RESET}  ({nv}/{nm})'
    if ctx.search_pattern:
        dir_char = '/' if ctx.search_direction == 1 else '?'
        m = f' [{ctx.search_match_idx + 1}/{len(ctx.search_matches)}]' if ctx.search_matches else ''
        pos_str += f'   {dir_char}{ctx.search_pattern}{m}'
    hints = '  Tab:fold  m/V:sel  >:fx  !:llm  gh:hist  ESC:close'
    if len(ansi_strip(pos_str)) + len(hints) <= inner_w:
        pos_str += hints
    return pos_str, False


# ── Renderer ──────────────────────────────────────────────────────────────────

def _style_cell(text: str, role: str, is_sel: bool, is_match: bool, in_sel: bool) -> str:
    if is_sel and is_match:
        return f'{SGR_REVERSE_YELLOW}{text}{SGR_RESET}'
    if is_sel:
        return f'{SGR_REVERSE}{text}{SGR_RESET}'
    if is_match:
        return f'{SGR_YELLOW}{text}{SGR_RESET}'
    color = _ROLE_COLOR.get(role, '')
    if in_sel:
        # Subtle indicator for non-cursor selected rows: dim reverse
        return f'{SGR_REVERSE}{SGR_DIM_GRAY}{text}{SGR_RESET}'
    return f'{color}{text}{SGR_RESET}' if color else text


def draw_messages_overlay(ctx: _MsgOverlayCtx, rows: int, cols: int) -> None:
    messages = ctx.messages
    nv = len(ctx.view)
    nm = len(messages)

    idx_w = max(2, len(str(max(nm, 1))))
    role_w = 9
    mark_w = 1  # * or space
    prefix_w = 1 + idx_w + 1 + role_w + 1 + mark_w + 1

    overhead = 4
    max_box_h = max(overhead + 1, int(rows * 0.95))
    visible_n = max_box_h - overhead
    box_h = visible_n + overhead

    max_inner_w = max(prefix_w + 10, int(cols * 0.95) - 2)
    inner_w = max_inner_w
    box_w = inner_w + 2

    start_r = max(1, (rows - box_h) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)

    all_rows = _build_rows(ctx, inner_w, prefix_w)
    # Find the row index of the selected message's header so scroll keeps
    # the cursor visible. Falls back to 0 if no rows.
    cursor_row_i = 0
    for i, (pos, ai, kind, _t) in enumerate(all_rows):
        if pos == ctx.selected_idx and kind == 'header':
            cursor_row_i = i
            break

    total_rows = len(all_rows)
    # Scroll
    if ctx.forced_scroll is not None:
        scroll = max(0, min(ctx.forced_scroll, max(0, total_rows - visible_n)))
        if not (scroll <= cursor_row_i < scroll + visible_n):
            ctx.forced_scroll = None
    if ctx.forced_scroll is None:
        scroll = ctx.scroll
        scroll = max(scroll, cursor_row_i - visible_n + 1)
        scroll = min(scroll, cursor_row_i)
        scroll = max(0, min(scroll, max(0, total_rows - visible_n)))
        ctx.scroll = scroll

    effective_sel = set(_msg_effective_selection(ctx))

    # Pre-compute search-match positions for highlight lookup
    match_positions = set(ctx.search_matches)

    # Box glyphs
    H = '─'; TL, TR = '┌', '┐'; BL, BR = '└', '┘'
    VL = '│'; ML, MR = '├', '┤'
    h_line = H * inner_w

    if ctx.context_size and ctx.tokens_est:
        pct = ctx.tokens_est / ctx.context_size * 100
        title = f'  Messages  {pct:.0f}% ({ctx.tokens_est}/{ctx.context_size})  '
    else:
        title = '  Messages  '
    title = title[:inner_w]
    pad_l = (inner_w - len(title)) // 2
    pad_r = inner_w - len(title) - pad_l
    title_border = f'{TL}{H * pad_l}{title}{H * pad_r}{TR}'

    new_lines: dict[int, tuple] = {}

    def put(row_off: int, text: str) -> None:
        r = start_r + row_off
        if 1 <= r <= rows:
            new_lines[row_off] = (r, text)

    put(0, title_border)

    for i in range(visible_n):
        src = i + scroll
        if src >= total_rows:
            put(1 + i, f'{VL}{" " * inner_w}{VL}')
            continue
        pos, ai, kind, text = all_rows[src]
        msg = messages[ai]
        role = msg.get('role', '?')
        is_cursor = (pos == ctx.selected_idx)
        is_match = (pos in match_positions)
        in_sel = (ai in effective_sel) and not is_cursor

        if kind == 'header':
            idx_str = str(ai + 1).rjust(idx_w)
            role_str = role[:role_w].ljust(role_w)
            mark_ch = '*' if ai in ctx.marks else ' '
            prefix = f' {idx_str} {role_str} {mark_ch} '
            cell = (prefix + text)[:inner_w].ljust(inner_w)
            styled = _style_cell(cell, role, is_cursor, is_match, in_sel)
            put(1 + i, f'{VL}{styled}{VL}')
        else:
            indent = '   ' + ' ' * (idx_w + role_w + mark_w)
            body = (indent + text)[:inner_w].ljust(inner_w)
            # Body rows never render with cursor highlight; selection dim only.
            styled = _style_cell(body, role, False, False, in_sel)
            put(1 + i, f'{VL}{styled}{VL}')

    put(1 + visible_n, f'{ML}{h_line}{MR}')

    raw_status, use_reverse = _build_status(ctx, nv, nm, inner_w, len(effective_sel))
    if use_reverse:
        put(1 + visible_n + 1,
            f'{VL}{SGR_REVERSE}{ansi_pad(raw_status, inner_w)}{SGR_RESET}{VL}')
    else:
        put(1 + visible_n + 1, f'{VL}{ansi_pad(raw_status, inner_w)}{VL}')

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

    status_row = start_r + 1 + visible_n + 1
    if ctx.filter_mode:
        filter_text = ''.join(ctx.filter_buf)
        cursor_col = start_c + 1 + len(' filter: ') + len(filter_text)
        out.append(f'{CUR_SHOW}{cur_move(status_row, cursor_col)}')
    elif ctx.search_mode:
        text = ''.join(ctx.search_buf)
        cursor_col = start_c + 1 + 2 + len(text)
        out.append(f'{CUR_SHOW}{cur_move(status_row, cursor_col)}')
    elif ctx.instruction_mode:
        text = ''.join(ctx.instruction_buf)
        cursor_col = start_c + 1 + len(' !> ') + len(text)
        out.append(f'{CUR_SHOW}{cur_move(status_row, cursor_col)}')
    else:
        out.append(CUR_HIDE)

    ctx.prev_lines.clear()
    for row_off, (r, text) in new_lines.items():
        ctx.prev_lines[row_off] = text

    sys.stdout.write(''.join(out))
    sys.stdout.flush()


def overlay_redraw(ctx: _MsgOverlayCtx, rows: int, cols: int) -> None:
    draw_messages_overlay(ctx, rows, cols)
    ctx.first_draw = False


# ── Mutation helper ──────────────────────────────────────────────────────────

def _rebind_view(ctx: _MsgOverlayCtx) -> None:
    """Rebuild all view state from the current ctx.messages list.

    Call after anything that can wholesale-replace messages (history jump,
    transform, undo/redo). Resets everything keyed against the previous
    message identities: marks, scroll, fold state. Applies the current
    filter so the view reflects the new contents, clamps the cursor, and
    invalidates the prev_lines cache so the next draw repaints every row.
    """
    ctx.marks.clear()
    ctx.visual_mode = False
    ctx.collapsed = {id(m) for m in ctx.messages}

    ctx.view = list(range(len(ctx.messages)))
    if ctx.filter_pattern:
        _overlay_apply_filter(ctx, ctx.filter_pattern)
    ctx.selected_idx = max(0, min(ctx.selected_idx, max(0, len(ctx.view) - 1)))
    ctx.scroll = 0
    ctx.forced_scroll = None

    ctx.search_matches = _overlay_find_matches(ctx)
    if ctx.search_matches:
        _overlay_sync_search_cursor(ctx)
    else:
        ctx.search_match_idx = -1

    # Rebase the token estimate's character baseline to the new content
    # so the percentage stays meaningful after a wholesale replacement.
    ctx.base_chars = sum(len(_overlay_msg_text(m)) for m in ctx.messages) or 1
    _overlay_recompute_tokens(ctx)

    ctx.prev_lines.clear()
    ctx.first_draw = True


def _record_mutation(ctx: _MsgOverlayCtx, label: str, meta: dict | None = None) -> None:
    """Fire messages_mutated so the tracker records, then refresh the view."""
    from cai.llm import fire_event
    fire_event('messages_mutated', {
        'messages': ctx.messages, 'label': label, 'meta': meta or {},
    })
    _rebind_view(ctx)


def _clear_selection(ctx: _MsgOverlayCtx) -> None:
    ctx.marks.clear()
    ctx.visual_mode = False


def _splice_selection(ctx: _MsgOverlayCtx, selected_indices: list[int],
                      replacement: list) -> None:
    """Remove selected messages from ctx.messages and insert replacement at the
    position of the lowest selected index. Non-contiguous selection collapses
    to a single splice point — see plan section 7.
    """
    if not selected_indices:
        return
    insert_at = selected_indices[0]
    # Remove from highest to lowest so indices stay valid.
    for i in sorted(selected_indices, reverse=True):
        if 0 <= i < len(ctx.messages):
            del ctx.messages[i]
    for j, m in enumerate(replacement):
        ctx.messages.insert(insert_at + j, m)


def _apply_transform(ctx: _MsgOverlayCtx, name: str, kwargs: dict) -> None:
    """Run transform on current selection and splice the result back in."""
    from cai.transforms import get_transform
    try:
        spec = get_transform(name)
    except KeyError:
        ctx.status_flash = f'unknown transform: {name}'
        return

    sel_indices = _msg_effective_selection(ctx)
    selected = [copy.deepcopy(ctx.messages[i]) for i in sel_indices]
    try:
        replacement = spec.fn(copy.deepcopy(selected), **kwargs)
    except Exception as e:
        ctx.status_flash = f'transform error: {e}'
        return

    if not isinstance(replacement, list):
        ctx.status_flash = 'transform error: return value is not a list'
        return

    _splice_selection(ctx, sel_indices, replacement)
    _record_mutation(ctx, f'transform:{name}', {
        'n_in': len(selected), 'n_out': len(replacement),
        'args': {k: str(v)[:80] for k, v in kwargs.items()},
    })
    _clear_selection(ctx)
    ctx.status_flash = f'applied {name}: {len(selected)} → {len(replacement)}'


# ── Filter / search / instruction key handlers ───────────────────────────────

def overlay_filter_key(ctx: _MsgOverlayCtx, key: str) -> None:
    if key in KEY_ENTER:
        pat = ''.join(ctx.filter_buf)
        if pat and (not ctx.filter_history or ctx.filter_history[0] != pat):
            ctx.filter_history.insert(0, pat)
        _overlay_apply_filter(ctx, pat)
        ctx.filter_mode = False
        ctx.filter_history_pos = -1
        ctx.first_draw = True
        return
    if key == KEY_ESC:
        ctx.filter_mode = False
        ctx.filter_history_pos = -1
        ctx.filter_buf[:] = list(ctx.filter_pattern)
        return
    if key == KEY_BACKSPACE:
        if ctx.filter_buf:
            ctx.filter_buf.pop()
            ctx.filter_history_pos = -1
        else:
            ctx.filter_mode = False
            ctx.filter_buf[:] = list(ctx.filter_pattern)
        return
    if key == KEY_UP and ctx.filter_history:
        ctx.filter_history_pos = min(ctx.filter_history_pos + 1,
                                     len(ctx.filter_history) - 1)
        ctx.filter_buf[:] = list(ctx.filter_history[ctx.filter_history_pos])
        return
    if key == KEY_DOWN:
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


def overlay_search_key(ctx: _MsgOverlayCtx, key: str) -> None:
    if key in KEY_ENTER:
        ctx.search_mode = False
        return
    if key == KEY_ESC:
        ctx.search_mode = False
        ctx.selected_idx = ctx.pre_search_idx
        ctx.search_pattern = ''
        ctx.search_buf = []
        ctx.search_matches = []
        ctx.search_match_idx = -1
        return
    if key == KEY_BACKSPACE:
        if ctx.search_buf:
            ctx.search_buf.pop()
            ctx.search_pattern = ''.join(ctx.search_buf)
            ctx.search_matches = _overlay_find_matches(ctx)
            _overlay_sync_search_cursor(ctx)
        else:
            ctx.search_mode = False
            ctx.selected_idx = ctx.pre_search_idx
            ctx.search_pattern = ''
            ctx.search_matches = []
            ctx.search_match_idx = -1
        return
    if len(key) == 1 and ord(key) >= 32:
        ctx.search_buf.append(key)
        ctx.search_pattern = ''.join(ctx.search_buf)
        ctx.search_matches = _overlay_find_matches(ctx)
        _overlay_sync_search_cursor(ctx)


def overlay_instruction_key(ctx: _MsgOverlayCtx, key: str) -> 'str | None':
    """Return the submitted instruction, empty string on cancel, or None if more keys pending."""
    if key in KEY_ENTER:
        text = ''.join(ctx.instruction_buf).strip()
        ctx.instruction_mode = False
        ctx.instruction_buf = []
        return text if text else ''
    if key == KEY_ESC:
        ctx.instruction_mode = False
        ctx.instruction_buf = []
        return ''
    if key == KEY_BACKSPACE:
        if ctx.instruction_buf:
            ctx.instruction_buf.pop()
        return None
    if len(key) == 1 and ord(key) >= 32:
        ctx.instruction_buf.append(key)
    return None


# ── Transform arg prompt ─────────────────────────────────────────────────────

def _prompt_transform_args(ctx: _MsgOverlayCtx, screen, spec) -> 'dict | None':
    """Prompt the user for transform parameters on the status line.

    Pre-fills ``model`` from ctx.model so the user usually just hits Enter.
    Returns kwargs dict, or None if cancelled.
    """
    # Auto-supply model if expected; only prompt for the remaining params.
    auto: dict = {}
    manual_params = []
    for name, typ, default in spec.params:
        if name == 'model' and ctx.model:
            auto[name] = ctx.model
        else:
            manual_params.append((name, typ, default))

    if not manual_params:
        return auto

    prompt_prefix = f' {spec.name}('
    prompt_suffix = ')'
    buf: list[str] = []
    # Show the form so the user can see what they need to fill in.
    hint = ', '.join(f'{n}=' for n, _t, _d in manual_params)

    # Reuse the existing redraw — we render a custom status line while
    # collecting input. Implemented inline as a small secondary loop.
    first = True
    while True:
        ctx.status_flash = prompt_prefix + hint + '  ' + ''.join(buf) + prompt_suffix
        overlay_redraw(ctx, screen._rows, screen._cols)
        ctx.status_flash = ''  # clear for next frame computation
        first = False
        rlist, _, _ = select.select([screen._tty_fd], [], [], 0.05)
        if not rlist:
            continue
        key = read_key(screen._tty_fd)
        if key == KEY_ESC or key == KEY_CTRL_C:
            return None
        if key in KEY_ENTER:
            break
        if key == KEY_BACKSPACE:
            if buf:
                buf.pop()
            continue
        if len(key) == 1 and ord(key) >= 32:
            buf.append(key)

    raw = ''.join(buf).strip()
    if not raw:
        # All manual params must have defaults to succeed without input.
        out = dict(auto)
        for name, _typ, default in manual_params:
            out[name] = default
        return out

    # Parse comma-separated literals in order.
    try:
        # Wrap in a tuple so ast.literal_eval handles single values too.
        parsed = ast.literal_eval('(' + raw + ',)')
    except (ValueError, SyntaxError) as e:
        ctx.status_flash = f'arg parse error: {e}'
        return None

    values = list(parsed)
    out = dict(auto)
    for i, (name, _typ, default) in enumerate(manual_params):
        if i < len(values):
            out[name] = values[i]
        else:
            out[name] = default
    return out


# ── Normal-mode nav / actions ────────────────────────────────────────────────

def _half_visible(rows: int) -> int:
    overhead = 4
    max_box_h = max(overhead + 1, int(rows * 0.95))
    vis = max_box_h - overhead
    return max(1, vis // 2)


def overlay_nav_key(ctx: _MsgOverlayCtx, key: str, rows: int, screen) -> 'str | None':
    """Handle one keypress in normal mode.

    Returns one of:
      None     — stay open
      'close'  — close the overlay
      'history' — open the history overlay nested inside this one
    """
    nv = len(ctx.view)
    if nv > 0:
        ctx.selected_idx = max(0, min(nv - 1, ctx.selected_idx))

    # Clear any transient flash on the next keypress so errors don't linger.
    if ctx.status_flash and key not in KEY_ENTER:
        ctx.status_flash = ''

    if nv == 0:
        if key in (KEY_ESC, 'q', KEY_CTRL_C):
            return 'close'
        return None

    pk = ctx.prev_key
    sel = ctx.selected_idx
    half = _half_visible(rows)

    if key in (KEY_ESC, 'q', KEY_CTRL_C):
        if ctx.visual_mode:
            ctx.visual_mode = False
            return None
        return 'close'

    # Nav
    if key in (KEY_UP, 'k'):
        ctx.selected_idx = max(0, sel - 1)
    elif key in (KEY_DOWN, 'j'):
        ctx.selected_idx = min(nv - 1, sel + 1)
    elif key == 'G':
        ctx.selected_idx = nv - 1
    elif key == 'g' and pk == 'g':
        ctx.selected_idx = 0
    elif key == KEY_CTRL_U:
        ctx.selected_idx = max(0, sel - half)
    elif key == KEY_CTRL_D:
        ctx.selected_idx = min(nv - 1, sel + half)
    # scroll align
    elif key == 'z' and pk == 'z':
        ctx.scroll = max(0, sel - half)
        ctx.forced_scroll = None
    elif key == 't' and pk == 'z':
        ctx.scroll = sel
        ctx.forced_scroll = None
    elif key == 'b' and pk == 'z':
        ctx.scroll = max(0, sel - half * 2 + 1)
        ctx.forced_scroll = None

    # Folding
    elif key == 'a' and pk == 'z':
        _toggle_fold(ctx, ctx.view[sel])
    elif key == '\t':
        # Tab is a quicker alias for za, no prefix required.
        _toggle_fold(ctx, ctx.view[sel])
    elif key == 'c' and pk == 'z':
        _set_fold(ctx, ctx.view[sel], True)
    elif key == 'o' and pk == 'z':
        _set_fold(ctx, ctx.view[sel], False)
    elif key == 'M' and pk == 'z':
        _fold_all(ctx, True)
    elif key == 'R' and pk == 'z':
        _fold_all(ctx, False)

    # Filter / search
    elif key == 'f':
        ctx.filter_mode = True
        ctx.filter_history_pos = -1
        ctx.filter_buf[:] = list(ctx.filter_pattern)
    elif key in ('/', '?'):
        ctx.search_mode = True
        ctx.search_direction = 1 if key == '/' else -1
        ctx.search_buf = []
        ctx.search_pattern = ''
        ctx.search_matches = []
        ctx.search_match_idx = -1
        ctx.pre_search_idx = sel
    elif key == 'n' and ctx.search_matches:
        step = 1 if ctx.search_direction == 1 else -1
        ctx.search_match_idx = (ctx.search_match_idx + step) % len(ctx.search_matches)
        ctx.selected_idx = ctx.search_matches[ctx.search_match_idx]
    elif key == 'N' and ctx.search_matches:
        step = -1 if ctx.search_direction == 1 else 1
        ctx.search_match_idx = (ctx.search_match_idx + step) % len(ctx.search_matches)
        ctx.selected_idx = ctx.search_matches[ctx.search_match_idx]

    # Selection
    elif key == 'm':
        ai = ctx.view[sel]
        if ai in ctx.marks:
            ctx.marks.discard(ai)
        else:
            ctx.marks.add(ai)
    elif key == 'M':
        ctx.marks.clear()
    elif key == 'V':
        if ctx.visual_mode:
            ctx.visual_mode = False
        else:
            ctx.visual_mode = True
            ctx.visual_anchor = sel

    # Actions
    elif key == 'y':
        sel_indices = _msg_effective_selection(ctx)
        ctx.yank_register = [copy.deepcopy(ctx.messages[i]) for i in sel_indices]
        ctx.status_flash = f'yanked {len(ctx.yank_register)} message(s)'
        _clear_selection(ctx)
    elif key == 'd':
        sel_indices = _msg_effective_selection(ctx)
        if sel_indices:
            ctx.yank_register = [copy.deepcopy(ctx.messages[i]) for i in sel_indices]
            _splice_selection(ctx, sel_indices, [])
            _record_mutation(ctx, 'overlay:delete', {'n': len(sel_indices)})
            _clear_selection(ctx)
            ctx.status_flash = f'deleted {len(sel_indices)} message(s)'
    elif key == 'p':
        if ctx.yank_register:
            paste = [copy.deepcopy(m) for m in ctx.yank_register]
            insert_at = ctx.view[sel] + 1
            for j, m in enumerate(paste):
                ctx.messages.insert(insert_at + j, m)
            _record_mutation(ctx, 'overlay:paste', {'n': len(paste)})
            ctx.status_flash = f'pasted {len(paste)} message(s)'
    elif key == '>':
        return 'transform'
    elif key == '!':
        ctx.instruction_mode = True
        ctx.instruction_buf = []
    elif key == 'h' and pk == 'g':
        return 'history'
    elif key in KEY_ENTER:
        _edit_in_nvim(ctx, screen)

    return None


def _toggle_fold(ctx: _MsgOverlayCtx, msg_idx: int) -> None:
    """Toggle the body-fold state for the message under the cursor."""
    key = id(ctx.messages[msg_idx])
    if key in ctx.collapsed:
        ctx.collapsed.discard(key)
    else:
        ctx.collapsed.add(key)


def _set_fold(ctx: _MsgOverlayCtx, msg_idx: int, folded: bool) -> None:
    key = id(ctx.messages[msg_idx])
    if folded:
        ctx.collapsed.add(key)
    else:
        ctx.collapsed.discard(key)


def _fold_all(ctx: _MsgOverlayCtx, folded: bool) -> None:
    if folded:
        ctx.collapsed = {id(m) for m in ctx.messages}
    else:
        ctx.collapsed.clear()


# ── Nvim edit (reuses context.py's logic verbatim, parameterised) ────────────

def _edit_in_nvim(ctx: _MsgOverlayCtx, screen) -> None:
    ai = ctx.view[ctx.selected_idx]
    msg = ctx.messages[ai]
    content = msg.get('content', '')
    tool_calls = msg.get('tool_calls')
    reasoning = msg.get('_reasoning', '')
    is_json = not isinstance(content, str)
    read_only = bool(tool_calls)

    sections = []
    if reasoning:
        sections.append(f"--- Reasoning ---\n{reasoning}")
    if tool_calls:
        parts = []
        for tc in tool_calls:
            func = tc.get('function', {})
            name = func.get('name', '?')
            raw_args = func.get('arguments', '')
            try:
                parsed = _json.loads(raw_args) if raw_args else {}
                args_str = _json.dumps(parsed, indent=2)
            except Exception:
                args_str = raw_args
            parts.append(f"Tool: {name}\nArguments:\n{args_str}")
        sections.append("--- Tool Calls ---\n" + '\n\n'.join(parts))
    if content:
        text = _json.dumps(content, indent=2) if is_json else content
        sections.append(f"--- Content ---\n{text}" if sections else text)
    text = '\n\n'.join(sections) if sections else (content or '')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        sys.stdout.write(f'{ALT_EXIT}{CUR_SHOW}')
        sys.stdout.flush()
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, screen._cooked_attrs)
        if read_only:
            subprocess.run(['nvim', '-R', tmp])
        else:
            subprocess.run(['nvim', tmp])
            with open(tmp, 'r') as f:
                new_text = f.read()
            if reasoning and new_text.startswith('--- Reasoning ---\n'):
                marker = '\n\n--- Content ---\n'
                p = new_text.find(marker)
                if p >= 0:
                    new_text = new_text[p + len(marker):]
                elif not content:
                    new_text = ''
            if is_json:
                try:
                    ctx.messages[ai]['content'] = _json.loads(new_text)
                except Exception:
                    ctx.messages[ai]['content'] = new_text
            else:
                ctx.messages[ai]['content'] = new_text
            _record_mutation(ctx, 'overlay:edit', {'idx': ai})
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
        sys.stdout.flush()
        tty.setraw(screen._tty_fd)
        ctx.first_draw = True


# ── Event loop ────────────────────────────────────────────────────────────────

def prompt_messages_overlay(
    screen,
    messages: list,
    tracker,
    *,
    model: str = '',
    context_size: int = 0,
    prompt_tokens: int = 0,
) -> tuple:
    """Interactive messages overlay. Returns (messages, new_tokens_estimate)."""
    if not messages:
        return messages, 0

    ctx = _MsgOverlayCtx(messages, tracker, model,
                        context_size=context_size,
                        prompt_tokens=prompt_tokens)

    old_attrs = termios.tcgetattr(screen._tty_fd)
    orig_handler = signal.getsignal(signal.SIGWINCH)

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        ctx.resize_pending = True
        ctx.first_draw = True

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

            rlist, _, _ = select.select([screen._tty_fd], [], [], 0.05)
            if not rlist:
                continue
            key = read_key(screen._tty_fd)

            if ctx.filter_mode:
                overlay_filter_key(ctx, key)
                overlay_redraw(ctx, screen._rows, screen._cols)
                continue
            if ctx.search_mode:
                overlay_search_key(ctx, key)
                overlay_redraw(ctx, screen._rows, screen._cols)
                continue
            if ctx.instruction_mode:
                result = overlay_instruction_key(ctx, key)
                if result is None:
                    overlay_redraw(ctx, screen._rows, screen._cols)
                    continue
                if not result:
                    # cancelled / empty
                    overlay_redraw(ctx, screen._rows, screen._cols)
                    continue

                # Hand off to the streaming popup so the user sees response
                # chunks land as the model produces them instead of staring
                # at a frozen screen.
                from .llm_stream import run_llm_transform
                sel_indices = _msg_effective_selection(ctx)
                selected = [copy.deepcopy(ctx.messages[i]) for i in sel_indices]
                replacement = run_llm_transform(screen, result, ctx.model, selected)

                # Return to the messages overlay's alt-screen + raw mode.
                sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
                sys.stdout.flush()
                tty.setraw(screen._tty_fd)
                ctx.first_draw = True

                if replacement is not None:
                    _splice_selection(ctx, sel_indices, replacement)
                    _record_mutation(ctx, 'llm_transform', {
                        'n_in': len(selected), 'n_out': len(replacement),
                        'instruction': result[:80],
                    })
                    _clear_selection(ctx)
                    ctx.status_flash = (
                        f'applied llm: {len(selected)} → {len(replacement)}'
                    )
                else:
                    ctx.status_flash = 'llm transform failed (see popup)'
                overlay_redraw(ctx, screen._rows, screen._cols)
                continue

            action = overlay_nav_key(ctx, key, screen._rows, screen)
            ctx.prev_key = key

            if action == 'close':
                break
            if action == 'transform':
                from .transform_picker import prompt_transform_picker
                from cai.transforms import list_transforms, get_transform
                picked = prompt_transform_picker(screen, list_transforms())
                # After picker returns, we're back in the alt screen via
                # ALT_ENTER done by the event-loop setup; redraw.
                sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
                sys.stdout.flush()
                tty.setraw(screen._tty_fd)
                ctx.first_draw = True
                if picked:
                    spec = get_transform(picked)
                    kwargs = _prompt_transform_args(ctx, screen, spec)
                    if kwargs is not None:
                        _apply_transform(ctx, picked, kwargs)
                overlay_redraw(ctx, screen._rows, screen._cols)
                continue
            if action == 'history':
                from .history import prompt_history_overlay
                pre_head = tracker.head() if tracker else None
                prompt_history_overlay(screen, tracker)
                sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
                sys.stdout.flush()
                tty.setraw(screen._tty_fd)
                # The tracker may have jumped; rebuild the whole view.
                _rebind_view(ctx)
                if tracker and tracker.head() != pre_head:
                    ctx.status_flash = f'jumped to #{tracker.head()}'
                overlay_redraw(ctx, screen._rows, screen._cols)
                continue

            overlay_redraw(ctx, screen._rows, screen._cols)

    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()

    return messages, ctx.tokens_est
