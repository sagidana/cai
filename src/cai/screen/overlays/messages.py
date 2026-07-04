"""Interactive messages overlay — vim-style viewer/editor for messages[].

Adds folding (per-message + assistant/tool pair), multi-message selection
(marks + visual-line), ad-hoc LLM rewrite (``!``), and edit/delete. edits are
written back to the host conversation on close.
"""

import ast
import copy
import json
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
    MsgOverlayCtx,
    _msg_header_preview, _msg_body_lines,
    _msg_is_folded, _msg_effective_selection,
    _overlay_find_matches, _overlay_sync_search_cursor,
    _overlay_apply_filter, _overlay_recompute_tokens,
)
from ..input import read_key, parse_mouse


_ROLE_COLOR = {
    'system':    SGR_MAGENTA,
    'user':      SGR_GREEN,
    'assistant': SGR_CYAN,
    'tool':      SGR_YELLOW,
}


def _build_rows(ctx, inner_w, prefix_w):
    """flatten ctx.view into display rows with folding applied.
    one message = one header row. expanded messages additionally emit one
    body row per wrapped content line."""
    rows = []
    content_w = max(4, inner_w - prefix_w)
    for pos, ai in enumerate(ctx.view):
        msg = ctx.messages[ai]
        folded = _msg_is_folded(ctx, ai)
        glyph = '▼'
        if folded:
            glyph = '▶'
        preview = _msg_header_preview(msg, content_w - 2)
        rows.append((pos, ai, 'header', f'{glyph} {preview}'))
        if not folded:
            for line in _msg_body_lines(msg, content_w - 2):  # 2 for indent
                rows.append((pos, ai, 'body', line))
    return rows


def _build_status(ctx, nv, nm, inner_w, sel_count):
    """return (raw_status, use_reverse)."""
    if ctx.filter_mode:
        return f' filter: {"".join(ctx.filter_buf)}', True
    if ctx.search_mode:
        dir_char = '?'
        if ctx.search_direction == 1:
            dir_char = '/'
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

    pos_str = ' 0/0'
    if nv > 0:
        pos_str = f' {ctx.selected_idx + 1}/{nv}'
    if sel_count:
        vis_tag = ''
        if ctx.visual_mode:
            vis_tag = ' V'
        pos_str += f'  {SGR_YELLOW}[sel:{sel_count}{vis_tag}]{SGR_RESET}'
    if ctx.filter_pattern:
        pos_str += f'  {SGR_YELLOW}[filter: {ctx.filter_pattern}]{SGR_RESET}  ({nv}/{nm})'
    if ctx.search_pattern:
        dir_char = '?'
        if ctx.search_direction == 1:
            dir_char = '/'
        m = ''
        if ctx.search_matches:
            m = f' [{ctx.search_match_idx + 1}/{len(ctx.search_matches)}]'
        pos_str += f'   {dir_char}{ctx.search_pattern}{m}'
    hints = '  Tab:fold  m/V:sel  !:rewrite  ESC:close'
    if len(ansi_strip(pos_str)) + len(hints) <= inner_w:
        pos_str += hints
    return pos_str, False


def _style_cell(text, role, is_sel, is_match, in_sel):
    if is_sel and is_match:
        return f'{SGR_REVERSE_YELLOW}{text}{SGR_RESET}'
    if is_sel:
        return f'{SGR_REVERSE}{text}{SGR_RESET}'
    if is_match:
        return f'{SGR_YELLOW}{text}{SGR_RESET}'
    if in_sel:
        # subtle indicator for non-cursor selected rows: dim reverse
        return f'{SGR_REVERSE}{SGR_DIM_GRAY}{text}{SGR_RESET}'
    color = _ROLE_COLOR.get(role, '')
    if color:
        return f'{color}{text}{SGR_RESET}'
    return text


def draw_messages_overlay(ctx, rows, cols):
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
    # find the row index of the selected message's header so scroll keeps
    # the cursor visible. falls back to 0 if no rows.
    cursor_row_i = 0
    for i, (pos, ai, kind, _t) in enumerate(all_rows):
        if pos == ctx.selected_idx and kind == 'header':
            cursor_row_i = i
            break

    total_rows = len(all_rows)
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

    # pre-compute search-match positions for highlight lookup
    match_positions = set(ctx.search_matches)

    H = '─'
    TL = '┌'
    TR = '┐'
    BL = '└'
    BR = '┘'
    VL = '│'
    ML = '├'
    MR = '┤'
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

    new_lines = {}

    def put(row_off, text):
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
            mark_ch = ' '
            if ai in ctx.marks:
                mark_ch = '*'
            prefix = f' {idx_str} {role_str} {mark_ch} '
            cell = (prefix + text)[:inner_w].ljust(inner_w)
            styled = _style_cell(cell, role, is_cursor, is_match, in_sel)
            put(1 + i, f'{VL}{styled}{VL}')
        else:
            indent = '   ' + ' ' * (idx_w + role_w + mark_w)
            body = (indent + text)[:inner_w].ljust(inner_w)
            # body rows never render with cursor highlight; selection dim only.
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

    out = []
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


def overlay_redraw(ctx, rows, cols):
    draw_messages_overlay(ctx, rows, cols)
    ctx.first_draw = False


def _rebind_view(ctx):
    """rebuild all view state from the current ctx.messages list.
    call after anything that can wholesale-replace messages (history jump,
    transform, undo/redo). resets everything keyed against the previous
    message identities: marks, scroll, fold state. applies the current
    filter so the view reflects the new contents, clamps the cursor, and
    invalidates the prev_lines cache so the next draw repaints every row."""
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

    _overlay_recompute_tokens(ctx)

    ctx.prev_lines.clear()
    ctx.first_draw = True


def _record_mutation(ctx, label, meta=None):
    """mark the conversation edited (so it's written back on close) and refresh
    the view. label/meta are unused now - kept for call-site readability."""
    ctx.modified = True
    _rebind_view(ctx)
    ctx.dirty = False


def _live_sync(ctx, refetch):
    """pull messages the turn appended on the host into ctx.messages so the
    overlay keeps up with a turn running underneath it. append-only: mid-turn
    the host only grows the conversation, so we add the new tail and keep the
    existing dict identities (fold/seen state stays put). frozen once the user
    makes a real edit (ctx.modified) so a refetch can never clobber their
    in-overlay work."""
    if refetch is None: return
    if ctx.modified: return
    fresh = refetch()
    if not fresh: return
    if len(fresh) <= len(ctx.messages): return
    ctx.messages.extend(fresh[len(ctx.messages):])


def _refresh_view_after_external_mutation(ctx):
    """rebuild ctx.view after the worker thread mutated ctx.messages.
    unlike _rebind_view (used for transforms/jumps that wholesale-replace
    messages), this preserves marks, fold state, and the cursor position
    where possible. the common case is a fresh assistant message arriving
    at the tail while the user is browsing - keep their selection pinned
    unless they were already at the bottom, in which case follow."""
    nv_before = len(ctx.view)
    cursor_was_at_tail = (nv_before == 0) or (ctx.selected_idx >= nv_before - 1)

    ctx.view = list(range(len(ctx.messages)))
    if ctx.filter_pattern:
        _overlay_apply_filter(ctx, ctx.filter_pattern)

    nv = len(ctx.view)
    if nv == 0:
        ctx.selected_idx = 0
    elif cursor_was_at_tail:
        ctx.selected_idx = nv - 1
    else:
        ctx.selected_idx = max(0, min(ctx.selected_idx, nv - 1))

    if ctx.search_pattern:
        ctx.search_matches = _overlay_find_matches(ctx)
        if ctx.search_matches:
            _overlay_sync_search_cursor(ctx)
        else:
            ctx.search_match_idx = -1

    # newly-arrived messages default to collapsed so the view stays compact
    # as content lands - the user expands with Tab to read. messages the
    # user already expanded keep their state: only unseen ids are folded.
    for m in ctx.messages:
        mid = id(m)
        if mid in ctx.seen_ids: continue
        ctx.collapsed.add(mid)
        ctx.seen_ids.add(mid)

    _overlay_recompute_tokens(ctx)

    ctx.prev_lines.clear()
    ctx.first_draw = True


def _clear_selection(ctx):
    ctx.marks.clear()
    ctx.visual_mode = False


def _splice_selection(ctx, selected_indices, replacement):
    """remove selected messages from ctx.messages and insert replacement at
    the position of the lowest selected index. non-contiguous selection
    collapses to a single splice point."""
    if not selected_indices: return

    insert_at = selected_indices[0]
    # remove from highest to lowest so indices stay valid.
    for i in sorted(selected_indices, reverse=True):
        if 0 <= i < len(ctx.messages):
            del ctx.messages[i]
    for j, m in enumerate(replacement):
        ctx.messages.insert(insert_at + j, m)


def _reenter_overlay(ctx, screen):
    """re-enter the overlay's alt-screen + raw mode after a sub-popup."""
    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()
    tty.setraw(screen._tty_fd)
    ctx.first_draw = True


def _run_rewrite(ctx, screen, instruction, *, header):
    """run the ad-hoc LLM rewrite (the ! action) via the live popup, then
    splice the result. the rewrite lives in this overlay (.rewrite) and runs a
    one-shot cai.Run, so the user sees the model's progress (tokens, reasoning)
    instead of a frozen screen."""
    from .llm_stream import run_streaming_transform
    from .rewrite import rewrite

    sel_indices = _msg_effective_selection(ctx)
    selected = [copy.deepcopy(ctx.messages[i]) for i in sel_indices]
    replacement = run_streaming_transform(screen,
                                          rewrite,
                                          'rewrite',
                                          selected,
                                          {'instruction': instruction},
                                          header=header)
    _reenter_overlay(ctx, screen)
    if replacement is None:
        ctx.status_flash = 'rewrite failed (see popup)'
        return

    _splice_selection(ctx, sel_indices, replacement)

    meta = {}
    meta['n_in'] = len(selected)
    meta['n_out'] = len(replacement)
    meta['instruction'] = str(instruction)[:80]
    _record_mutation(ctx, 'llm_rewrite', meta)

    _clear_selection(ctx)
    ctx.status_flash = f'rewrote {len(selected)} → {len(replacement)}'


def overlay_filter_key(ctx, key):
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


def overlay_search_key(ctx, key):
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


def overlay_instruction_key(ctx, key):
    """return the submitted instruction, empty string on cancel, or None if
    more keys are pending."""
    if key in KEY_ENTER:
        text = ''.join(ctx.instruction_buf).strip()
        ctx.instruction_mode = False
        ctx.instruction_buf = []
        return text
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


def _half_visible(rows):
    overhead = 4
    max_box_h = max(overhead + 1, int(rows * 0.95))
    vis = max_box_h - overhead
    return max(1, vis // 2)


def overlay_nav_key(ctx, key, rows, screen):
    """handle one keypress in normal mode. returns one of:
      None      - stay open
      'close'   - close the overlay"""
    nv = len(ctx.view)
    if nv > 0:
        ctx.selected_idx = max(0, min(nv - 1, ctx.selected_idx))

    # clear any transient flash on the next keypress so errors don't linger.
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

    # folding
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

    # filter / search
    elif key == 'f':
        ctx.filter_mode = True
        ctx.filter_history_pos = -1
        ctx.filter_buf[:] = list(ctx.filter_pattern)
    elif key in ('/', '?'):
        ctx.search_mode = True
        ctx.search_direction = -1
        if key == '/':
            ctx.search_direction = 1
        ctx.search_buf = []
        ctx.search_pattern = ''
        ctx.search_matches = []
        ctx.search_match_idx = -1
        ctx.pre_search_idx = sel
    elif key == 'n' and ctx.search_matches:
        step = ctx.search_direction
        ctx.search_match_idx = (ctx.search_match_idx + step) % len(ctx.search_matches)
        ctx.selected_idx = ctx.search_matches[ctx.search_match_idx]
    elif key == 'N' and ctx.search_matches:
        step = -ctx.search_direction
        ctx.search_match_idx = (ctx.search_match_idx + step) % len(ctx.search_matches)
        ctx.selected_idx = ctx.search_matches[ctx.search_match_idx]

    # selection
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

    # actions
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
    elif key == '!':
        ctx.instruction_mode = True
        ctx.instruction_buf = []
    elif key in KEY_ENTER:
        _edit_in_nvim(ctx, screen)

    return None


def _toggle_fold(ctx, msg_idx):
    """toggle the body-fold state for the message under the cursor."""
    key = id(ctx.messages[msg_idx])
    if key in ctx.collapsed:
        ctx.collapsed.discard(key)
    else:
        ctx.collapsed.add(key)


def _set_fold(ctx, msg_idx, folded):
    key = id(ctx.messages[msg_idx])
    if folded:
        ctx.collapsed.add(key)
    else:
        ctx.collapsed.discard(key)


def _fold_all(ctx, folded):
    if folded:
        ctx.collapsed = {id(m) for m in ctx.messages}
    else:
        ctx.collapsed.clear()


def _unpack_tool_args(msg):
    """return a copy of msg with each tool_call's stringified arguments
    parsed into a nested JSON value. arguments that don't parse (malformed
    or already non-string) are left untouched."""
    out = copy.deepcopy(msg)
    for tc in out.get('tool_calls') or []:
        fn = tc.get('function')
        if not isinstance(fn, dict): continue
        raw = fn.get('arguments')
        if isinstance(raw, str) and raw.strip():
            try:
                fn['arguments'] = json.loads(raw)
            except json.JSONDecodeError:
                pass
    return out


def _repack_tool_args(msg):
    """inverse of _unpack_tool_args: re-serialize any nested tool-call
    arguments back to a string, which is the wire format the LLM and tool
    dispatcher expect."""
    for tc in msg.get('tool_calls') or []:
        fn = tc.get('function')
        if not isinstance(fn, dict): continue
        args = fn.get('arguments')
        if isinstance(args, (dict, list)):
            fn['arguments'] = json.dumps(args)
    return msg


def _as_json_text(text):
    """return pretty JSON if text is a JSON or Python-literal dict/list.
    tool results are stored as str(result) so dict/list payloads arrive as
    a Python repr (single quotes, True/False/None) that jq can't parse.
    returns the reformatted JSON string, or None if text isn't structured."""
    s = text.strip()
    if not s or s[0] not in '{[':
        return None
    obj = None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(obj, (dict, list)):
        return None
    return json.dumps(obj, indent=2, ensure_ascii=False)


_REASONING_MARK = '═══ reasoning ═══'
_RESPONSE_MARK = '═══ response ═══'


def _sectioned_text(reasoning, response):
    """two-section editor layout for assistant messages with reasoning."""
    return (f'{_REASONING_MARK}\n{reasoning}\n\n'
            f'{_RESPONSE_MARK}\n{response}')


def _parse_sections(text):
    """split sectioned editor text back into (reasoning, response).
    returns None when a marker was removed or the two were reordered."""
    lines = text.split('\n')
    reasoning_at = None
    response_at = None
    for i, line in enumerate(lines):
        if line.strip() == _REASONING_MARK and reasoning_at is None:
            reasoning_at = i
        elif line.strip() == _RESPONSE_MARK and response_at is None:
            response_at = i
    if reasoning_at is None: return None
    if response_at is None: return None
    if response_at < reasoning_at: return None
    reasoning = '\n'.join(lines[reasoning_at + 1:response_at]).strip('\n')
    response = '\n'.join(lines[response_at + 1:]).strip('\n')
    return (reasoning, response)


def _edit_in_nvim(ctx, screen):
    """open the selected message in nvim. messages with structure
    (tool_calls or list content) are edited as the raw JSON object so the
    user can rewrite tool names, arguments, call ids, etc. assistant
    messages with stored reasoning edit as two plain-text sections
    (reasoning / response); each section writes back only if it changed.
    plain text messages edit as raw content - no JSON escaping needed
    for the common case."""
    ai = ctx.view[ctx.selected_idx]
    msg = ctx.messages[ai]

    needs_json = (bool(msg.get('tool_calls'))
                  or isinstance(msg.get('content'), list))
    sectioned = bool(msg.get('_reasoning')) and not needs_json

    if needs_json:
        editable = _unpack_tool_args(msg)
        # ensure_ascii=False so non-ASCII content stays readable in the editor.
        text = json.dumps(editable, indent=2, ensure_ascii=False)
        suffix = '.json'
    elif sectioned:
        response = msg.get('content', '') or ''
        json_text = _as_json_text(response)
        if json_text is not None:
            response = json_text
        text = _sectioned_text(msg['_reasoning'], response)
        suffix = '.md'
    else:
        text = msg.get('content', '') or ''
        suffix = '.txt'
        # tool results are stringified with str() (tools.py), so dict/list
        # payloads land here as a Python repr ('-quotes, True/False/None).
        # reformat those as real JSON so the user can pipe through !jq.
        json_text = _as_json_text(text)
        if json_text is not None:
            text = json_text
            suffix = '.json'

    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(text)
        tmp = f.name

    try:
        sys.stdout.write(f'{ALT_EXIT}{CUR_SHOW}')
        sys.stdout.flush()
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, screen._cooked_attrs)
        subprocess.run(['nvim', tmp])
        with open(tmp, 'r') as f:
            new_text = f.read()

        # pure view: nothing changed in the editor. must not mark the
        # conversation modified - that freezes live-sync and pushes a stale
        # snapshot back on close, rolling back a turn streaming underneath.
        if new_text == text:
            return

        if needs_json:
            try:
                new_msg = json.loads(new_text)
            except json.JSONDecodeError as e:
                ctx.status_flash = f'edit: invalid JSON — {e}'
                return
            if not isinstance(new_msg, dict) or 'role' not in new_msg:
                ctx.status_flash = 'edit: expected a JSON object with "role"'
                return
            _repack_tool_args(new_msg)
            ctx.messages[ai] = new_msg
        elif sectioned:
            old_sections = _parse_sections(text)
            new_sections = _parse_sections(new_text)
            if new_sections is None:
                ctx.status_flash = 'edit: section markers missing or reordered'
                return
            # whitespace-only difference around the markers parses back to
            # identical sections - treat as a pure view, same as new_text == text.
            if new_sections == old_sections:
                return
            edited = dict(msg)
            if new_sections[0] != old_sections[0]:
                if new_sections[0]:
                    edited['_reasoning'] = new_sections[0]
                else:
                    edited.pop('_reasoning', None)
            if new_sections[1] != old_sections[1]:
                edited['content'] = new_sections[1]
            ctx.messages[ai] = edited
        else:
            # replace the dict rather than mutate in place: the tree diffs by
            # content against the message objects it holds, so an in-place edit
            # would be invisible (no fork recorded, no write-back).
            edited = dict(msg)
            edited['content'] = new_text
            ctx.messages[ai] = edited

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


def prompt_messages_overlay(screen, messages, *,
                            context_size=0, prompt_tokens=0, sample_chars=0,
                            refetch=None, revision=None):
    """interactive messages overlay.
    returns (messages, new_tokens_estimate, modified). modified is True when
    the user edited the conversation (so the caller writes it back).
    refetch (optional) returns the host's current messages; when the turn
    appends under the overlay it pulls the new tail in so the view live-
    updates (see _live_sync)."""
    if not messages:
        return messages, 0, False

    ctx = MsgOverlayCtx(messages,
                         context_size=context_size,
                         prompt_tokens=prompt_tokens,
                         sample_chars=sample_chars)

    old_attrs = termios.tcgetattr(screen._tty_fd)
    orig_handler = signal.getsignal(signal.SIGWINCH)

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        ctx.resize_pending = True
        ctx.first_draw = True

    # live refresh: the host bumps a mutation counter (over the socket) each
    # time a turn appends under us. revision() reads it; the tick loop polls
    # for a change and flags dirty. None -> no live refresh (e.g. opened as a
    # sub-overlay with no client in scope).
    last_rev = None
    if revision is not None:
        last_rev = revision()

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

            if revision is not None:
                cur_rev = revision()
                if cur_rev != last_rev:
                    last_rev = cur_rev
                    ctx.dirty = True

            if ctx.dirty:
                ctx.dirty = False
                _live_sync(ctx, refetch)
                _refresh_view_after_external_mutation(ctx)
                overlay_redraw(ctx, screen._rows, screen._cols)

            rlist, _, _ = select.select([screen._tty_fd], [], [], 0.05)
            if not rlist:
                continue
            key = read_key(screen._tty_fd)

            mouse = parse_mouse(key)
            if mouse is not None:
                if not (ctx.filter_mode or ctx.search_mode or ctx.instruction_mode):
                    nv = len(ctx.view)
                    if mouse[0] == 'wheel_up':
                        ctx.selected_idx = max(0, ctx.selected_idx - 1)
                    elif mouse[0] == 'wheel_down':
                        ctx.selected_idx = min(max(0, nv - 1), ctx.selected_idx + 1)
                    overlay_redraw(ctx, screen._rows, screen._cols)
                continue

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

                # hand off to the streaming popup so the user sees the model's
                # tokens and reasoning land as they're produced instead of
                # staring at a frozen screen.
                _run_rewrite(ctx, screen, result, header=f'instruction: {result}')
                overlay_redraw(ctx, screen._rows, screen._cols)
                continue

            action = overlay_nav_key(ctx, key, screen._rows, screen)
            ctx.prev_key = key

            if action == 'close':
                break

            overlay_redraw(ctx, screen._rows, screen._cols)

    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()

    return messages, ctx.tokens_est, ctx.modified
