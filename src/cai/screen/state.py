"""state classes and pure logic helpers for the TUI and overlays."""

import json
import re
from enum import Enum, auto

from cai import usage
from .ansi import ansi_strip


class Mode(Enum):
    NORMAL      = auto()
    INSERT      = auto()
    VISUAL      = auto()
    VISUAL_LINE = auto()
    COMMAND     = auto()
    SEARCH      = auto()   # typing a search pattern (/ or ?)


class TUIState:
    """mutable state bag for the alternate-screen TUI."""
    __slots__ = (
        'mode',
        'viewport_offset', 'cursor_row',
        'visual_anchor_row', 'visual_anchor_col', 'cursor_col',
        'search_direction', 'search_buf', 'search_pattern',
        'search_matches', 'search_match_idx',
        'search_origin_row', 'search_origin_col', 'search_origin_viewport',
        'command_buf', 'command_cursor',
        'command_history', 'command_history_pos',
        'pending_key', 'auto_scroll', 'yank_register',
        'last_ctrl_c',
        'mouse_press_row', 'mouse_press_col', 'mouse_drag_active',
    )

    def __init__(self):
        self.mode = Mode.INSERT

        # viewport / navigation
        self.viewport_offset = 0
        self.cursor_row = 0       # line in content buffer (for normal/visual)
        self.cursor_col = 0

        # visual selection anchors
        self.visual_anchor_row = 0
        self.visual_anchor_col = 0

        # search
        self.search_direction = 1    # 1 = forward, -1 = backward
        self.search_buf = []
        self.search_pattern = ''
        self.search_matches = []     # (row, start, end)
        self.search_match_idx = -1

        # origin saved when entering SEARCH mode so Esc can restore
        # cursor/viewport if the user cancels mid-pattern.
        self.search_origin_row = 0
        self.search_origin_col = 0
        self.search_origin_viewport = 0

        # command mode
        self.command_buf = []
        self.command_cursor = 0
        # per-session history of submitted commands, newest-first.
        # command_history_pos == -1 means "not browsing"; otherwise it's the
        # index into command_history of the currently-displayed entry.
        self.command_history = []
        self.command_history_pos = -1

        # multi-key sequences (e.g. gg)
        self.pending_key = ''

        # auto-scroll: pin viewport to bottom when True
        self.auto_scroll = True

        # yank register (internal clipboard)
        self.yank_register = ''

        # double Ctrl-C to quit
        self.last_ctrl_c = 0.0

        # mouse: buffer position of the last press in the content area, and
        # whether a drag has promoted it into a visual selection.
        self.mouse_press_row = 0
        self.mouse_press_col = 0
        self.mouse_drag_active = False


class SubmitException(Exception):
    def __init__(self, value):
        self.value = value


class CommandException(Exception):
    """raised when command mode produces a command to execute."""
    def __init__(self, value):
        self.value = value


def _overlay_msg_text(msg):
    """flat text representation of a message (for search / filter matching)."""
    content = msg.get('content', '')
    tc_parts = []
    for tc in (msg.get('tool_calls') or []):
        name = tc.get('function', {}).get('name', '?')
        args = tc.get('function', {}).get('arguments', '')
        tc_parts.append(f'{name}({args})')

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = json.dumps(content, indent=2)
    elif content is None:
        text = ''
    else:
        text = str(content)

    if not tc_parts:
        return text
    tc_str = ' | '.join(tc_parts)
    if text:
        return f'[{tc_str}] {text}'
    return f'[{tc_str}]'


def _overlay_parse_filter(raw):
    """parse a filter pattern for optional field flags (~r role, ~c content)."""
    if raw.startswith('~r '):
        return ('role', raw[3:].strip())
    if raw.startswith('~c '):
        return ('content', raw[3:].strip())
    return ('all', raw)


def _overlay_make_rx(pattern):
    """compile pattern to a regex, falling back to a literal match on error."""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(pattern), re.IGNORECASE)


def _overlay_apply_filter(ctx, pattern):
    """rebuild ctx.view using pattern as a filter. clears when pattern is empty."""
    ctx.filter_pattern = pattern

    if not pattern:
        ctx.view = list(range(len(ctx.messages)))
    else:
        field, pat = _overlay_parse_filter(pattern)
        rx = None
        if pat:
            rx = _overlay_make_rx(pat)

        if rx is None:
            ctx.view = list(range(len(ctx.messages)))
        else:
            view = []
            for i, msg in enumerate(ctx.messages):
                role_hit = rx.search(msg.get('role', ''))
                content_hit = rx.search(_overlay_msg_text(msg))
                if field == 'role' and not role_hit: continue
                if field == 'content' and not content_hit: continue
                if field == 'all' and not role_hit and not content_hit: continue
                view.append(i)
            ctx.view = view

    ctx.selected_idx = 0
    ctx.scroll = 0
    ctx.search_matches = _overlay_find_matches(ctx)
    ctx.search_match_idx = -1


def _overlay_find_matches(ctx):
    """return view-positions whose messages match ctx.search_pattern."""
    if not ctx.search_pattern: return []

    rx = _overlay_make_rx(ctx.search_pattern)
    matches = []
    for pos, ai in enumerate(ctx.view):
        msg = ctx.messages[ai]
        if rx.search(msg.get('role', '')) or rx.search(_overlay_msg_text(msg)):
            matches.append(pos)
    return matches


def _overlay_sync_search_cursor(ctx):
    """move ctx.selected_idx to the nearest search match."""
    if not ctx.search_matches:
        ctx.search_match_idx = -1
        ctx.selected_idx = ctx.pre_search_idx
        return

    if ctx.search_direction == 1:
        for i, m in enumerate(ctx.search_matches):
            if m < ctx.pre_search_idx: continue
            ctx.search_match_idx = i
            ctx.selected_idx = m
            return
        ctx.search_match_idx = 0
    else:
        for i in range(len(ctx.search_matches) - 1, -1, -1):
            if ctx.search_matches[i] > ctx.pre_search_idx: continue
            ctx.search_match_idx = i
            ctx.selected_idx = ctx.search_matches[i]
            return
        ctx.search_match_idx = len(ctx.search_matches) - 1

    ctx.selected_idx = ctx.search_matches[ctx.search_match_idx]


def _overlay_recompute_tokens(ctx):
    """re-estimate token count after a message was edited or pruned, through the
    shared estimator so the overlay agrees with the live status line."""
    ctx.tokens_est = usage.estimate_tokens(ctx.messages,
                                             ctx.prompt_tokens,
                                             ctx.base_chars)
    ctx.prev_lines.clear()
    ctx.first_draw = True


class MsgRow:
    """one render row in the :messages overlay.

    kind is one of:
      'header'    - the first (and only, if collapsed) row of a message
      'body'      - a wrapped content line under an expanded message
      'pair-head' - collapsed assistant+tool_result pair, one row"""
    __slots__ = ('msg_idx', 'kind', 'text', 'fold_group', 'partner_idx')

    def __init__(self, msg_idx, kind, text, fold_group=None, partner_idx=None):
        self.msg_idx = msg_idx
        self.kind = kind
        self.text = text
        self.fold_group = fold_group
        self.partner_idx = partner_idx


def _msg_pair_partner(messages, i):
    """return the index of the paired tool_result (or assistant tool_call).

    assistant with a single tool_call at messages[i] pairs with the next
    tool message whose tool_call_id matches. the reverse direction also
    resolves: given a tool message, find the assistant preceding it that
    holds the matching tool_call id. returns None when no pair exists."""
    if not (0 <= i < len(messages)): return None

    m = messages[i]
    role = m.get('role')

    if role == 'assistant':
        calls = m.get('tool_calls') or []
        if not calls: return None
        call_id = calls[0].get('id')
        if not call_id: return None

        j = i + 1
        if j >= len(messages): return None
        if messages[j].get('role') != 'tool': return None
        if messages[j].get('tool_call_id') != call_id: return None
        return j

    if role == 'tool':
        call_id = m.get('tool_call_id')
        if not call_id: return None

        j = i - 1
        if j < 0: return None
        if messages[j].get('role') != 'assistant': return None
        for c in (messages[j].get('tool_calls') or []):
            if c.get('id') == call_id:
                return j
        return None

    return None


def _msg_pair_summary(messages, assistant_idx, tool_idx):
    """one-line summary for a collapsed assistant+tool_result pair."""
    calls = messages[assistant_idx].get('tool_calls') or []
    name = '?'
    if calls:
        name = calls[0].get('function', {}).get('name', '?')

    result = messages[tool_idx].get('content') or ''
    if isinstance(result, list):
        result = json.dumps(result)
    preview = str(result).replace('\n', ' ').replace('\r', ' ')[:80]
    return f'⚙ {name} → {preview}'


def _msg_header_preview(msg, width):
    """short one-line header summary of a message."""
    raw = _overlay_msg_text(msg)
    return ansi_strip(raw.replace('\n', ' ').replace('\r', ' '))[:width]


def _msg_body_lines(msg, width):
    """full-content body lines wrapped to width (for expanded messages)."""
    parts = []
    content = msg.get('content', '')
    if isinstance(content, list):
        text = json.dumps(content, indent=2)
    elif content is None:
        text = ''
    else:
        text = str(content)

    if msg.get('tool_calls'):
        tc_parts = []
        for tc in msg['tool_calls']:
            name = tc.get('function', {}).get('name', '?')
            args = tc.get('function', {}).get('arguments', '')
            tc_parts.append(f'{name}({args})')
        if tc_parts:
            header = '[' + ' | '.join(tc_parts) + ']'
            if text:
                text = header + '\n' + text
            else:
                text = header

    for raw_line in text.split('\n'):
        if not raw_line:
            parts.append('')
            continue
        # wrap on width; don't break escape codes (content here is plain text).
        while len(raw_line) > width:
            parts.append(raw_line[:width])
            raw_line = raw_line[width:]
        parts.append(raw_line)

    if not parts:
        return ['']
    return parts


class MsgOverlayCtx:
    """mutable state bag for the :messages overlay."""
    __slots__ = (
        # shared with caller, mutated in place
        'messages',
        # True once the user edits/deletes/forks in the overlay (vs the view
        # merely growing from live-sync); gates the write-back on close.
        'modified',

        # filter / view (view holds indices into messages)
        'view', 'selected_idx', 'scroll', 'forced_scroll',

        # fold state - one bit per message, keyed by id(msg).
        # message is collapsed (body hidden) when id(msg) is in this set.
        'collapsed',

        # ids of every message the overlay has already seen, so a live
        # refresh can tell genuinely-new arrivals (collapse them) from
        # messages the user manually expanded (leave them expanded).
        'seen_ids',

        # selection
        'visual_mode',    # bool
        'visual_anchor',  # int view position (mirrors selected_idx)

        # search / filter plumbing (consumed by the _overlay_* helpers).
        'search_mode', 'search_direction', 'search_buf',
        'search_pattern', 'search_matches', 'search_match_idx', 'pre_search_idx',
        'filter_mode', 'filter_buf', 'filter_pattern',
        'filter_history', 'filter_history_pos',

        # ad-hoc LLM-transform instruction input (`!`)
        'instruction_mode', 'instruction_buf',

        # yank register (in-memory, session-local)
        'yank_register',

        # event-loop plumbing
        'prev_key', 'resize_pending', 'prev_lines', 'first_draw',

        # live-refresh: set by the messages_mutated hook (worker thread)
        # when ctx.messages has changed under the overlay; the event loop
        # drains it on the next tick by rebuilding the view and redrawing.
        'dirty',

        # token estimate
        'context_size', 'base_chars', 'prompt_tokens', 'tokens_est',

        # pending status-line message (errors / feedback)
        'status_flash',
    )

    def __init__(self, messages, context_size=0, prompt_tokens=0, sample_chars=0):
        self.messages = messages
        self.modified = False

        self.view = list(range(len(messages)))
        self.selected_idx = 0
        self.scroll = 0
        self.forced_scroll = None

        # everything defaults to folded - one row per message keeps long
        # content from overwhelming the view. press Tab/za to expand.
        self.collapsed = {id(m) for m in messages}
        self.seen_ids = {id(m) for m in messages}

        self.visual_mode = False
        self.visual_anchor = 0

        self.search_mode = False
        self.search_direction = 1
        self.search_buf = []
        self.search_pattern = ''
        self.search_matches = []
        self.search_match_idx = -1
        self.pre_search_idx = 0

        self.filter_mode = False
        self.filter_buf = []
        self.filter_pattern = ''
        self.filter_history = []
        self.filter_history_pos = -1

        self.instruction_mode = False
        self.instruction_buf = []

        self.yank_register = []

        self.prev_key = ''
        self.resize_pending = False
        self.prev_lines = {}
        self.first_draw = True
        self.dirty = False

        # base_chars is the calibration anchor: the conversation's char size
        # when prompt_tokens was last really measured by the host. fixed for the
        # overlay's lifetime so edits scale against a true sample, not a moving
        # baseline. 0 (no sample yet) -> the estimator falls back to chars/4.
        self.context_size = context_size
        self.base_chars = sample_chars
        self.prompt_tokens = prompt_tokens
        self.tokens_est = usage.estimate_tokens(messages, prompt_tokens, sample_chars)

        self.status_flash = ''


def _msg_is_folded(ctx, msg_idx):
    """True if the message is currently rendered as a single-row header."""
    return id(ctx.messages[msg_idx]) in ctx.collapsed


def _msg_effective_selection(ctx):
    """ordered list of message indices currently selected (visual-line)."""
    sel = set()
    if ctx.visual_mode and ctx.view:
        lo = min(ctx.visual_anchor, ctx.selected_idx)
        hi = max(ctx.visual_anchor, ctx.selected_idx)
        for pos in range(lo, hi + 1):
            if 0 <= pos < len(ctx.view):
                sel.add(ctx.view[pos])
    # if nothing is explicitly selected, operate on the message under the
    # cursor (matches vim's "no range means current line" semantics).
    if not sel and ctx.view:
        sel.add(ctx.view[ctx.selected_idx])
    return sorted(i for i in sel if 0 <= i < len(ctx.messages))
