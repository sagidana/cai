"""State classes and pure logic helpers for the TUI and overlays."""

import json as _json
import re
from enum import Enum, auto

from .ansi import ansi_strip


# ── Mode enum ────────────────────────────────────────────────────────────────

class Mode(Enum):
    NORMAL      = auto()
    INSERT      = auto()
    VISUAL      = auto()
    VISUAL_LINE = auto()
    COMMAND     = auto()
    SEARCH      = auto()   # typing a search pattern (/ or ?)


class TUIState:
    """Mutable state bag for the alternate-screen TUI."""
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
    )

    def __init__(self) -> None:
        self.mode: Mode = Mode.INSERT

        # viewport / navigation
        self.viewport_offset: int = 0
        self.cursor_row: int = 0       # line in content buffer (for normal/visual)
        self.cursor_col: int = 0

        # visual selection anchors
        self.visual_anchor_row: int = 0
        self.visual_anchor_col: int = 0

        # search
        self.search_direction: int = 1    # 1 = forward, -1 = backward
        self.search_buf: list[str] = []
        self.search_pattern: str = ''
        self.search_matches: list[int] = []   # line indices
        self.search_match_idx: int = -1

        # Origin saved when entering SEARCH mode so Esc can restore
        # cursor/viewport if the user cancels mid-pattern.
        self.search_origin_row: int = 0
        self.search_origin_col: int = 0
        self.search_origin_viewport: int = 0

        # command mode
        self.command_buf: list[str] = []
        self.command_cursor: int = 0
        # Per-session history of submitted commands, newest-first.
        # command_history_pos == -1 means "not browsing"; otherwise it's the
        # index into command_history of the currently-displayed entry.
        self.command_history: list[str] = []
        self.command_history_pos: int = -1

        # multi-key sequences (e.g. gg)
        self.pending_key: str = ''

        # auto-scroll: pin viewport to bottom when True
        self.auto_scroll: bool = True

        # yank register (internal clipboard)
        self.yank_register: str = ''

        # double Ctrl-C to quit
        self.last_ctrl_c: float = 0.0


class _SubmitException(Exception):
    def __init__(self, value: str):
        self.value = value


class _CommandException(Exception):
    """Raised when command mode produces a command to execute."""
    def __init__(self, value: str):
        self.value = value


class _OverlayCtx:
    """Mutable state bag for the context-overlay event loop."""
    __slots__ = (
        'messages', 'view', 'selected_idx',
        'search_mode', 'search_direction', 'search_buf',
        'search_pattern', 'search_matches', 'search_match_idx', 'pre_search_idx',
        'filter_mode', 'filter_buf', 'filter_pattern',
        'filter_history', 'filter_history_pos',
        'scroll', 'forced_scroll', 'prev_key',
        'resize_pending', 'prev_lines', 'first_draw',
        'context_size', 'base_chars', 'prompt_tokens', 'tokens_est',
    )

    def __init__(self, messages: list, context_size: int, prompt_tokens: int) -> None:
        self.messages   = messages
        self.view: list[int] = list(range(len(messages)))
        self.selected_idx = 0

        self.search_mode      = False
        self.search_direction = 1
        self.search_buf:  list[str] = []
        self.search_pattern   = ''
        self.search_matches: list[int] = []
        self.search_match_idx = -1
        self.pre_search_idx   = 0

        self.filter_mode      = False
        self.filter_buf:  list[str] = []
        self.filter_pattern   = ''
        self.filter_history: list[str] = []
        self.filter_history_pos = -1

        self.scroll: int = 0
        self.forced_scroll: 'int | None' = None
        self.prev_key  = ''

        self.resize_pending = False
        self.prev_lines: dict[int, str] = {}
        self.first_draw = True

        self.context_size  = context_size
        self.base_chars    = sum(len(_overlay_msg_text(m)) for m in messages) or 1
        self.prompt_tokens = prompt_tokens
        self.tokens_est    = prompt_tokens


def _overlay_msg_text(msg: dict) -> str:
    """Flat text representation of a message (for search / filter matching)."""
    content  = msg.get('content', '')
    tc_parts = []
    for tc in (msg.get('tool_calls') or []):
        name = tc.get('function', {}).get('name', '?')
        args = tc.get('function', {}).get('arguments', '')
        tc_parts.append(f'{name}({args})')
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = _json.dumps(content, indent=2)
    else:
        text = str(content) if content is not None else ''
    if not tc_parts:
        return text
    tc_str = ' | '.join(tc_parts)
    return f'[{tc_str}] {text}' if text else f'[{tc_str}]'


def _overlay_visible_n(ctx: _OverlayCtx, rows: int) -> int:
    """Number of message rows that fit in the overlay at current terminal height."""
    overhead  = 4
    max_box_h = max(overhead + 1, int(rows * 0.85))
    return max(1, min(len(ctx.messages), max_box_h - overhead))


def _overlay_parse_filter(raw: str) -> tuple:
    """Parse a filter pattern for optional field flags (~r role, ~c content)."""
    if raw.startswith('~r '):
        return ('role', raw[3:].strip())
    if raw.startswith('~c '):
        return ('content', raw[3:].strip())
    return ('all', raw)


def _overlay_make_rx(pattern: str):
    """Compile pattern to a regex, falling back to a literal match on error."""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(pattern), re.IGNORECASE)


def _overlay_apply_filter(ctx: _OverlayCtx, pattern: str) -> None:
    """Rebuild ctx.view using *pattern* as a filter. Clears when pattern is empty."""
    ctx.filter_pattern = pattern

    if not pattern:
        ctx.view = list(range(len(ctx.messages)))
    else:
        field, pat = _overlay_parse_filter(pattern)
        rx = _overlay_make_rx(pat) if pat else None

        if rx is None:
            ctx.view = list(range(len(ctx.messages)))
        elif field == 'role':
            ctx.view = [
                i for i, msg in enumerate(ctx.messages)
                if rx.search(msg.get('role', ''))
            ]
        elif field == 'content':
            ctx.view = [
                i for i, msg in enumerate(ctx.messages)
                if rx.search(_overlay_msg_text(msg))
            ]
        else:
            ctx.view = [
                i for i, msg in enumerate(ctx.messages)
                if rx.search(msg.get('role', '')) or rx.search(_overlay_msg_text(msg))
            ]

    ctx.selected_idx     = 0
    ctx.scroll           = 0
    ctx.search_matches   = _overlay_find_matches(ctx)
    ctx.search_match_idx = -1


def _overlay_find_matches(ctx: _OverlayCtx) -> list:
    """Return view-positions whose messages match ctx.search_pattern."""
    if not ctx.search_pattern:
        return []
    rx = _overlay_make_rx(ctx.search_pattern)
    return [
        pos for pos, ai in enumerate(ctx.view)
        if rx.search(ctx.messages[ai].get('role', ''))
        or rx.search(_overlay_msg_text(ctx.messages[ai]))
    ]


def _overlay_sync_search_cursor(ctx: _OverlayCtx) -> None:
    """Move ctx.selected_idx to the nearest search match."""
    if not ctx.search_matches:
        ctx.search_match_idx = -1
        ctx.selected_idx     = ctx.pre_search_idx
        return

    if ctx.search_direction == 1:
        for i, m in enumerate(ctx.search_matches):
            if m >= ctx.pre_search_idx:
                ctx.search_match_idx = i
                ctx.selected_idx     = m
                return
        ctx.search_match_idx = 0
    else:
        for i in range(len(ctx.search_matches) - 1, -1, -1):
            if ctx.search_matches[i] <= ctx.pre_search_idx:
                ctx.search_match_idx = i
                ctx.selected_idx     = ctx.search_matches[i]
                return
        ctx.search_match_idx = len(ctx.search_matches) - 1

    ctx.selected_idx = ctx.search_matches[ctx.search_match_idx]


def _overlay_recompute_tokens(ctx) -> None:
    """Re-estimate token count after a message was edited or pruned.

    Accepts either an ``_OverlayCtx`` or a ``_MsgOverlayCtx`` — both expose the
    same set of attributes used here.
    """
    if not ctx.prompt_tokens or not ctx.base_chars:
        return
    new_chars      = sum(len(_overlay_msg_text(m)) for m in ctx.messages) or 1
    ctx.tokens_est = max(0, round(ctx.prompt_tokens * new_chars / ctx.base_chars))
    ctx.prev_lines.clear()
    ctx.first_draw = True


# ── :messages overlay state ──────────────────────────────────────────────────

class _MsgRow:
    """One render row in the :messages overlay.

    kind is one of:
      'header'      — the first (and only, if collapsed) row of a message
      'body'        — a wrapped content line under an expanded message
      'pair-head'   — collapsed assistant+tool_result pair, one row
    """
    __slots__ = ('msg_idx', 'kind', 'text', 'fold_group', 'partner_idx')

    def __init__(self, msg_idx: int, kind: str, text: str,
                 fold_group: str | None = None, partner_idx: int | None = None) -> None:
        self.msg_idx     = msg_idx
        self.kind        = kind
        self.text        = text
        self.fold_group  = fold_group
        self.partner_idx = partner_idx


def _msg_pair_partner(messages: list, i: int) -> int | None:
    """Return the index of the paired tool_result (or assistant tool_call).

    Assistant with a single tool_call at messages[i] pairs with the next
    tool message whose tool_call_id matches. The reverse direction also
    resolves: given a tool message, find the assistant preceding it that
    holds the matching tool_call id. Returns None when no pair exists.
    """
    if not (0 <= i < len(messages)):
        return None
    m = messages[i]
    role = m.get('role')
    if role == 'assistant':
        calls = m.get('tool_calls') or []
        if not calls:
            return None
        call_id = calls[0].get('id') if calls else None
        if not call_id:
            return None
        j = i + 1
        if j < len(messages) and messages[j].get('role') == 'tool' \
                and messages[j].get('tool_call_id') == call_id:
            return j
        return None
    if role == 'tool':
        call_id = m.get('tool_call_id')
        if not call_id:
            return None
        j = i - 1
        if j >= 0 and messages[j].get('role') == 'assistant':
            calls = messages[j].get('tool_calls') or []
            for c in calls:
                if c.get('id') == call_id:
                    return j
        return None
    return None


def _msg_pair_summary(messages: list, assistant_idx: int, tool_idx: int) -> str:
    """One-line summary for a collapsed assistant+tool_result pair."""
    a = messages[assistant_idx]
    calls = a.get('tool_calls') or []
    if calls:
        name = calls[0].get('function', {}).get('name', '?')
    else:
        name = '?'
    t = messages[tool_idx]
    result = t.get('content') or ''
    if isinstance(result, list):
        result = _json.dumps(result)
    preview = str(result).replace('\n', ' ').replace('\r', ' ')[:80]
    return f'⚙ {name} → {preview}'


def _msg_header_preview(msg: dict, width: int) -> str:
    """Short one-line header summary of a message."""
    raw = _overlay_msg_text(msg)
    return ansi_strip(raw.replace('\n', ' ').replace('\r', ' '))[:width]


def _msg_body_lines(msg: dict, width: int) -> list[str]:
    """Full-content body lines wrapped to width (for expanded messages)."""
    parts: list[str] = []
    content = msg.get('content', '')
    if isinstance(content, list):
        text = _json.dumps(content, indent=2)
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
            text = '[' + ' | '.join(tc_parts) + ']' + (('\n' + text) if text else '')
    for raw_line in text.split('\n'):
        if not raw_line:
            parts.append('')
            continue
        # Wrap on width; don't break escape codes (content here is plain text).
        while len(raw_line) > width:
            parts.append(raw_line[:width])
            raw_line = raw_line[width:]
        parts.append(raw_line)
    return parts or ['']


class _MsgOverlayCtx:
    """Mutable state bag for the :messages overlay."""
    __slots__ = (
        # shared with caller, mutated in place
        'messages', 'tracker', 'model',

        # filter / view (same semantics as _OverlayCtx.view)
        'view', 'selected_idx', 'scroll', 'forced_scroll',

        # fold state — one bit per message, keyed by id(msg).
        # Message is collapsed (body hidden) when id(msg) is in this set.
        'collapsed',

        # selection
        'marks',          # set[int] msg indices
        'visual_mode',    # bool
        'visual_anchor',  # int view position (mirrors selected_idx)

        # Shared search / filter plumbing (attr names align with _OverlayCtx
        # so helpers in this module can be reused for both overlays).
        'search_mode', 'search_direction', 'search_buf',
        'search_pattern', 'search_matches', 'search_match_idx', 'pre_search_idx',
        'filter_mode', 'filter_buf', 'filter_pattern',
        'filter_history', 'filter_history_pos',

        # Ad-hoc LLM-transform instruction input (`!`)
        'instruction_mode', 'instruction_buf',

        # Yank register (in-memory, session-local)
        'yank_register',

        # Event-loop plumbing
        'prev_key', 'resize_pending', 'prev_lines', 'first_draw',

        # Token estimate (same fields as _OverlayCtx)
        'context_size', 'base_chars', 'prompt_tokens', 'tokens_est',

        # Pending status-line message (errors / feedback)
        'status_flash',
    )

    def __init__(self, messages: list, tracker, model: str,
                 context_size: int = 0, prompt_tokens: int = 0) -> None:
        self.messages   = messages
        self.tracker    = tracker
        self.model      = model

        self.view: list[int] = list(range(len(messages)))
        self.selected_idx    = 0
        self.scroll: int     = 0
        self.forced_scroll: 'int | None' = None

        # Everything defaults to folded — one row per message matches the
        # familiar :context look and keeps long content from overwhelming
        # the view. Press Tab/za to expand what you want.
        self.collapsed: set[int] = {id(m) for m in messages}

        self.marks: set[int]    = set()
        self.visual_mode        = False
        self.visual_anchor      = 0

        self.search_mode        = False
        self.search_direction   = 1
        self.search_buf: list[str] = []
        self.search_pattern     = ''
        self.search_matches: list[int] = []
        self.search_match_idx   = -1
        self.pre_search_idx     = 0

        self.filter_mode        = False
        self.filter_buf: list[str] = []
        self.filter_pattern     = ''
        self.filter_history: list[str] = []
        self.filter_history_pos = -1

        self.instruction_mode   = False
        self.instruction_buf: list[str] = []

        self.yank_register: list[dict] = []

        self.prev_key           = ''
        self.resize_pending     = False
        self.prev_lines: dict[int, str] = {}
        self.first_draw         = True

        self.context_size  = context_size
        self.base_chars    = sum(len(_overlay_msg_text(m)) for m in messages) or 1
        self.prompt_tokens = prompt_tokens
        self.tokens_est    = prompt_tokens

        self.status_flash  = ''


def _msg_is_folded(ctx: _MsgOverlayCtx, msg_idx: int) -> bool:
    """True if the message is currently rendered as a single-row header."""
    return id(ctx.messages[msg_idx]) in ctx.collapsed


def _msg_effective_selection(ctx: _MsgOverlayCtx) -> list[int]:
    """Ordered list of message indices currently selected (marks ∪ visual)."""
    sel = set(ctx.marks)
    if ctx.visual_mode and ctx.view:
        lo = min(ctx.visual_anchor, ctx.selected_idx)
        hi = max(ctx.visual_anchor, ctx.selected_idx)
        for pos in range(lo, hi + 1):
            if 0 <= pos < len(ctx.view):
                sel.add(ctx.view[pos])
    # If nothing is explicitly selected, operate on the message under the
    # cursor (matches vim's "no range means current line" semantics).
    if not sel and ctx.view:
        sel.add(ctx.view[ctx.selected_idx])
    return sorted(i for i in sel if 0 <= i < len(ctx.messages))
