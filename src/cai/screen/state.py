"""State classes and pure logic helpers for the context overlay."""

import json as _json
import re

from .ansi import ansi_strip


class _SubmitException(Exception):
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


def _overlay_recompute_tokens(ctx: _OverlayCtx) -> None:
    """Re-estimate token count after a message was edited or pruned."""
    if not ctx.prompt_tokens or not ctx.base_chars:
        return
    new_chars      = sum(len(_overlay_msg_text(m)) for m in ctx.messages) or 1
    ctx.tokens_est = max(0, round(ctx.prompt_tokens * new_chars / ctx.base_chars))
    ctx.prev_lines.clear()
    ctx.first_draw = True
