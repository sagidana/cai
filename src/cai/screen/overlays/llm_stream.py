"""Streaming progress popup for the :messages overlay's ad-hoc LLM rewrite.

Opened from the ``!`` handler (ad-hoc instruction). Drives the rewrite's
``stream_callback`` with every event the model produces — content, reasoning,
and tool calls — appending each to an in-memory buffer and redrawing, so the
user sees live progress and the model's workings instead of a frozen screen.
"""

import json
import select
import shutil
import signal
import sys
import termios
import tty

from ..ansi import (
    ALT_ENTER, ALT_EXIT,
    CUR_HIDE,
    ERASE_SCREEN,
    SGR_RESET, SGR_DIM_GRAY, SGR_BOLD, SGR_YELLOW, SGR_GREEN, SGR_BOLD_RED,
    SGR_CYAN, SGR_MAGENTA,
    ansi_pad, wrap_ansi,
    cur_move,
)
from ..input import read_key


class StreamState:
    __slots__ = (
        'title', 'header', 'n_selected',
        'content', 'reasoning', 'tool_calls',
        'status', 'status_color',
        'resize_pending', 'first_draw', 'prev_lines',
    )

    def __init__(self, title, header, n_selected):
        self.title = title
        self.header = header
        self.n_selected = n_selected
        self.content = []
        self.reasoning = []
        self.tool_calls = []
        self.status = 'connecting...'
        self.status_color = SGR_DIM_GRAY
        self.resize_pending = False
        self.first_draw = True
        self.prev_lines = {}


def _transcript_lines(state, width):
    """Render the model's workings into styled, wrapped lines.

    Reasoning is dimmed, content is plain, and each tool call is shown on a
    cyan ``⚙ name(args)`` line — so the user sees the model think, call
    tools, and answer in the order it happens.
    """
    lines = []

    reasoning = ''.join(state.reasoning)
    if reasoning:
        lines.append(f'{SGR_DIM_GRAY}{SGR_BOLD}⋯ reasoning{SGR_RESET}')
        for ln in wrap_ansi(reasoning, width):
            lines.append(f'{SGR_DIM_GRAY}{ln}{SGR_RESET}')

    for tc in state.tool_calls:
        fn = tc.get('function') or {}
        name = fn.get('name') or '?'
        args = fn.get('arguments') or ''
        if isinstance(args, str):
            try:
                args = json.dumps(json.loads(args))
            except (ValueError, TypeError):
                pass
        for ln in wrap_ansi(f'⚙ {name}({args})', width):
            lines.append(f'{SGR_CYAN}{ln}{SGR_RESET}')

    content = ''.join(state.content)
    if content:
        if lines:
            lines.append('')
        for ln in wrap_ansi(content, width):
            lines.append(ln)

    return lines


def _draw(screen, state):
    rows = screen._rows
    cols = screen._cols

    inner_w = max(40, int(cols * 0.9) - 2)
    box_w = inner_w + 2

    # Layout: top border | header | sep | body (N) | sep | status | bottom
    overhead = 6
    max_box_h = max(overhead + 3, int(rows * 0.9))
    body_h = max(3, max_box_h - overhead)
    box_h = body_h + overhead

    start_r = max(1, (rows - box_h) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)

    H = '─'
    TL = '┌'
    TR = '┐'
    BL = '└'
    BR = '┘'
    VL = '│'
    ML = '├'
    MR = '┤'
    h_line = H * inner_w

    title = f'  {state.title}  '[:inner_w]
    pad_l = max(0, (inner_w - len(title)) // 2)
    pad_r = max(0, inner_w - len(title) - pad_l)
    title_border = f'{TL}{H * pad_l}{title}{H * pad_r}{TR}'

    new_lines = {}

    def put(row_off, text):
        r = start_r + row_off
        if 1 <= r <= rows:
            new_lines[row_off] = (r, text)

    put(0, title_border)

    # Header: the instruction / transform + selection count
    info = (f' {SGR_BOLD}{state.header}{SGR_RESET}'
            f'  {SGR_DIM_GRAY}(input: {state.n_selected} msg){SGR_RESET}')
    put(1, f'{VL}{ansi_pad(info, inner_w)}{VL}')

    put(2, f'{ML}{h_line}{MR}')

    # Body: the model's workings (reasoning + tool calls + content), tail.
    lines = _transcript_lines(state, inner_w - 2)
    visible = lines
    if len(lines) > body_h:
        visible = lines[-body_h:]
    for i in range(body_h):
        if i < len(visible):
            body_cell = ansi_pad(f' {visible[i]}', inner_w)
        else:
            body_cell = ' ' * inner_w
        put(3 + i, f'{VL}{body_cell}{VL}')

    put(3 + body_h, f'{ML}{h_line}{MR}')

    # Status line — status + a breakdown of what has streamed so far.
    n_content = len(''.join(state.content))
    n_reason = len(''.join(state.reasoning))
    parts = [f'{n_content} chars']
    if n_reason:
        parts.append(f'{n_reason} reasoning')
    if state.tool_calls:
        parts.append(f'{len(state.tool_calls)} tool call(s)')
    meta = '  '.join(parts)
    status_text = (f' {state.status_color}{state.status}{SGR_RESET}'
                   f'  {SGR_DIM_GRAY}({meta}){SGR_RESET}')
    put(3 + body_h + 1, f'{VL}{ansi_pad(status_text, inner_w)}{VL}')

    put(3 + body_h + 2, f'{BL}{h_line}{BR}')

    out = []
    if state.first_draw:
        sys.stdout.write(ERASE_SCREEN)
        for row_off, (r, text) in new_lines.items():
            out.append(f'{cur_move(r, start_c)}{text}')
        state.first_draw = False
    else:
        for row_off, (r, text) in new_lines.items():
            if state.prev_lines.get(row_off) != text:
                out.append(f'{cur_move(r, start_c)}{text}')
    state.prev_lines = {}
    for row_off, (r, text) in new_lines.items():
        state.prev_lines[row_off] = text

    out.append(CUR_HIDE)
    sys.stdout.write(''.join(out))
    sys.stdout.flush()


def run_streaming_transform(screen, fn, name, selected, kwargs, *, header=''):
    """Run a streaming message operation with a live progress popup.

    Calls ``fn(selected, stream_callback=..., **kwargs)``. The callback
    receives ``(content, reasoning, tool_calls)`` per event and redraws the
    popup so the user sees the model think and answer as it happens. Blocks
    until the response completes, then waits for a keypress. Returns the
    replacement list, or None on error (the popup will have shown it).
    """
    state = StreamState(f'{name}', header or name, len(selected))

    old_attrs = termios.tcgetattr(screen._tty_fd)
    orig_handler = signal.getsignal(signal.SIGWINCH)

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        state.resize_pending = True
        state.first_draw = True

    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()

    result = None
    try:
        signal.signal(signal.SIGWINCH, _on_resize)
        tty.setraw(screen._tty_fd)
        state.status = 'waiting for model...'
        state.status_color = SGR_YELLOW
        _draw(screen, state)

        def on_event(content=None, reasoning=None, tool_calls=None):
            # SIGWINCH in raw mode still fires while we're inside the HTTP
            # read; pick it up on the next event so the popup resizes
            # without needing input events.
            if state.resize_pending:
                state.resize_pending = False
            if reasoning:
                state.reasoning.append(reasoning)
            if content:
                state.content.append(content)
            if tool_calls:
                state.tool_calls = tool_calls
            # Reflect what the model is currently doing.
            if state.content:
                state.status = 'streaming...'
                state.status_color = SGR_YELLOW
            elif state.tool_calls:
                state.status = 'calling tools...'
                state.status_color = SGR_CYAN
            elif state.reasoning:
                state.status = 'thinking...'
                state.status_color = SGR_MAGENTA
            _draw(screen, state)

        try:
            result = fn(selected, stream_callback=on_event, **kwargs)
            # Pure transforms never invoke the callback and finish instantly —
            # auto-dismiss so the user isn't asked to acknowledge an empty
            # popup. LLM-backed ones streamed something; keep it on screen.
            if not (state.content or state.reasoning or state.tool_calls):
                return result
            state.status = f'done — {len(result)} replacement message(s). Press any key.'
            state.status_color = SGR_GREEN
        except Exception as e:
            # Keep the streamed body visible next to the error so the user
            # can see what the model returned when parsing failed.
            state.status = f'error: {e}  Press any key.'
            state.status_color = SGR_BOLD_RED
            result = None

        state.first_draw = True
        _draw(screen, state)

        while True:
            rlist, _, _ = select.select([screen._tty_fd], [], [], 0.1)
            if rlist:
                read_key(screen._tty_fd)
                break
            if state.resize_pending:
                state.resize_pending = False
                _draw(screen, state)
    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()

    return result
