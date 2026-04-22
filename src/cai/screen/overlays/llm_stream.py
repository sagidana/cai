"""Streaming progress popup for the :messages overlay's LLM transform.

Opened from the ``!`` handler in overlays/messages.py. Drives
cai.transforms._tx_llm with a stream_callback that appends every response
chunk to an in-memory buffer and redraws, so the user sees live progress
instead of a frozen screen.
"""

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
    ansi_strip, ansi_pad, wrap_ansi,
    cur_move,
    KEY_ESC, KEY_ENTER, KEY_CTRL_C,
)
from ..input import read_key


class _StreamState:
    __slots__ = (
        'instruction', 'n_selected', 'content', 'status', 'status_color',
        'resize_pending', 'first_draw', 'prev_lines',
    )

    def __init__(self, instruction: str, n_selected: int) -> None:
        self.instruction = instruction
        self.n_selected = n_selected
        self.content: list[str] = []
        self.status = 'connecting...'
        self.status_color = SGR_DIM_GRAY
        self.resize_pending = False
        self.first_draw = True
        self.prev_lines: dict[int, str] = {}


def _draw(screen, state: _StreamState) -> None:
    rows, cols = screen._rows, screen._cols

    inner_w = max(40, int(cols * 0.9) - 2)
    box_w = inner_w + 2

    # Layout: top border | instruction | sep | body (N) | sep | status | bottom
    overhead = 6
    max_box_h = max(overhead + 3, int(rows * 0.9))
    body_h = max(3, max_box_h - overhead)
    box_h = body_h + overhead

    start_r = max(1, (rows - box_h) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)

    H = '─'
    TL, TR, BL, BR, VL, ML, MR = '┌', '┐', '└', '┘', '│', '├', '┤'
    h_line = H * inner_w

    title = '  LLM Transform  '[:inner_w]
    pad_l = max(0, (inner_w - len(title)) // 2)
    pad_r = max(0, inner_w - len(title) - pad_l)
    title_border = f'{TL}{H * pad_l}{title}{H * pad_r}{TR}'

    new_lines: dict[int, tuple] = {}

    def put(row_off: int, text: str) -> None:
        r = start_r + row_off
        if 1 <= r <= rows:
            new_lines[row_off] = (r, text)

    put(0, title_border)

    # Header: instruction + selection count
    info = (f' {SGR_BOLD}instruction:{SGR_RESET} {state.instruction}'
            f'  {SGR_DIM_GRAY}(input: {state.n_selected} msg){SGR_RESET}')
    put(1, f'{VL}{ansi_pad(info, inner_w)}{VL}')

    put(2, f'{ML}{h_line}{MR}')

    # Body: wrap accumulated content and show the tail.
    full = ''.join(state.content)
    lines = wrap_ansi(full, inner_w - 2) if full else []
    visible = lines[-body_h:] if len(lines) > body_h else lines
    for i in range(body_h):
        if i < len(visible):
            body_cell = ansi_pad(f' {visible[i]}', inner_w)
        else:
            body_cell = ' ' * inner_w
        put(3 + i, f'{VL}{body_cell}{VL}')

    put(3 + body_h, f'{ML}{h_line}{MR}')

    # Status line
    chars = len(full)
    status_text = f' {state.status_color}{state.status}{SGR_RESET}  {SGR_DIM_GRAY}({chars} chars){SGR_RESET}'
    put(3 + body_h + 1, f'{VL}{ansi_pad(status_text, inner_w)}{VL}')

    put(3 + body_h + 2, f'{BL}{h_line}{BR}')

    out: list[str] = []
    if state.first_draw:
        sys.stdout.write(ERASE_SCREEN)
        for row_off, (r, text) in new_lines.items():
            out.append(f'{cur_move(r, start_c)}{text}')
        state.first_draw = False
    else:
        for row_off, (r, text) in new_lines.items():
            if state.prev_lines.get(row_off) != text:
                out.append(f'{cur_move(r, start_c)}{text}')
    state.prev_lines = {k: v[1] for k, v in new_lines.items()}

    out.append(CUR_HIDE)
    sys.stdout.write(''.join(out))
    sys.stdout.flush()


def run_llm_transform(screen, instruction: str, model: str,
                      selected: list) -> 'list[dict] | None':
    """Run the LLM transform with a live streaming popup.

    Blocks until the response completes, then waits for a keypress. Returns
    the parsed replacement list, or None if the transform errored (the
    popup will have shown the error to the user).
    """
    from cai.transforms import get_transform

    state = _StreamState(instruction, len(selected))

    old_attrs = termios.tcgetattr(screen._tty_fd)
    orig_handler = signal.getsignal(signal.SIGWINCH)

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        state.resize_pending = True
        state.first_draw = True

    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()

    result: 'list[dict] | None' = None
    try:
        signal.signal(signal.SIGWINCH, _on_resize)
        tty.setraw(screen._tty_fd)
        state.status = 'streaming...'
        state.status_color = SGR_YELLOW
        _draw(screen, state)

        def on_chunk(chunk: str) -> None:
            # SIGWINCH in raw mode still fires while we're inside the HTTP
            # read; pick it up on the next chunk so the popup resizes
            # without needing input events.
            if state.resize_pending:
                state.resize_pending = False
            state.content.append(chunk)
            _draw(screen, state)

        try:
            spec = get_transform('llm')
            result = spec.fn(
                selected,
                instruction=instruction,
                model=model,
                stream_callback=on_chunk,
            )
            state.status = f'done — parsed {len(result)} message(s). Press any key.'
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
